# Scalar diffusion

`krona::transport::scalar::diffusion` implements finite-volume isotropic diffusion for one scalar on a static 3-D mesh. It provides additive matrix assembly, a matrix-free residual and an explicit nonorthogonal correction. `transport::scalar` supplies the shared system slice, source operator and assembly diagnostics. An optional `SteadyDiffusion` wrapper assembles and solves a CSR system.

## Equation and signs

The steady equation is

$$
-\nabla\cdot(\Gamma\nabla\phi)=s,
\qquad q_n=-\Gamma\nabla\phi\cdot\mathbf n.
$$

Here `s` is a source **per unit volume**, and `q_n` is an outward flux **per unit face area**. The operator uses integrated cell equations:

$$
A\phi=b,\qquad r=b-A\phi=s_iV_i-\sum_f Q_{i,f}^{\mathrm{out}}.
$$

Face areas and source volumes are included once. **Neither assembly nor the residual divides by cell volume.** If an explicit update needs a rate per unit volume, divide the resulting integrated residual by `V_i` in the application.

For an uncorrected inner face, with owner `O`, neighbour `N` and unit normal from `O` to `N`:

$$
h=(\mathbf x_N-\mathbf x_O)\cdot\mathbf n>0,
\quad a_f=\Gamma_f A_f/h,
\quad Q_{O\to N}=a_f(\phi_O-\phi_N).
$$

Each owned row receives `+a_f` on its diagonal and `-a_f` in the neighbour column. Its residual receives minus its outward flux. A ghost neighbour contributes a column/input value; its row is assembled on its owning rank.

## Responsibilities

| Object | Purpose |
| --- | --- |
| `diffusion::DiffusionPlan<Exec, Real>` | Validates and borrows mesh geometry/connectivity; retains the execution instance and communicator. No algebra graph. |
| `diffusion::DiffusionOperator<Exec, Real>` | Adds diffusion to supplied system/residual buffers. Stores a copy of the borrowed plan. |
| `linalg::FaceAssemblyPlan<Exec>` | Builds and owns the shared cell-neighbour graph and face-to-native-entry maps. Reuse it across equations. |
| `scalar::SystemView` | Describes writable rank-one coefficient/RHS slices and their matching entry map. |
| `scalar::SourceOperator<Exec, Real>` | Adds `source(cell) * volume(cell)` to an owned output prefix. |
| `scalar::AssemblyStatus<Exec>` | Collects device errors for one additive batch; checks them collectively. |
| `scalar::SteadyDiffusion<Storage, Scalar, Exec, Real, PrecScalar>` | Owns a CSR matrix, RHS, solution and linear solver; borrows one application field component. |

`Real` controls diffusion arithmetic; destination views control coefficient/output storage. The convenience solver separately selects field `Storage`, algebra `Scalar` and preconditioner `PrecScalar`. Native KRONA CSR/BSR matrices support `float` and `double`, subject to Trilinos instantiation support.

## Prepare the mesh and boundary conditions

The examples assume active MPI/Kokkos and an already prepared `mesh::host::MeshPart host`. Algebra examples additionally assume an active Tpetra runtime. No mesh reader is imposed.

```cpp
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/host/aux_connectivity.hpp>
#include <krona/mesh/host/aux_geometry.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/transport/scalar/diffusion/operator.hpp>

using namespace krona;
namespace ts = transport::scalar;
namespace td = transport::scalar::diffusion;
namespace bc = boundary::scalar;
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Space = typename Exec::memory_space;

Exec exec{};
const auto comm = mpi::CommView::world(); // Must match host.
mesh::device::MeshStorage<Exec> device_mesh(
    host, {.data = mesh::device::MeshData::FiniteVolume}, exec);
const auto m = device_mesh.view();

const auto host_conn = mesh::host::build_aux_connectivity(
    host, mesh::AuxConnType::CellFaces);
const auto host_geom = mesh::host::build_aux_geometry(host,
    mesh::AuxGeomType::FaceNormalDistance |
    mesh::AuxGeomType::FaceAreaOverNormalDistance);
mesh::device::AuxConnectivityStorage<Exec> conn(host_conn, exec);
mesh::device::AuxGeometryStorage<Exec> geom(host_geom, exec);

td::DiffusionPlan<Exec, Real> plan(
    host, m, conn.view(), geom.view(), comm, exec);
td::DiffusionOperator<Exec, Real> diffusion(plan);
ts::SourceOperator<Exec, Real> sources(m, exec);
ts::AssemblyStatus<Exec> status(exec);

bc::Manager<Exec, Real> boundary(host, m, exec);
for (std::size_t p = 0; p < host.patches.size(); ++p)
    boundary.set(static_cast<mesh::PatchId>(p), bc::FixedValue<Real>{300});
boundary.set_unlabelled(bc::NormalGradient<Real>{0});

const auto law = transport::IsotropicDiffusion{
    transport::UniformCoefficient<Real>{2}};
const ts::UniformSource<Real> source{0};
```

This example assigns `phi = 300` on labelled patches and zero normal gradient on unlabelled faces. Assign each patch its intended condition in a real case. Every labelled patch must be assigned, including patches with no faces on this rank; unlabelled faces also need a condition when present. A configured scalar boundary manager can be supplied instead.

The boundary manager must reference the same mesh arrays as the diffusion plan; matching counts alone are insufficient. Diffusion contributions, `balance()` and `solve_owned()` accept a final `time` argument, defaulting to `0`, for time-dependent providers/conditions. This argument does not advance a time integrator.

The plan requires matching host/device counts, grouped face intervals, `FaceCells`, `FaceGeometry`, `CellGeometry`, complete owned `CellFaces`, `FaceNormalDistance` and `FaceAreaOverNormalDistance`. Cell centres include real MPI ghosts; owned volumes must be finite and positive. Cached distances/area ratios must match the attached geometry.

`DiffusionOptions` defaults to `{require_orthogonal = true, orthogonality_tolerance = 1e-6}`. Orthogonality is checked for centre-to-centre inner segments and centre-to-face boundary segments: tangential length must not exceed `tolerance * normal_distance`. Tolerance must be finite and in `[0, 1)`. Setting `require_orthogonal = false` permits skew geometry; **it does not enable a correction**. Attached interpolation weights/nonorthogonality vectors are also checked for consistency. Rebuild/revalidate after changing mesh geometry or numbering.

## Choose the execution path

### Convenience steady solve

```cpp
#include <krona/transport/scalar/solvers/steady_diffusion.hpp>

linalg::FaceAssemblyPlan<Exec> assembly(host, comm, exec);
fields::FieldStorage<StorageReal, 1, Exec> phi(
    "phi", boundary.required_field_size(), fields::FieldLocation::Custom, exec);
phi.fill(exec, StorageReal{300});

ts::SteadyDiffusionOptions options;
options.linear.relative_tolerance = 1e-6;
options.linear.maximum_iterations = 500;
ts::SteadyDiffusion<> solver(plan, assembly, phi.view().component(0), options);

const auto result = solver.solve_owned(boundary, law, source);
if (result.converged) {
    halo::Exchange<StorageReal, Exec> exchange(
        m, device_mesh.halo_plan(), comm, 41, exec);
    exchange.add_field(phi.view());
    exchange.prepare();
    exchange.exchange();
    boundary.apply_async(phi.view(), law);
    exec.fence("phi published");
}
```

`solve_owned()` clears/rebuilds its system, adds diffusion and the volume source, checks diagnostics, commits algebra access, copies current owned field values as the initial guess and calls the linear solver. On convergence it copies the solution back to **owned cells only**. On nonconvergence application field storage is preserved. Ghost/boundary slots and other field components are never published by this call.

The borrowed component may be strided. Owned-only storage is enough for this wrapper; the example allocates extended storage for later halo/boundary publication. Reuse a registered exchanger between solves when needed.

The wrapper requires a positive assembled diagonal in every owned row and at least one global diffusion anchor for a nonempty system. A fixed value or a suitable Robin condition supplies an anchor; gradient/flux-only conditions do not. Pure Neumann problems are rejected by this wrapper: it does not add a reference or handle a nullspace. One global anchor does not establish solvability of disconnected components; each component needs appropriate constraints.

`matrix()`, `rhs()`, `solution()`, `assembly_plan()`, `operation()` and `component()` expose the associated objects. `linear_solver()` gives mutable access to the linear solver. Each `solve_owned()` performs a full rebuild; use the lower-level path to retain a fixed matrix across correction iterations.

### Assemble into your own CSR or BSR matrix

`make_system_view()` supports:

| Matrix | Selected scalar slice |
| --- | --- |
| `linalg::CsrMatrix<Scalar, Exec>` | All native scalar entries; `component` must be `0`. |
| `linalg::BsrMatrix<Scalar, Exec>` | Entries `(block, component, component)` and RHS DOFs `cell * block_size + component`. Other block components are untouched. |

**The matrix must share the exact native graph object with `assembly.pattern()`.** An independently rebuilt equivalent graph is rejected. The RHS must share the matrix's exact point-distribution object. The assembly plan must use the diffusion mesh's local cell/face numbering; this semantic agreement is a caller contract.

```cpp
#include <krona/transport/scalar/assembly.hpp>
#include <krona/linalg/solver.hpp>

// Reuse assembly from above, or construct it once for this mesh.
linalg::CsrMatrix<SolverReal, Exec> matrix(assembly.pattern());
linalg::Vector<SolverReal, Exec> rhs(matrix.distribution());
const auto target = ts::make_system_view(matrix, rhs, assembly, exec);
Kokkos::deep_copy(exec, target.values, SolverReal{0});
Kokkos::deep_copy(exec, target.rhs, SolverReal{0});

status.begin_async();
diffusion.add_system_async(target, boundary, status, law);
sources.add_async(target.rhs, status, source);
const auto report = status.check(comm);
matrix.commit();
rhs.commit();

linalg::Vector<SolverReal, Exec> x(matrix.distribution());
x.copy_from_component(phi.view().component(0), exec);
x.commit();
linalg::Solver<decltype(matrix)> linear(matrix, options.linear);
const auto solved = linear.solve(rhs, x);
if (solved.converged) {
    x.copy_to_component(phi.view().component(0), exec);
    x.commit();
}
```

All contributions are **additive**. Initialize once for a fresh equation, then add all required terms before `check()` and commit. `make_system_view()` opens native ReadWrite access; it neither clears nor commits. `check()` completes diagnostics but does not close algebra access. `report.has_diffusion_anchor` reports whether this batch encountered an anchoring boundary contribution; it is not a whole-system solvability test. The low-level path does not impose the convenience solver's anchor/diagonal checks.

For a block system:

```cpp
#include <krona/linalg/bsr_matrix.hpp>

linalg::BsrMatrix<SolverReal, Exec> blocks(assembly.pattern(), 3);
linalg::Vector<SolverReal, Exec> block_rhs(blocks.distribution());
Kokkos::deep_copy(exec, blocks.overwrite_blocks(exec), SolverReal{0});
Kokkos::deep_copy(exec, block_rhs.overwrite(exec), SolverReal{0});
const auto equation = ts::make_system_view(blocks, block_rhs, assembly, exec, 1);

status.begin_async();
diffusion.add_system_async(equation, boundary, status, law);
sources.add_async(equation.rhs, status, source);
(void)status.check(comm);
blocks.commit();
block_rhs.commit();
// Only equation/component 1 has been assembled. Fill the other equations before solving.
```

The helper does not create cross-component couplings. To target another equation/variable pair or another backend, construct `SystemView{values, rhs, entries}` directly. Raw Tpetra matrices are not accepted by `make_system_view()`; adapt their storage explicitly if required.

A direct `SystemView` needs writable, accessible, floating-point rank-one Kokkos views, including strided slices. `rhs` must have **exactly** `n_own` entries. The entry-map members are `diagonal` (`n_own`), `owner_entries` (`n_inner_faces`) and `neighbour_entries` (`n_internal_faces`); they index the supplied coefficient slice in native entry order. All mapped positions must exist. A diagonal maps an owned cell; owner/neighbour maps address the respective off-diagonal row/column connection. Repeated connecting faces may map to the same entry and their contributions accumulate.

`FaceAssemblyPlan` omits the original-input permutation. Use native access and entry maps; original-order `set_values*()` is unavailable for this pattern.

### Matrix-free integrated residual

No `FaceAssemblyPlan`, matrix, algebra vector or linear solver is required for this path.

```cpp
// q already contains current owned and real MPI ghost values.
Kokkos::View<Real*, Space> q("q", static_cast<std::size_t>(host.n_cells));
Kokkos::View<Real*, Space> residual("r", static_cast<std::size_t>(host.n_own));
Kokkos::deep_copy(exec, q, Real{300}); // Constant example: no ghost exchange needed.
Kokkos::deep_copy(exec, residual, Real{0});

status.begin_async();
diffusion.add_residual_async(q, residual, boundary, status, law);
sources.add_async(residual, status, source);
(void)status.check(comm);
```

`add_residual_async()` adds `b_boundary - A_diffusion * q`; sources are separate. Input needs at least `n_cells` entries, including current real ghosts. Output needs at least `n_own`; only its owned prefix is changed. Extended boundary samples are not read: boundary fluxes come directly from the manager's conditions. Consequently boundary-value filling is unnecessary for this operation itself.

Input, gradients/coefficient buffers and written output must not overlap. In particular, in-place residual evaluation is unsupported; the runtime only detects equal starting pointers, so avoiding offset/strided aliases is also the caller's responsibility.

## Coefficients, sources and boundary relations

Pass `transport::IsotropicDiffusion{provider}`. Although the standalone constitutive law permits zero conductivity in its forward operation, **this spatial diffusion operator requires finite, strictly positive coefficients on every processed inner/boundary face**. Transmissibilities must remain positive and representable in the target scalar type.

| Coefficient provider | Input and requirements |
| --- | --- |
| `UniformCoefficient<Real>{gamma}` | Constant face coefficient; default is `1`. |
| `FaceCoefficient{values}` | Rank-one array indexed by local face ID; at least `n_faces` entries. |
| `InterpolatedCellCoefficient{values}` | Cell coefficients including real ghosts, at least `n_cells` entries. Requires `FaceInterpolationWeights` when inner faces exist. Default interpolation is harmonic; `ArithmeticCoefficient` can be selected explicitly. Boundary faces use the owner coefficient. |
| `OwnerCellCoefficient{values}` | Exterior-only owner extrapolation. This operator rejects it if any inner faces exist. |
| Custom callable | Device-callable `operator()(const Context&) const` returning the face coefficient. Storage/accessibility and parameter validation are the application's responsibility. |

Inner contexts provide `face`, `owner`, `neighbour`, face coordinates `x/y/z`, unit normal `nx/ny/nz`, `distance`, `area`, `time` and `owner_weight`. Orientation stays owner-to-neighbour even while assembling the neighbour row. Boundary contexts are `boundary::FaceContext<Real>`: owner-outward geometry, boundary/extended indices and tangential offset `tx/ty/tz`; they have no `neighbour` member. A custom provider must compile for both context types and produce consistent shared-face values across ranks. Do not assume `owner_weight` is available unless the auxiliary weights were attached.

For example, a spatially varying coefficient needs no coefficient array. Define the provider at namespace scope:

```cpp
struct SpatialCoefficient {
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& c) const
    { return Real{2} + Real{0.1} * Kokkos::sin(c.x); }
};
const auto spatial_law = transport::IsotropicDiffusion{SpatialCoefficient{}};
```

Providers are sampled during each operation, not cached in the plan. Assembly freezes their current values: no derivatives of a nonlinear coefficient model are included. Update coefficients and repeat solves externally for a nonlinear problem. See `TRANSPORT-PROPERTIES.md` for transport-property models and coefficient interpolation formulas.

`UniformSource<Real>{s}` supplies a constant volume-source density; `CellSource{view}` reads a rank-one array with at least `n_own` entries. A custom device-callable `source(cell)` can evaluate it on demand. Sources must be finite and their integrated contributions representable. Positive sources add to the RHS/residual; no source diagonal or nonlinear derivative is assembled.

Boundary handling uses the scalar manager's affine normal-gradient relation. `FixedValue`, `NormalGradient`, `FixedFlux` and custom scalar conditions are supported. `FixedFlux` specifies outward **flux density**, so positive flux removes the scalar from the domain; the assembler multiplies by area. It uses the same face coefficient for flux conversion and assembly. Every relation must yield finite affine flux coefficients with a nonnegative owner slope. Fixed-value boundaries anchor the solution; prescribed gradients/fluxes contribute only to the RHS.

A custom condition is a device-copyable callable `(boundary::FaceContext<Real>, Real increment) -> bc::Relation<Real>`; add its type to `bc::Manager<Exec, Real, Custom...>`. The gradient portion determines diffusion assembly. It may use `c.time`, supplied through the operation's finite `time` argument. See `BOUNDARY-SCALAR.md` for relation fields and custom-condition examples.

## Nonorthogonal correction

To enable corrected diffusion:

1. Build auxiliary geometry with `FaceNonOrthogonalityVector` and `FaceInterpolationWeights` in addition to the two normal metrics above.
2. Construct the plan with `DiffusionOptions{false, 1e-6}` for a nonorthogonal mesh.
3. Supply current gradients for owned **and real ghost** cells.

```cpp
// gx, gy, gz are ready rank-one cell-gradient views, each with n_cells entries.
const auto correction = td::NonOrthogonalCorrection{
    td::CellGradients{gx, gy, gz}};

status.begin_async();
diffusion.add_system_async(target, boundary, status, law, correction);
// Or: diffusion.add_residual_async(q, residual, boundary, status, law, correction);
(void)status.check(comm);
matrix.commit();
rhs.commit();
```

This example **adds** a corrected contribution; use it instead of an uncorrected assembly, or clear the target before rebuilding. It does not replace an earlier contribution.

`NonOrthogonalCorrection` interpolates inner gradients using the actual owner weight, not an assumed midpoint. With `d = x_N - x_O`, `h = d · n`:

$$
\mathbf T_f=A_f(\mathbf n-\mathbf d/h),\qquad
\mathbf g_f=w\mathbf g_O+(1-w)\mathbf g_N,
\qquad Q_{O\to N}=a_f(\phi_O-\phi_N)-\Gamma_f\mathbf T_f\cdot\mathbf g_f.
$$

The gradient contribution is explicit: built-in boundary conditions and the inner stencil retain the uncorrected matrix, while the RHS changes. At boundaries, the same provider returns the reconstruction increment `(x_face - x_owner - h*n) · gradient_owner` used by scalar boundary relations. These boundary increments and the integrated inner correction have different geometric factors.

The operator does not calculate gradients, fill boundary samples, exchange halos or run outer iterations. A typical outer iteration publishes field ghosts/boundary samples, computes owned gradients with the gradient module, exchanges gradient ghosts, then assembles the corrected RHS and solves. Initialize gradients before their first use; use consistent corrections when publishing boundary samples and evaluating diffusion. Linear-solver convergence for one frozen gradient does not establish convergence of this outer iteration.

For a fixed matrix, first assemble an uncorrected base system and save its RHS. Each outer iteration restores that base RHS and calls:

```cpp
const auto b = Kokkos::subview(rhs.overwrite(exec), Kokkos::ALL(), 0);
Kokkos::deep_copy(exec, b, base_rhs); // Saved zero-correction RHS, including sources.
status.begin_async();
diffusion.add_correction_async(b, boundary, status, law, correction);
(void)status.check(comm);
rhs.commit();
// Solve with the existing committed matrix, then update gradients externally.
```

`add_correction_async()` adds **corrected RHS minus zero-correction RHS**. It never edits the matrix or re-adds prescribed base fluxes/sources. Do not accumulate corrections from previous iterations. Coefficients, conditions and evaluation time must match the frozen base system. Custom boundary matrix slopes must not depend on the correction increment; this is checked when using the correction-only path. If matrix coefficients change, rebuild the base system instead.

`CellGradients{x,y,z}` accepts strided component views without copying. A custom gradient provider can return three indexable components from `gradients(cell)`; its extents/accessibility are not automatically checked. A custom correction must expose `.gradients`, an `interior(context, tx, ty, tz)` method and a boundary-context call operator. The same auxiliary-vector/weight requirements still apply.

## Diagnostics, ownership and asynchrony

`AssemblyStatus` owns a small device flag array and a host mirror. Use **one `begin_async()` → contributions → `check(comm)` batch** on the same execution instance. Flags persist across contributions: a later valid contribution cannot erase an earlier error. Host precondition failures use `mpi::require`; detected kernel errors terminate collectively during `check()`. Checks cover bad coefficients, sources, boundary relations, corrections and nonfinite/unrepresentable additions. A failed batch is not rolled back.

`balance(field, boundary, law, source[, correction], time)` is an optional blocking collective diagnostic. It returns global `source_integral`, `outward_boundary_flux` and `defect = outward_boundary_flux - source_integral`. It integrates owned sources and physical boundary fluxes; inner fluxes are not summed. Only owned field values are needed, plus any correction/coefficient inputs. It is a global conservation check, not a residual norm or a proof of local convergence. Pass the same law/source/correction/time as the equation being checked.

| Operation | Completion and communication |
| --- | --- |
| `DiffusionPlan` construction | Blocking local geometry validation; no MPI collective. |
| `FaceAssemblyPlan` construction | Blocking and collective; builds numbering, graph and entry maps. |
| `make_system_view()` | Local native access acquisition; may synchronize native host/device data. Does not commit. |
| `AssemblyStatus::begin_async()` | Queues flag reset; no collective. Does not clear equation arrays. |
| `add_system_async()`, `add_residual_async()`, `add_correction_async()`, `SourceOperator::add_async()` | Queue local kernels; no fence, halo exchange or MPI collective. |
| `AssemblyStatus::check(comm)` | Copies flags to host, fences the execution instance and performs an MPI all-reduce. Does not commit algebra wrappers. |
| Matrix/vector `commit()` | Completes pending native access on its recorded instance and closes the access batch; no MPI collective. |
| `SteadyDiffusion::solve_owned()` | Blocking, collective assembly/check/linear solve/publication of owned values. |
| `balance()` | Blocking reductions followed by an MPI all-reduce. |

All ranks participate in collective calls in the same order, including empty ranks. Order producers and consumers on the plan's execution instance; synchronize work on other instances before consuming its data. Avoid concurrent writes by separate assembly batches. Keep buffers live until queued work completes. Destroying a pending `AssemblyStatus` fences its instance but does **not** perform the collective error check.

Plans/operators store mesh/auxiliary descriptors and a non-owning communicator; they do not allocate field, face-work, matrix or vector arrays. Keep the underlying mesh/auxiliary owners and communicator alive and stable. Host storage must also remain alive when device storage borrows it on CPU. The host mesh argument is used during plan validation, not retained as a separate host-mesh object.

`FaceAssemblyPlan` copies share graph/entry-map allocations. `SystemView` copies descriptors and owns no independent equation buffers: retain the matrix/RHS or custom storage owners. Managed views in coefficient/gradient/source providers retain their allocations; unmanaged views borrow them. `SteadyDiffusion` additionally allocates its matrix coefficients, algebra vectors, status storage and solver workspace, and copies owned field values into/out of its algebra solution vector. It shares the supplied assembly plan and borrows the application component.
