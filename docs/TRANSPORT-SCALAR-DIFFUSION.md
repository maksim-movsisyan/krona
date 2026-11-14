# Scalar diffusion

`krona::transport::scalar::diffusion` implements finite-volume isotropic diffusion for one scalar on a static 3-D mesh. It provides additive matrix assembly, a matrix-free residual and an explicit nonorthogonal correction. `transport::scalar` supplies the shared system slice, source operator and assembly diagnostics. The application owns the matrix/vectors, configures a linear solver and controls result publication.

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

`Real` controls diffusion arithmetic; destination views control coefficient/output storage. Field storage, algebra `Scalar` and preconditioner `PrecScalar` are selected separately by the application. Native KRONA CSR/BSR matrices support `float` and `double`, subject to Trilinos instantiation support.

CMake targets: `krona::transport_diffusion` provides the constitutive law/coefficient providers. `krona::transport_scalar_diffusion` adds the mesh operator, scalar boundary conditions and gradient correction, without Trilinos. Native algebra assembly helpers and linear solves additionally require `krona::linalg`; the existing `krona::transport_scalar` target includes that integration.

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
namespace grad = numerics::gradient;
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
    m, conn.view(), geom.view(), comm, exec);
td::DiffusionOperator<Exec, Real> diffusion(plan);
ts::SourceOperator<Exec, Real> sources(m, exec);
ts::AssemblyStatus<Exec> status(exec);

boundary::BoundaryLayout<Exec, Real> boundary_layout(host, m, exec);
bc::Manager<Exec, Real> boundary(boundary_layout);
for (std::size_t p = 0; p < host.patches.size(); ++p)
    boundary.set(static_cast<mesh::PatchId>(p), bc::FixedValue<Real>{300});
boundary.set_unlabelled(bc::NormalGradient<Real>{0});
boundary.finalize();

const auto law = transport::IsotropicDiffusion{
    transport::UniformCoefficient<Real>{2}};
const ts::UniformSource<Real> source{0};
```

This example assigns `phi = 300` on labelled patches and zero normal gradient on unlabelled faces. Assign each patch its intended condition in a real case. Every labelled patch must be assigned, including patches with no faces on this rank; unlabelled faces also need a condition when present. A configured scalar boundary manager can be supplied instead. Call `finalize()` after setting/configuring conditions and again after any assignment changes; the shared layout must outlive the manager.

The boundary manager must reference the same mesh arrays as the diffusion plan; matching counts alone are insufficient. Diffusion contributions and `balance()` accept a final `time` argument, defaulting to `0`, for time-dependent providers/conditions. This argument does not advance a time integrator.

The plan accepts only device mesh/auxiliary views, a non-null communicator, an execution instance and options. It borrows these descriptors and has no host-mesh argument. Their partition/local numbering must match the communicator; device descriptors do not contain rank metadata, so that agreement is a caller contract. The plan requires grouped face intervals, `FaceCells`, `FaceGeometry`, `CellGeometry`, complete owned `CellFaces`, `FaceNormalDistance` and `FaceAreaOverNormalDistance`. Cell centres include real MPI ghosts; owned volumes must be finite and positive. Cached distances/area ratios must match the attached geometry.

`DiffusionOptions` defaults to `{require_orthogonal = true, orthogonality_tolerance = 1e-6}`. Orthogonality is checked for centre-to-centre inner segments and centre-to-face boundary segments: tangential length must not exceed `tolerance * normal_distance`. Tolerance must be finite and in `[0, 1)`. Setting `require_orthogonal = false` permits skew geometry; **it does not enable a correction**. Attached interpolation weights/nonorthogonality vectors are also checked for consistency. Rebuild/revalidate after changing mesh geometry or numbering.

## Choose the execution path

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

// Allocate once; the application owns the assembled system and its publication policy.
fields::FieldStorage<StorageReal, 1, Exec> phi(
    "phi", boundary.required_field_size(), fields::FieldLocation::Custom, exec);
phi.fill_async(exec, StorageReal{300});
linalg::FaceAssemblyPlan<Exec> assembly(host, comm, exec);
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
x.copy_from_component_async(phi.view().component(0), exec);
x.commit();
linalg::SolverOptions options;
options.relative_tolerance = 1e-6;
options.maximum_iterations = 500;
linalg::Solver<decltype(matrix)> linear(matrix, options);
const auto solved = linear.solve(rhs, x);
if (solved.converged) {
    x.copy_to_component_async(phi.view().component(0), exec);
    x.commit();
}
```

All contributions are **additive**. Initialize once for a fresh equation, then add all required terms before `check()` and commit. `make_system_view()` opens native ReadWrite access; it neither clears nor commits. `check()` completes diagnostics but does not close algebra access. `report.has_diffusion_anchor` reports whether this batch encountered an anchoring boundary contribution; it is not a whole-system solvability test. The operators do not impose a whole-system anchor/diagonal policy. For pure steady diffusion, a gradient/flux-only (Neumann) problem needs a reference or nullspace treatment; this example assumes suitable Dirichlet/Robin constraints. A global anchor does not prove that every disconnected component is constrained. Temporal, reaction or coupled terms may change the system's solvability conditions.

The example copies the result into the application's owned prefix only on convergence. On nonconvergence it leaves the field unchanged. Ghost and boundary slots are separate: publish them through the halo/boundary pipeline when later consumers need them. Assembling diffusion uses the condition objects directly; the initial field does not need prefilled boundary slots. Reuse the graph, matrix and vectors across solves, and choose explicitly when coefficients/RHS need rebuilding.

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

Boundary handling uses the scalar manager's affine normal-gradient relation. `FixedValue`, `NormalGradient`, `FixedFlux`, their context-dependent profile providers and custom scalar conditions are supported. Profiles are sampled at the current boundary face/time inside the diffusion kernel; no separate boundary-field fill is needed to assemble their contribution. Register each profile condition type in the manager's `CustomConditions` template list, as described in `BOUNDARY-SCALAR.md`. `FixedFlux` specifies outward **flux density**, so positive flux removes the scalar from the domain; the assembler multiplies by area. It uses the same face coefficient for flux conversion and assembly. Every relation must yield finite affine flux coefficients with a nonnegative owner slope. Fixed-value boundaries anchor the solution; prescribed gradients/fluxes contribute only to the RHS.

A custom condition is a device-copyable callable `(boundary::FaceContext<Real>, Real increment) -> bc::Relation<Real>`; add its type to `bc::Manager<Exec, Real, Custom...>`. The gradient portion determines diffusion assembly. It may use `c.time`, supplied through the operation's finite `time` argument. See `BOUNDARY-SCALAR.md` for relation fields and custom-condition examples.

For example, a space/time-dependent Dirichlet condition uses the ordinary scalar profile interface:

```cpp
struct WallValue {
    KOKKOS_INLINE_FUNCTION Real operator()(const krona::boundary::FaceContext<Real>& c) const
    { return Real{300}+Real{2}*c.y+c.time; }
};
const auto wall = bc::fixed_value<Real>(WallValue{});
bc::Manager<Exec, Real, decltype(wall)> profiled(boundary_layout);
for (std::size_t p = 0; p < host.patches.size(); ++p)
    profiled.set(static_cast<mesh::PatchId>(p), wall);
profiled.set_unlabelled(bc::NormalGradient<Real>{0});
profiled.finalize();
// Pass profiled instead of boundary to add_system_async/add_residual_async.
```

## Nonorthogonal correction

To enable corrected diffusion:

1. Build auxiliary geometry with `FaceNonOrthogonalityVector` and `FaceInterpolationWeights` in addition to the two normal metrics above.
2. Construct the plan with `DiffusionOptions{false, 1e-6}` for a nonorthogonal mesh.
3. Supply current gradients for owned **and real ghost** cells.

```cpp
// gx, gy, gz are ready rank-one cell-gradient views, each with n_cells entries.
const auto correction = grad::GradientCorrection{
    grad::cell_gradients(gx, gy, gz)};

status.begin_async();
diffusion.add_system_async(target, boundary, status, law, correction);
// Or: diffusion.add_residual_async(q, residual, boundary, status, law, correction);
(void)status.check(comm);
matrix.commit();
rhs.commit();
```

This example **adds** a corrected contribution; use it instead of an uncorrected assembly, or clear the target before rebuilding. It does not replace an earlier contribution.

`numerics::gradient::GradientCorrection` interpolates inner gradients using the actual owner weight, not an assumed midpoint. With `d = x_N - x_O`, `h = d · n`:

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

`grad::cell_gradients(gx, gy, gz)` borrows rank-one component views, including strided and mixed-precision views. Their common extent must cover all local real cells, including ghosts. For a `3*N` gradient FieldView, use `grad::cell_gradients(field, component)`; it selects `3*component + {0,1,2}`. The same correction object can be passed to the scalar boundary manager.

A custom gradient provider returns three indexable values from `gradients(cell)`. If it exposes `validate<Exec>(count)`, diffusion calls that host validator; otherwise it checks declared `memory_space` and `size()` metadata when available. Providers without metadata retain those caller contracts.

A custom correction supplies `interior(context, tx, ty, tz)` and a boundary-context call operator. It need not store a member named `gradients`. If that member exists, its metadata is validated; otherwise an optional `correction.validate<Exec>(count)` is called. Nonzero correction types still require the auxiliary nonorthogonality vectors and, when inner faces exist, interpolation weights. Use `grad::NoCorrection{}` to select the zero-correction path without those requirements.

## Diagnostics, ownership and asynchrony

`AssemblyStatus` owns a small device flag array and a host mirror. Use **one `begin_async()` → contributions → `check(comm)` batch** on the same execution instance. Flags persist across contributions: a later valid contribution cannot erase an earlier error. Host precondition failures use `mpi::require`; diffusion validation/submission is protected by `mpi::guarded`. Detected kernel errors terminate collectively during `check()`. Checks cover bad coefficients, sources, boundary relations, corrections and nonfinite/unrepresentable additions. A failed batch is not rolled back.

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
| `linalg::Solver::solve(rhs, x)` | Blocking collective linear solve; operates on committed algebra objects. Does not publish application fields. |
| `balance()` | Blocking reductions followed by an MPI all-reduce. |

All ranks participate in collective calls in the same order, including empty ranks. Order producers and consumers on the plan's execution instance; synchronize work on other instances before consuming its data. Avoid concurrent writes by separate assembly batches. Keep buffers live until queued work completes. Destroying a pending `AssemblyStatus` fences its instance but does **not** perform the collective error check.

Plans/operators store mesh/auxiliary descriptors and a non-owning communicator; they do not allocate field, face-work, matrix or vector arrays. Keep the underlying mesh/auxiliary owners and communicator alive and stable. Host storage must also remain alive when device storage borrows it on CPU. The plan does not require host metadata. Boundary layouts and algebra plans have their own setup requirements; do not replace those with a diffusion plan.

`FaceAssemblyPlan` copies share graph/entry-map allocations. `SystemView` copies descriptors and owns no independent equation buffers: retain the matrix/RHS or custom storage owners. Managed views in coefficient/source providers retain their allocations; unmanaged views borrow them. The common `grad::cell_gradients()` adapter borrows raw component storage. The application owns matrix coefficients and algebra vectors; the linear solver manages its workspace. Field/vector transfers copy the owned values and convert scalar types as needed.
