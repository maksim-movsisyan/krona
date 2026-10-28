# Scalar diffusion

KRONA provides a cell-centred finite-volume discretization of isotropic scalar diffusion:

$$
-\nabla\cdot(D\nabla\phi)=s,\qquad q_n=-D\,\partial_n\phi.
$$

The spatial operator can add its contribution to an existing algebraic system or evaluate its residual directly from a field. It supports an optional explicit nonorthogonal correction using externally supplied gradients. The convenience `SteadyDiffusion` solver builds and solves a complete steady diffusion equation. Convection and time derivatives are not implemented here yet.

The shared convention is **cell-integrated equations and residuals**:

$$
A\phi=b,\qquad r=b-A\phi,\qquad b_i^{\rm source}=s_iV_i.
$$

Sources are densities per unit volume; prescribed outward fluxes are densities per unit area. Neither RHS entries nor residuals are divided by cell volume. A positive source increases both `b` and `r`; an outward diffusive flux decreases `r`.

## Objects, namespaces and ownership

The examples use these aliases:

```cpp
namespace ts = krona::transport::scalar;
namespace td = krona::transport::scalar::diffusion;
namespace bc = krona::boundary::scalar;
```

Header paths below are relative to `include/krona/`.

| Object / header | Responsibility |
| --- | --- |
| `linalg::FaceAssemblyPlan<Exec>` / `linalg/face_assembly.hpp` | Shared cell-neighbour graph and face-to-native-entry map for CSR and BSR assembly. Independent of the diffusion geometry plan. |
| `td::DiffusionPlan<Exec, Real>` / `transport/scalar/diffusion/plan.hpp` | Validated borrowed mesh, cell-face connectivity, geometry and communicator. Allocates no algebra graph, matrix or vector. |
| `td::DiffusionOperator<Exec, Real>` / `transport/scalar/diffusion/operator.hpp` | Adds integrated diffusion coefficients or a matrix-free residual. Stores a copy of the borrowed plan; allocates no coefficient or work arrays. |
| `td::CellGradients`, `td::NonOrthogonalCorrection`, `td::NoCorrection` / `transport/scalar/diffusion/correction.hpp` | Borrowed gradient adapter and explicit correction policy. No gradient construction, halo exchange or work-array allocation. `NoCorrection` is the default. |
| `ts::SystemView<Values, Rhs, Entries>` / `transport/scalar/system_view.hpp` | A descriptor containing writable rank-one coefficient/RHS views and a face-entry map. |
| `ts::make_system_view(...)` / `transport/scalar/assembly.hpp` | Opens compatible native CSR or selected BSR-component views without allocating another coefficient array. |
| `ts::AssemblyStatus<Exec>` / `transport/scalar/status.hpp` | Reusable device flags and host buffer for a complete batch of contributions. `check(comm)` completes work and performs collective diagnostics. |
| `ts::SourceOperator<Exec, Real>`, `UniformSource`, `CellSource` / `transport/scalar/source.hpp` | Independent additive volume source and its providers. The operator borrows owned-cell volumes. |
| `ts::SteadyDiffusion<Storage, Scalar, Exec, Real, PrecScalar>` / `transport/scalar/solvers/steady_diffusion.hpp` | Owns a scalar CSR matrix, RHS, solution, linear solver and assembly status. Shares the algebra plan and borrows one application-field component. |
| `transport::IsotropicDiffusion<Coefficient>` / `transport/diffusion/isotropic.hpp` | Shared physical law used by the spatial operator and scalar boundary conditions. Its namespace remains `krona::transport`. |
| `td::DiffusionOptions`, `td::BalanceResult` / `transport/scalar/diffusion/types.hpp` | Geometry-validation options and integrated balance diagnostics. |

Common assembly objects and sources remain in `transport::scalar`; diffusion-specific numerical types live in `transport::scalar::diffusion`. The ready-made steady solver remains in `transport::scalar`.

The low-level diffusion/residual and source headers do not require Tpetra. Include the algebra adapter or steady solver when using KRONA's linear-algebra wrappers. MPI and Kokkos must be active; initialize Tpetra as well for the algebra/solver path. Destroy all dependent objects before finalizing their runtimes.

Plan copies copy descriptors, not mesh arrays. A shared `FaceAssemblyPlan` allows several matrices to reuse one graph and entry map while retaining independent coefficients. Keep mesh/auxiliary owners, source/coefficient/gradient backing buffers, borrowed field allocations and communicators alive until their users and submitted work finish. Host containers must remain alive whenever compute views borrow their buffers.

## Mesh and geometry contracts

The host and compute mesh must use the same topology, local numbering and communicator. Faces are grouped into owned/owned, owned/ghost and physical-boundary intervals. Required data are:

- Face-cell connectivity and face geometry; cell centres for owned and real MPI ghost cells; positive owned-cell volumes.
- Complete `AuxConnType::CellFaces` incidence for owned cells.
- `AuxGeomType::FaceNormalDistance` and `AuxGeomType::FaceAreaOverNormalDistance` for all faces.
- `AuxGeomType::FaceInterpolationWeights` when using interpolated cell coefficients or the nonorthogonal correction on inner faces.
- `AuxGeomType::FaceNonOrthogonalityVector` for all faces when using the correction.

Normals are unit vectors directed from the canonical owner to the neighbour; physical-boundary normals point out of the domain. Cached normal distances must be positive and consistent with coordinates. Area/distance factors, any attached interpolation weights and any attached nonorthogonality vectors are checked against geometry. Attached vectors are validated even if a later call does not request the correction.

`td::DiffusionOptions` defaults to `require_orthogonal = true` and `orthogonality_tolerance = 1e-6`. The check compares tangential centre separation with its normal projection. At a physical boundary, the separation is from the owner centre to the face centre.

For a nonorthogonal mesh, set `require_orthogonal = false` when constructing the plan and explicitly pass a correction to the operations below. Disabling the check or attaching the vectors alone does not enable the correction. Normal distances and area/distance factors remain mandatory in both modes; computing them on demand is not implemented.

The scalar boundary manager must borrow the **same compute mesh buffers** as the diffusion plan. Configure every labelled patch, including patches empty on a particular rank, and handle unlabelled boundary faces explicitly when present.

The matrix-free input needs the real-cell prefix of size `n_cells`: owned cells followed by MPI ghosts. It does not read boundary slots. A field used with boundary `apply_async()` needs the extended size

```text
n_cells + n_faces - n_inner_faces
```

where the tail stores one face-centroid value per physical boundary face. The steady solver itself only requires a writable component covering owned cells; allocate the extended field when downstream halo/boundary operations need it.

Topology and geometry are static while a diffusion plan is in use. After changing geometry, complete pending work, refresh the underlying geometry/metrics and revalidate by constructing a new diffusion plan. Connectivity or numbering changes also invalidate algebra graphs and face-entry maps. A geometry-only update does not require rebuilding an unchanged algebra graph/map.

## Constitutive law and spatial discretization

Use the same `transport::IsotropicDiffusion` law for interior diffusion and boundary flux conversion. A face-coefficient array is optional:

| Coefficient provider | Data and behaviour |
| --- | --- |
| `UniformCoefficient<Real>{D}` | Constant coefficient; the provider itself needs no coefficient array or weights. |
| `FaceCoefficient{view}` | Rank-one face-indexed Kokkos View covering all local faces; the provider itself needs no weights. |
| `InterpolatedCellCoefficient{view}` | Rank-one View covering owned and MPI ghost cells; harmonic interpolation by default. Refresh coefficient ghosts first. Inner faces require attached weights; physical boundaries use the owner value. |
| `InterpolatedCellCoefficient<View, ArithmeticCoefficient>` | The same storage/weight contract, with arithmetic interpolation. |
| `OwnerCellCoefficient{view}` | Owner extrapolation for exterior faces. The diffusion operator rejects this provider when inner faces exist. |
| Custom provider | Device-callable `coefficient(context)`; may compute a model directly or read existing fields. The caller guarantees accessible, correctly sized backing storage. |

These provider names are in `krona::transport`. Every sampled diffusion coefficient must be finite and strictly positive, including both cell values used by an interpolating provider. **Zero diffusion is currently rejected.** Coefficients are frozen at provider values during assembly: no derivatives of nonlinear coefficient models are inserted into the matrix.

For projected owner-to-face and face-to-neighbour distances `d_P` and `d_N`, the existing owner interpolation weight is `w = d_N/(d_P+d_N)`:

$$
D_f^{\rm arith}=wD_P+(1-w)D_N,\qquad
D_f^{\rm harm}=\frac{1}{(1-w)/D_P+w/D_N}.
$$

The harmonic expression follows from series resistances `d_P/D_P + d_N/D_N`; its distance fractions therefore appear in the opposite positions to arithmetic interpolation. Missing geometric weights are not replaced by `0.5`.

With `d = x_N-x_P` and `d_n = d\cdot n_f > 0`, the default two-point integrated owner-outward flux is

$$
Q_P=-D_f\frac{A_f}{d_n}(\phi_N-\phi_P),\qquad a_f=D_fA_f/d_n>0.
$$

An owned row receives `+a_f` on its diagonal and `-a_f` at the neighbour entry. Multiple connecting faces accumulate into the same graph entry. One kernel iteration owns one row, so interior assembly needs no atomics. Boundary faces sharing an owner use atomic additions.

Calls without a correction use `td::NoCorrection`: they do not read gradients and pass a zero reconstruction increment to boundary relations. The optional corrected path is described below.

Built-in scalar conditions are `FixedValue`, `NormalGradient` and `FixedFlux`; registered custom conditions return a scalar `Relation`, allowing affine closures such as Robin conditions. Assembly converts the relation into integrated outward flux `Q = a*phi_owner + c`, requiring finite coefficients and `a >= 0`. It adds `+a` to the diagonal and `-c` to the RHS. `FixedFlux` is converted using the supplied physical law.

## Shared setup and a complete steady solve

This example assumes an orthogonal mesh with exactly two labelled patches, IDs `0` and `1`, and no unlabelled boundary faces. `host_mesh`, `mesh`, `connectivity`, `geometry`, `comm` and `exec` are already prepared and remain alive. Attach mesh halo data if using the exchange below.

```cpp
#include <krona/transport/scalar/solvers/steady_diffusion.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>

using namespace krona;
namespace ts = transport::scalar;
namespace td = transport::scalar::diffusion;
namespace bc = boundary::scalar;
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;

// Separate plans: borrowed spatial data and shared algebra graph/entry map.
td::DiffusionPlan<Exec, Real> plan(
    host_mesh, mesh.view(), connectivity.view(), geometry.view(), comm, exec);
linalg::FaceAssemblyPlan<Exec> shared(host_mesh, comm, exec);

bc::Manager<Exec, Real> boundary(host_mesh, mesh.view(), exec);
boundary.set(0, bc::FixedValue<Real>{0});
boundary.set(1, bc::FixedValue<Real>{1});

fields::FieldStorage<StorageReal, 1, Exec> phi(
    "phi", boundary.required_field_size(), fields::FieldLocation::Custom, exec);
phi.fill(exec, StorageReal{0}); // Initial owned-cell guess and initialized tail.

ts::SteadyDiffusionOptions options;
options.linear.method = "GMRES";
options.linear.preconditioner.type = "AMG"; // Already the defaults.
ts::SteadyDiffusion<> solver(plan, shared, phi.view().component(0), options);

halo::Exchange<StorageReal, Exec> exchange(
    mesh.view(), mesh.halo_plan(), comm, 17, exec, halo::Transfer::HostStaged);
exchange.add_field(phi.view());
exchange.prepare(); // Reuse this exchange; register other fields before prepare().

const auto law = transport::IsotropicDiffusion{
    transport::UniformCoefficient<Real>{2}};
const ts::UniformSource<Real> source{0};
const auto result = solver.solve_owned(boundary, law, source);
if (result.converged) {
    exchange.exchange(); // Blocking publication to application MPI ghosts.
    boundary.apply_async(phi.view(), law); // Boundary-face slots on boundary's exec.
    exec.fence("Scalar field ready for host access or another execution instance");
}
```

Every rank participates in algebra-plan construction and collective solves, including empty ranks. The diffusion plan performs blocking local validation; it no longer constructs an algebra plan implicitly.

`solve_owned(boundary, law, source, time)` clears its own matrix/RHS once, adds diffusion and source, checks the complete steady system, commits the wrappers and solves. The current owned field is the initial guess. On convergence it completes the owned-cell copy back before returning. On nonconvergence the application field stays unchanged; `solution()` retains the latest algebraic iterate.

Application ghosts and boundary slots are not published by the solve. Tpetra imports its own remote vector entries, independently of KRONA's field halos. Several steady solvers can share the plans and publish their fields through one grouped exchange afterwards.

The borrowed field component may be strided. It must be writable, accessible from `Exec` and use the selected `Storage` type. Template defaults are `StorageReal`, `SolverReal`, `execution::ExecutionSpace`, `ComputeReal` and `PreconditionerReal`. Different solver/preconditioner scalar types introduce the conversion storage and operator-adapter costs described in `LINALG.md`; matching types avoid that precision adapter.

## Adding diffusion to an external system

The low-level operator owns neither a matrix nor an RHS. The following continues from the setup above but creates a separate externally owned system:

```cpp
#include <krona/transport/scalar/assembly.hpp>
#include <krona/linalg/solver.hpp>

linalg::CsrMatrix<SolverReal, Exec> matrix(shared.pattern());
linalg::Vector<SolverReal, Exec> rhs(matrix.distribution());
linalg::Vector<SolverReal, Exec> solution(matrix.distribution());
linalg::Solver<decltype(matrix)> linear(matrix);
td::DiffusionOperator<Exec, Real> diffusion(plan);
ts::SourceOperator<Exec, Real> sources(mesh.view(), exec);
ts::AssemblyStatus<Exec> status(exec); // Allocate once; reuse for later batches.

const auto target = ts::make_system_view(matrix, rhs, shared, exec);
Kokkos::deep_copy(exec, target.values, SolverReal{0});
Kokkos::deep_copy(exec, target.rhs, SolverReal{0});

status.begin_async();
diffusion.add_system_async(target, boundary, status, law);
sources.add_async(target.rhs, status, source);
// Other compatible contributions can be added here, on the same exec instance.
const auto report = status.check(comm); // Blocking collective diagnostics.
(void)report;
matrix.commit();
rhs.commit();

solution.copy_from_component(phi.view().component(0), exec);
solution.commit();
const auto result = linear.solve(rhs, solution);
if (result.converged) {
    solution.copy_to_component(phi.view().component(0), exec);
    solution.commit();
}
```

`add_system_async(target, boundary, status, law, time)` preserves existing coefficients and RHS entries. Clear the complete system **once**, outside the contributing operators, then add every term. Repeating an additive call adds its contribution again. Source terms are separate and are never implicitly included by the diffusion call.

`SystemView` contains `values`, `rhs` and `entries`. Its entry map provides `diagonal(cell)`, `owner_entries(face)` and `neighbour_entries(face)`; the last applies to faces between owned cells. These are native matrix-entry positions, so kernels do not search matrix rows.

`make_system_view` requires the exact shared native graph and the matrix's exact point-distribution object for the RHS. Matching dimensions alone are insufficient. It also checks the component index. The application must ensure that the assembly map and diffusion plan describe the same local mesh numbering. Native ReadWrite access can synchronize Tpetra's existing host/device storage, but the adapter allocates no additional coefficient or packing array.

Unlike `SteadyDiffusion`, this low-level path does not impose positive whole-system diagonals, a value anchor or general solvability. The application owns these complete-system decisions.

### Selecting a scalar equation in BSR storage

The same face-entry map works for a BSR graph with identical cell numbering. For example, assemble the third scalar equation/variable component in blocks of width three:

```cpp
#include <krona/linalg/bsr_matrix.hpp>

linalg::BsrMatrix<SolverReal, Exec> block_matrix(shared.pattern(), 3);
linalg::Vector<SolverReal, Exec> block_rhs(block_matrix.distribution());
Kokkos::deep_copy(exec, block_matrix.overwrite_blocks(exec), SolverReal{0});
Kokkos::deep_copy(exec, block_rhs.overwrite(exec), SolverReal{0});

const auto slice = ts::make_system_view(block_matrix, block_rhs, shared, exec, 2);
status.begin_async();
diffusion.add_system_async(slice, boundary, status, law);
sources.add_async(slice.rhs, status, source);
(void)status.check(comm);
block_matrix.commit();
block_rhs.commit();
```

The selected coefficient view addresses `(native_block_entry, 2, 2)` with the correct strides; the RHS slice addresses cell-major scalar DOFs `cell*3 + 2`. Other components and cross-component coefficients are preserved. Here they were initialized to zero; this snippet fills only one equation component, so the remaining equations must be assembled before solving the complete block system. This is scalar contribution assembly into BSR storage, not a general coupled diffusion law.

## Matrix-free residual

With current real-cell ghost values, compute the integrated residual without constructing a graph, matrix or Tpetra vector:

```cpp
#include <krona/transport/scalar/diffusion/operator.hpp>

using Space = typename Exec::memory_space;
Kokkos::View<Real*, Space> residual("residual", host_mesh.n_own);
td::DiffusionOperator<Exec, Real> diffusion(plan);
ts::SourceOperator<Exec, Real> sources(mesh.view(), exec);
ts::AssemblyStatus<Exec> status(exec);

// Refresh phi's MPI ghosts before submission if its owned values have changed.
Kokkos::deep_copy(exec, residual, Real{0});
status.begin_async();
diffusion.add_residual_async(phi.view().component(0), residual, boundary, status, law);
sources.add_async(residual, status, source);
(void)status.check(comm);
// residual now contains b_diffusion + b_source - A_diffusion*phi.
```

The diffusion call adds only `b_boundary - A_diffusion*phi`. Source integration is a separate contribution. Existing residual values are preserved; zero them only when starting a new complete residual.

The input must cover `n_cells`; output must cover owned cells. Both may be strided native rank-one Kokkos Views. Only the owned output prefix is modified, and extended boundary values are not read. **Input and output must not overlap**, including partially overlapping/strided aliases. The implementation rejects identical base pointers for nonempty owned ranges; detecting every possible overlapping alias remains the caller's responsibility. Pure Neumann diffusion is allowed in this path.

## Nonorthogonal correction

### Interior flux and boundary closure

Let `g_P` and `g_N` be supplied cell gradients. With the same geometric owner weight used above,

$$
g_f=w g_P+(1-w)g_N,\qquad
T_f=A_f\left(n_f-\frac{d}{d_n}\right).
$$

The cached `face_non_ortho_*` components store `T_f`. For unit normals the corrected integrated flux is

$$
Q_P=-D_f\left[\frac{A_f}{d_n}(\phi_N-\phi_P)+T_f\cdot g_f\right].
$$

The two-point part remains implicit. The correction is explicit in the supplied gradients: it adds `+D_f*T_f·g_f` to the owner's RHS/residual and the opposite contribution to an owned neighbour. `T_f` already contains face area; do not multiply this term by area again. The canonical owner-to-neighbour orientation is retained when assembling the neighbour row. Matching coefficient and gradient data across MPI interfaces are required for conservation.

At a physical boundary the correction uses a different geometric quantity:

$$
d_b=x_f-x_P,\qquad h_b=d_b\cdot n_f,\qquad
t_b=d_b-h_bn_f,\qquad \delta_b=t_b\cdot g_P.
$$

`NonOrthogonalCorrection::operator()(context)` returns this reconstruction increment `delta_b`. The operator passes it to `boundary::scalar::make_relation` and obtains the outward flux from the resulting normal derivative. For example, `FixedValue{phi_B}` gives

$$
\partial_n\phi=\frac{\phi_B-\phi_P-\delta_b}{h_b}.
$$

The correction is therefore included through the boundary relation; no extra interior-style `T_f` term is added at that boundary. `NormalGradient` and `FixedFlux` keep their prescribed normal derivative/flux; the increment affects their reconstructed face value instead.

### Preparing borrowed gradients

Before constructing the corrected plan, build and attach `FaceNormalDistance`, `FaceAreaOverNormalDistance`, `FaceNonOrthogonalityVector` and, if inner faces exist, `FaceInterpolationWeights`. Weights are needed to interpolate gradients even for uniform diffusion.

The following setup uses the same mesh/communicator objects as the earlier examples, with those caches present:

```cpp
#include <krona/transport/scalar/diffusion/correction.hpp>

td::DiffusionOptions spatial_options;
spatial_options.require_orthogonal = false;
td::DiffusionPlan<Exec, Real> corrected_plan(
    host_mesh, mesh.view(), connectivity.view(), geometry.view(),
    comm, exec, spatial_options);
td::DiffusionOperator<Exec, Real> corrected_diffusion(corrected_plan);

fields::FieldStorage<Real, 3, Exec> grad_phi(
    "grad_phi", host_mesh.n_cells, fields::FieldLocation::Custom, exec);
const auto correction = td::NonOrthogonalCorrection{td::CellGradients{
    grad_phi.view().component(0),
    grad_phi.view().component(1),
    grad_phi.view().component(2)}};
```

This only creates storage and descriptors. **Populate the gradients and refresh their real MPI ghosts before submitting a corrected operation.** The diffusion module does neither. It is independent of the gradient algorithm and does not include the gradient module. Reuse the storage and correction descriptor when values change in place.

`CellGradients` accepts three native rank-one Kokkos Views, including strided field components, covering at least `n_cells`. A custom gradient provider can instead implement device-callable `gradient(cell)[axis]` for axes `0`, `1`, `2`. The known adapter is checked for shape and accessibility; opaque providers remain responsible for their backing storage. All inputs must remain valid and unmodified while kernels read them, and writable targets must not overlap gradient storage.

### Corrected assembly, residual and solve

Existing overloads remain unchanged. To enable correction, insert it after `law` for the operator or after `source` for the ready-made solver and balance diagnostics:

| Operation | Corrected call |
| --- | --- |
| Full additive assembly | `diffusion.add_system_async(target, boundary, status, law, correction, time)` |
| Full additive residual | `diffusion.add_residual_async(phi_component, residual, boundary, status, law, correction, time)` |
| RHS/residual correction only | `diffusion.add_correction_async(output, boundary, status, law, correction, time)` |
| One ready-made solve | `solver.solve_owned(boundary, law, source, correction, time)` |
| Operator balance | `diffusion.balance(phi_component, boundary, law, source, correction, time)` |
| Ready-made solver balance | `solver.balance(boundary, law, source, correction, time)` |

`time` is optional and defaults to zero. The objects named `diffusion`/`solver` in this table must use the corrected plan. Full corrected assembly still adds to existing coefficients and RHS entries:

```cpp
const auto corrected_target = ts::make_system_view(matrix, rhs, shared, exec);
Kokkos::deep_copy(exec, corrected_target.values, SolverReal{0});
Kokkos::deep_copy(exec, corrected_target.rhs, SolverReal{0});
status.begin_async();
corrected_diffusion.add_system_async(
    corrected_target, boundary, status, law, correction);
sources.add_async(corrected_target.rhs, status, source);
(void)status.check(comm);
matrix.commit();
rhs.commit();
// Solve and publish the owned solution as in the external-system example.
```

For the built-in boundary conditions, this uses the same matrix coefficients as zero-correction assembly; only the RHS changes. A custom boundary relation may make its slope depend on the increment, in which case full corrected assembly includes that changed slope.

The matrix-free path uses the same flux formulas. With the same frozen coefficient/gradient data and boundary conditions, it evaluates `b_corrected - A_corrected*phi`; sources remain a separate contribution. It needs current field ghosts as well as current gradient ghosts.

The ready-made solver also accepts the correction:

```cpp
ts::SteadyDiffusion<> corrected_solver(
    corrected_plan, shared, phi.view().component(0), options);
const auto corrected_result = corrected_solver.solve_owned(
    boundary, law, source, correction);
```

This performs **one solve with frozen supplied gradients** and rebuilds its owned matrix/RHS on every call. Linear convergence does not establish convergence of the gradient-dependent corrected equation. Updating gradients, publishing halos and controlling outer iterations remain application responsibilities.

If another algorithm needs extended boundary slots, the same correction can be passed to the scalar boundary manager:

```cpp
boundary.apply_async(phi.view(), law, correction);
```

The diffusion operator itself does not read these slots. Whether a gradient algorithm needs them is part of that algorithm's contract.

### Reusing a frozen matrix

For outer correction iterations with unchanged coefficients, boundary parameters and time, external ownership lets the application retain the matrix and linear solver. First assemble a zero-correction system with `corrected_diffusion`, including the baseline boundary RHS and source, and save that RHS once:

```cpp
const auto base_target = ts::make_system_view(matrix, rhs, shared, exec);
Kokkos::deep_copy(exec, base_target.values, SolverReal{0});
Kokkos::deep_copy(exec, base_target.rhs, SolverReal{0});
status.begin_async();
corrected_diffusion.add_system_async(base_target, boundary, status, law);
sources.add_async(base_target.rhs, status, source);
(void)status.check(comm);
matrix.commit();
rhs.commit();

using Space = typename Exec::memory_space;
Kokkos::View<SolverReal*, Space> base_rhs("base_rhs", host_mesh.n_own);
const auto native_base = rhs.read(exec);
Kokkos::deep_copy(exec, base_rhs,
    Kokkos::subview(native_base, Kokkos::ALL(), 0));
rhs.commit(); // Complete the copy before modifying rhs again.
```

After each external gradient update and gradient halo exchange, execute this step using the same matrix and `linear` object:

```cpp
const auto native_rhs = rhs.overwrite(exec);
const auto owned_rhs = Kokkos::subview(native_rhs, Kokkos::ALL(), 0);
Kokkos::deep_copy(exec, owned_rhs, base_rhs); // Restore b0 every iteration.
status.begin_async();
corrected_diffusion.add_correction_async(
    owned_rhs, boundary, status, law, correction);
(void)status.check(comm);
rhs.commit();

solution.copy_from_component(phi.view().component(0), exec);
solution.commit();
const auto iteration_result = linear.solve(rhs, solution);
if (iteration_result.converged) {
    solution.copy_to_component(phi.view().component(0), exec);
    solution.commit();
}
```

`add_correction_async()` adds **corrected RHS minus zero-correction RHS**, including the boundary difference. It does not add the source or prescribed baseline flux again. Its output is a native rank-one owned-cell RHS/residual slice; strided slices from BSR systems also work. No matrix access is required and its revision is unchanged. Open only the vector for this step: calling `make_system_view()` would unnecessarily open matrix write access and invalidate its revision-based reuse.

Restore `base_rhs` on every iteration; adding another correction onto the previous corrected RHS accumulates stale terms. Do not add this correction-only contribution after full corrected assembly unless intentionally adding a separate term.

Matrix reuse requires boundary normal-gradient slopes to be independent of the increment. The correction-only kernel checks the corrected slope against the zero-increment slope and reports a mismatch through `AssemblyStatus`. Full corrected assembly remains available for such custom conditions. If diffusion coefficients or boundary matrix coefficients change, rebuild the system. If only a source or other baseline RHS data change, refresh the saved baseline accordingly. The caller controls outer convergence, relaxation and failure handling.

## Sources and variable coefficients

A source is a device-callable `source(std::size_t cell)` returning a finite density. Negative densities are sinks. Only owned entries are read, so no source halo is required. `UniformSource<Real>{3}` contributes `3*cell_volume` to the RHS/residual. `CellSource{view}` reads an existing rank-one Kokkos View.

To compute a source directly without a separate density array:

```cpp
template<class X, class R = krona::ComputeReal>
struct LinearSource {
    X x;
    R offset = 0, slope = 0;
    KOKKOS_INLINE_FUNCTION R operator()(std::size_t cell) const
    { return offset + slope*static_cast<R>(x(cell)); }
};

// In the setup/assembly code:
const auto x = mesh.view().geometry.cell_centroid_x;
const LinearSource<decltype(x), Real> spatial_source{x, Real{3}, Real{2}};
const auto result = solver.solve_owned(boundary, law, spatial_source);
```

The provider returns `3 + 2*x`; the source operator performs volume multiplication. Carry any model parameters or time explicitly in the functor: assembly's `time` argument is passed to face contexts, not as a separate source argument. `sources.add_async(output, status, provider)` can target a system RHS or an independent residual View.

For cell-dependent diffusion, attach interpolation weights and prepare an existing coefficient View `D` covering owned and MPI ghost cells:

```cpp
const auto variable_law = transport::IsotropicDiffusion{
    transport::InterpolatedCellCoefficient{D}}; // Harmonic by default.

using CellD = std::remove_cvref_t<decltype(D)>;
const auto arithmetic_law = transport::IsotropicDiffusion{
    transport::InterpolatedCellCoefficient<CellD, transport::ArithmeticCoefficient>{D}};
```

Refresh `D`'s ghosts before use. Custom providers must support both interior and boundary contexts. Common members include `face`, `owner`, position, unit normal, `distance`, `area` and `time`; interior contexts also supply `neighbour` and `owner_weight`. The latter is meaningful only with attached weights. Provider values must agree across MPI interfaces for conservative assembly.

Defaults are `D = 1` and `time = 0`; a default source provider gives zero. Calls without a correction retain the original argument order: `add_system_async(target, boundary, status, law, time)`, `add_residual_async(field, output, boundary, status, law, time)` and `solve_owned(boundary, law, source, time)`. Corrected calls use the explicit overloads listed above.

## Asynchrony and completion

All contributions in a batch use the plan/source operator's stored execution instance. `AssemblyStatus::device_flags(exec)` requires it to match the status instance. Diffusion boundary kernels use host patch dispatch and run on the diffusion plan's instance; boundary `apply_async()` separately uses the boundary manager's instance.

| Operation | Completion contract |
| --- | --- |
| `DiffusionPlan` construction | Blocking local geometry/incidence validation; no implicit algebra-plan construction. |
| `FaceAssemblyPlan` construction | Collective blocking topology/graph setup and upload. |
| `make_system_view()` | Opens native ReadWrite views and marks algebra wrappers pending; native storage access may synchronize. No zeroing or commit. |
| `status.begin_async()` | Queues flag reset and opens one contribution batch. Does not clear coefficients, RHS or residual. |
| `add_system_async()`, `add_residual_async()`, `add_correction_async()`, source `add_async()` | Local kernel submission on the stored instance; no MPI, fence, automatic reset or algebra commit. |
| `status.check(comm)` | Copies flags to host, fences that instance, collectively reduces diagnostics and closes the batch. Does not commit algebra wrappers. |
| Matrix/vector `commit()` | Completes pending access on its recorded instance and clears wrapper pending state. Does not replace transport diagnostics. |
| `solve_owned()` | Collective blocking assembly, diagnostics, commits, solve and successful owned-field publication. With a correction, performs one frozen-gradient solve. |
| `balance()` | Blocking local reductions followed by a collective MPI sum. |

The normal external-system sequence is: initialize targets once, `begin_async()`, submit contributions, `check(comm)`, commit all touched algebra wrappers, then call the linear solver. A fence alone does not clear wrapper state or perform collective diagnostics. Even though `check()` fences submitted work, explicit wrapper commits are still required.

Keep every captured buffer alive and unmodified until its consumers finish. Data produced on the **same execution instance** is ordered before later submissions without intermediate fences. For a different instance, complete the producer or establish an explicit dependency before reading its data. Apply the same rule to halo unpacking and boundary-value publication. The module does not perform application halo exchanges for you.

For a corrected outer iteration, the dependency order is:

1. Publish current application-field ghosts and prepare any boundary slots required by the chosen gradient algorithm.
2. Compute owned-cell gradients externally, then publish their real MPI ghosts. Also refresh coefficient ghosts if the coefficient model requires them.
3. Submit corrected assembly/residual kernels, or restore the base RHS and submit `add_correction_async()`.
4. Check the batch, commit touched algebra wrappers, solve, and publish the new owned field. Repeat with updated gradients until the application's outer criterion is met.

If boundary-slot preparation itself uses the correction, it reads the gradients already stored at that point. Initialize them and deliberately choose the lagged/update policy; the module does not resolve this outer dependency automatically. No halo or gradient-input buffer may be overwritten while a submitted correction still reads it. A linear-solver tolerance alone is not the outer corrected-equation tolerance.

Do not start another batch on a pending status; call `check()` first. One status can aggregate several contributions without losing an earlier error. Its destructor drains pending local work but does not replace collective checking. For matrix-free arrays, no algebra-wrapper commits exist: `check()` completes that batch, and later operations require their own ordering.

## Balance, validation and limitations

An optional collective diagnostic is available for any supplied valid field:

```cpp
const auto balance = solver.balance(boundary, law, source);
// Or: diffusion.balance(phi_component, boundary, law, source, time).
// source_integral, outward_boundary_flux, defect = flux - source.
```

For a corrected calculation, pass the same correction policy and the intended gradient data:

```cpp
const auto corrected_balance = corrected_solver.balance(
    boundary, law, source, correction);
// Or: corrected_diffusion.balance(
//     phi.view().component(0), boundary, law, source, correction, time).
```

Balance evaluates the current source and boundary law from owned values. It does not read boundary slots or reuse assembly integrals, and it has no assembled/committed-operator prerequisite. Supply the law, source, correction, time and backing data corresponding to the state being checked. It integrates physical boundary flux and volume source; interior fluxes are not summed by this diagnostic. This is a synchronized physical boundary/source balance, not a residual norm, an outer-convergence criterion or a solvability test. Omitting the correction selects the zero-increment boundary closure, even after a corrected solve.

Validation is deliberately split by scope:

- Plan construction checks mesh incidence, geometry, cached metrics, attached nonorthogonality vectors and optional orthogonality.
- Host metadata checks known provider/View shapes and execution accessibility. Corrected calls require the nonorthogonality vectors and, for inner faces, interpolation weights. Known gradient components must cover owned and real ghost cells. Opaque custom functors are responsible for their own accesses and backing-storage sizes.
- Contribution kernels flag invalid sources, nonpositive/nonfinite diffusion, unacceptable affine boundary fluxes, nonfinite correction data and nonfinite/unrepresentable updates. The correction-only boundary kernel also rejects increment-dependent matrix slopes. `check()` aggregates these flags collectively.
- The status report records whether full diffusion boundary assembly encountered a value/Robin anchor. A correction-only batch does not establish or check a complete-system anchor.
- Only `SteadyDiffusion` additionally checks a positive finite complete-system diagonal and requires a global diffusion anchor for nonempty systems. The low-level contribution/status path imposes neither requirement.

Validation does not prove discretization accuracy or general matrix solvability. The steady solver rejects pure Neumann problems because no reference, mean constraint or nullspace handling is implemented. Its global anchor check does not detect separately unanchored disconnected components. Empty ranks and globally empty systems are supported.

The implicit stencil remains two-point; nonorthogonal terms are optional and explicit in externally supplied gradients. There is no built-in correction limiter, relaxation, gradient/halo update or outer-iteration controller, and convergence on strongly nonorthogonal meshes is not guaranteed. Other limitations are strictly positive isotropic diffusion and frozen coefficients/explicit source providers. The ready-made steady solver owns a scalar CSR system. Low-level assembly also supports selected diagonal BSR component slices, preserving other components; it does not create cross-component diffusion terms. The matrix is the frozen diffusion linearization, not a full Jacobian of nonlinear coefficient/source/gradient models.

## Future extensions

- Add convection driven by a supplied face mass flux and conservative temporal terms at the equation/system layer. External ownership and additive diffusion assembly are already implemented foundations for this composition.
- Add an optional outer-iteration driver and correction limiting/relaxation policies without making the spatial operator own gradients or halo exchanges.
- Allow a geometry policy to select cached metrics or compute actual distances, area/distance factors and interpolation weights inside kernels. Missing weights must not silently become `0.5`.
- Extend coefficient/assembly policies to anisotropic or coupled diffusion, and provide explicit reference/nullspace treatment where needed.
- Define geometry refresh and graph/map invalidation rules for moving or adaptive meshes before using these static-mesh plans there.
