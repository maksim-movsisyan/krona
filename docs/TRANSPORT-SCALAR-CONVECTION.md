# Scalar convection

`krona::transport::scalar::convection` assembles conservative finite-volume transport of one scalar using a supplied face mass flux. It supports donor-cell upwind, explicit reconstruction increments, matrix-free residuals and independent inflow profiles. It does not calculate the velocity/mass flux, reconstruct gradients or run a linear/time solver.

## Equation and signs

For the conservative equation

$$
\nabla\cdot(\rho\mathbf u\phi)=s,
$$

the convective face flux is `F = m * phi_face`, where

$$
m_f=\int_f\rho\mathbf u\cdot\mathbf n\,dA
$$

is **integrated mass per unit time**, signed along the stored owner-outward unit normal. Area and density must already be included. Positive inner `m` selects the owner as donor; negative `m` selects the neighbour. On a physical boundary, positive `m` is outflow and negative `m` is inflow, including local backflow on an outlet patch.

The integrated residual is

$$
r_i=b_i-(A\phi)_i=-\sum_f F_{i,f}^{\mathrm{out}}+s_iV_i.
$$

The operator includes neither the source nor division by cell volume. Add volume sources separately; divide by volume externally if a time update needs a rate per unit volume. It implements conservative transport and does not subtract `phi * div(mass_flux)`: a constant scalar has a nonzero convective residual when the supplied mass flux has nonzero discrete divergence.

The matrix is always **donor-cell upwind**. A reconstructed face value is

$$
\phi_f=\phi_{\mathrm{donor}}+\delta_f.
$$

The donor value is implicit; `m * delta` is explicit. For an inner face the owner RHS receives `-m * delta`, and an owned neighbour receives `+m * delta`. Reconstruction changes the RHS, not the upwind matrix. Mass flux is frozen during linearization; its dependence on unknowns is not differentiated.

## Objects and public operations

| Object | Responsibility |
| --- | --- |
| `ConvectionPlan<Exec, Real>` | Validates and borrows mesh/owned `CellFaces` descriptors; retains the execution instance and communicator. |
| `ConvectionOperator<Exec, Real>` | Adds system, residual or reconstruction-only contributions. Stores a copy of the plan; allocates no equation/work arrays. |
| `FaceMassFlux`, `UniformMassFlux`, custom flux callable | Supply signed integrated mass fluxes. |
| `Reconstruction` / `make_reconstruction()` | Adapt one selected component of reconstruction data to inner donor and boundary-outflow increments. |
| `FixedValueInflow`, custom inflow callable, `InflowProfiles` | Supply scalar values on faces whose mass flux is inward. |
| `scalar::SystemView`, `AssemblyStatus`, `SourceOperator` | Shared scalar-system slices, batch diagnostics and volume-source assembly. |

The operator methods are:

```cpp
convection.add_system_async(target, boundary, status, flux,
    reconstruction, inflow, time);
convection.add_residual_async(q, residual, boundary, status, flux,
    reconstruction, inflow, time);
convection.add_correction_async(output, boundary, status, flux,
    reconstruction, time);
```

Defaults are `NoReconstruction{}`, `FixedValueInflow{}` and `time = 0` for the first two methods. The correction-only method requires an explicit reconstruction and has no inflow argument. To pass time, supply the preceding policy arguments explicitly. All three methods add to existing output; none clears arrays, exchanges halos, fences or commits algebra access.

`Real` controls transport arithmetic; destination views control stored coefficients/results. A single call handles one scalar equation, not every component of a field.

## Prepare the mesh and mass flux

The examples assume active MPI/Kokkos and an already prepared `mesh::host::MeshPart host`. Matrix/solve examples additionally require an active Tpetra runtime.

```cpp
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/host/aux_connectivity.hpp>
#include <krona/transport/scalar/convection/operator.hpp>

using namespace krona;
namespace ts = transport::scalar;
namespace cv = transport::scalar::convection;
namespace bc = boundary::scalar;
namespace rec = numerics::reconstruction;
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
mesh::device::AuxConnectivityStorage<Exec> conn(host_conn, exec);

cv::ConvectionPlan<Exec, Real> plan(host, m, conn.view(), comm, exec);
cv::ConvectionOperator<Exec, Real> convection(plan);
ts::AssemblyStatus<Exec> status(exec);

bc::Manager<Exec, Real> boundary(host, m, exec);
for (std::size_t p = 0; p < host.patches.size(); ++p)
    boundary.set(static_cast<mesh::PatchId>(p), bc::FixedValue<Real>{2});
boundary.set_unlabelled(bc::FixedValue<Real>{2});
```

The current constructor takes `host` for setup checks; it does not retain that host-mesh object. Host/device counts and communicator rank/size must agree. The plan requires grouped faces, `FaceCells`, `CellGeometry`, `FaceGeometry` and complete owned `CellFaces`. Geometry and cell-face incidence are validated locally. Nonorthogonal meshes are supported; diffusion distance metrics and auxiliary geometry are not required. Rebuild the plan after changing topology, numbering or geometry.

The boundary manager must use the same mesh arrays as the plan, not merely matching counts. Assign every labelled patch, even if empty locally, and unlabelled faces when present. This example uses the same possible inflow value everywhere; real cases usually assign different patch conditions.

Mass-flux providers are:

| Provider | Contract |
| --- | --- |
| `FaceMassFlux{values}` | Accessible rank-one Kokkos view with at least `n_faces` entries, indexed by local face ID; strided views work. |
| `UniformMassFlux<Real>{value}` | Returns the same signed integrated flux for every face; default `0`. This is not a uniform velocity field. |
| Custom callable | Device-copyable `operator()(const Context&) const`, returning a finite integrated flux. Custom array extents/accessibility are the caller's responsibility. |

For example, define this provider at namespace scope to transport with constant density and velocity:

```cpp
struct AdvectionFlux {
    Real density = 1, ux = 1, uy = 0, uz = 0;
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& c) const
    { return density * (ux*c.nx + uy*c.ny + uz*c.nz) * c.area; }
};
```

Inner `InteriorFaceContext<Real>` provides `face`, `owner`, `neighbour`, `x/y/z`, `nx/ny/nz`, `area` and `time`. Its orientation remains owner-to-neighbour even while assembling the neighbour row. Physical faces use `boundary::FaceContext<Real>`, with owner-outward geometry and no `neighbour` member. A custom flux must compile for both context types.

Interface copies on different ranks must represent the same physical flux with their respective normal signs; the module does not reconcile them. All accessed coefficients/fluxes must be ready before use. Zero mass flux contributes nothing and skips scalar/reconstruction/inflow evaluation on that face. A nonzero flux used in matrix assembly must remain finite and nonzero when converted to the matrix scalar type.

## Matrix-free residual

```cpp
Kokkos::View<Real*, Space> q("q", static_cast<std::size_t>(host.n_cells));
Kokkos::View<Real*, Space> residual("r", static_cast<std::size_t>(host.n_own));
Kokkos::deep_copy(exec, q, Real{2}); // Constant example, including real ghosts.
Kokkos::deep_copy(exec, residual, Real{0});
const AdvectionFlux flux{};

status.begin_async();
convection.add_residual_async(q, residual, boundary, status, flux);
(void)status.check(comm);
```

This adds the negative outward convective-flux sum to the owned output prefix. `q` must contain at least `n_cells` real-cell entries, including current MPI ghosts. Output must contain at least `n_own`; any tail is untouched. A selected rank-one field component can replace either view, including a strided component.

Extended boundary slots are not read by this operation or the standard convection reconstruction adapter. Inflow values come from its inflow policy. Boundary-value filling may still be needed by an external gradient/limiter algorithm.

Read buffers must not overlap written output. In-place residual evaluation is unsupported; equal starting pointers are rejected, but partial/strided aliases remain the caller's responsibility. No algebra graph, matrix, vector or Tpetra solver is needed for this residual path.

## Assemble and solve a system

```cpp
#include <krona/transport/scalar/assembly.hpp>
#include <krona/transport/scalar/source.hpp>
#include <krona/linalg/solver.hpp>

linalg::FaceAssemblyPlan<Exec> assembly(host, comm, exec);
linalg::CsrMatrix<SolverReal, Exec> matrix(assembly.pattern());
linalg::Vector<SolverReal, Exec> rhs(matrix.distribution());
const auto target = ts::make_system_view(matrix, rhs, assembly, exec);
Kokkos::deep_copy(exec, target.values, SolverReal{0});
Kokkos::deep_copy(exec, target.rhs, SolverReal{0});

status.begin_async();
convection.add_system_async(target, boundary, status, flux);
// Add diffusion, transient terms or volume sources to the same batch as needed.
(void)status.check(comm);
matrix.commit();
rhs.commit();

linalg::Vector<SolverReal, Exec> x(matrix.distribution());
x.copy_from_component(q, exec); // Owned initial guess.
x.commit();
linalg::SolverOptions options;
options.method = "GMRES";
options.relative_tolerance = 1e-6;
options.preconditioner.type = "RELAXATION";
linalg::Solver<decltype(matrix)> linear(matrix, options);
const auto solved = linear.solve(rhs, x);
if (solved.converged) {
    x.copy_to_component(q, exec); // Owned values only.
    x.commit();
}
```

The example assembles first-order steady convection. A unique steady solution requires appropriate inflow/reference constraints; the operator does not impose them or test global solvability. There is no convection-specific convenience solver. Use the algebra solver directly, or compose this contribution into a larger scalar equation. Publish updated field ghosts externally before the next residual/reconstruction operation.

For an owned row and a face with row-outward mass flux `M`, upwind assembly adds `M` to the diagonal when `M > 0`, or `M` to the donor-neighbour column when `M < 0`. Boundary outflow adds positive `m` to the diagonal; inflow adds `-m * phi_inflow` to the RHS. A boundary `FixedValue` is therefore not enforced on convective outflow.

### Matrix and storage contracts

| Target | Supported use |
| --- | --- |
| `linalg::CsrMatrix<float/double, Exec>` | One scalar unknown per cell; helper component `0`. |
| `linalg::BsrMatrix<float/double, Exec>` | A diagonal equation/variable slice `(block, component, component)`, plus RHS DOFs `cell * block_size + component`. |
| Direct `SystemView{values, rhs, entries}` | Custom rank-one coefficient/RHS slices and matching entry positions; no particular matrix backend is imposed by the operator. |

Native scalar support also depends on the configured Trilinos instantiations. `make_system_view()` requires the matrix to share the **exact native graph object** with `assembly.pattern()` and the RHS to share its exact point distribution. An independently reconstructed equivalent graph/map is insufficient. Local cell/face numbering must agree with the convection plan.

For example, after constructing/initializing a three-component BSR system:

```cpp
#include <krona/linalg/bsr_matrix.hpp>

linalg::BsrMatrix<SolverReal, Exec> blocks(assembly.pattern(), 3);
linalg::Vector<SolverReal, Exec> block_rhs(blocks.distribution());
Kokkos::deep_copy(exec, blocks.overwrite_blocks(exec), SolverReal{0});
Kokkos::deep_copy(exec, block_rhs.overwrite(exec), SolverReal{0});
const auto equation = ts::make_system_view(blocks, block_rhs, assembly, exec, 1);
status.begin_async();
convection.add_system_async(equation, boundary, status, flux);
(void)status.check(comm);
blocks.commit();
block_rhs.commit();
// Fill the other equations before solving the whole block system.
```

This changes component `1` only; it does not create cross-component couplings. A direct `SystemView` may select a different equation/variable pair. Raw Tpetra matrices need an explicit storage adapter rather than `make_system_view()`.

For a direct slice, include `<krona/transport/scalar/system_view.hpp>`. Coefficient/RHS views must be writable, accessible, floating-point and rank one. RHS extent must be exactly `n_own`. Entry-map members are `diagonal` (`n_own`), `owner_entries` (`n_inner_faces`) and `neighbour_entries` (`n_internal_faces`), indexing existing positions in the supplied native-order coefficient slice. Repeated connecting faces may share an entry and their contributions accumulate. The operator checks descriptor extents but does not prove that custom positions describe the intended graph.

Initialize equation buffers once, then add contributions. `make_system_view()` opens native ReadWrite access without clearing or committing. `FaceAssemblyPlan` has no original-input permutation: assemble through native views/entry maps, not original-order `set_values*()`.

`SourceOperator<Exec, Real>(m, exec).add_async(output, status, source)` adds `source(cell) * volume(cell)` to an owned output prefix. `UniformSource{value}` and `CellSource{view}` are available; positive sources add to RHS/residual. Convection, diffusion and sources can share the same initialized target and status batch.

## Reconstruction and scheme selection

Prepare reconstruction inputs externally, including all accessed ghosts, then pass an adapter:

```cpp
// gx/gy/gz are ready cell-gradient views; theta contains ready factors in [0,1].
// All four views cover real cells, including ghosts, and remain alive.
const auto data = rec::data(rec::pack(q),
    rec::gradients(rec::pack(gx), rec::pack(gy), rec::pack(gz)), rec::pack(theta));
const auto reconstruction = cv::make_reconstruction(
    plan, rec::LimitedLinear{}, data, 0); // Component 0 of this Data pack.

Kokkos::deep_copy(exec, residual, Real{0});
status.begin_async();
convection.add_residual_async(q, residual, boundary, status, flux, reconstruction);
(void)status.check(comm);
```

The adapter returns an **increment**, not a complete face value. The operator selects the donor using `m` and adds that donor's increment to its cell value. For residual evaluation, `data.values` must describe the same scalar/component and current state as the supplied `q` view. `component` indexes the reconstruction pack; it is independent of the BSR equation component selected by `make_system_view()`.

| Choice | Inputs and behaviour |
| --- | --- |
| `NoReconstruction{}` | Donor-cell upwind; no reconstruction data needed for assembly. |
| `rec::FirstOrder{}` | Zero increment through the reconstruction adapter. |
| `rec::SecondOrderUpwind{}` | Gradient-based, unlimited linear reconstruction. |
| `rec::LimitedLinear{}` / `rec::Limited<Scheme>{}` | Precomputed cell factors multiply the complete selected scheme increment. |
| `rec::MusclKappa{}` | Unlimited MUSCL interpolation; reads gradients and the other cell's value. |
| `rec::Tvd{}` | Evaluates the ratio/flux limiter at use; no separately prepared cell factors are needed for its inner TVD increment. |
| `rec::Quadratic{}` | Gradients, Hessians and central cell moments. |
| `rec::Quick{}` | Explicit three-sample side stencils; cannot use the plan-based factory. |

For plan-based reconstruction, schemes that read `OtherValues` (MUSCL/TVD) automatically use `LimitedLinear` on outflow. This avoids reading an extended boundary slot as a neighbour. It uses `data.factors`; omitted factors default to `UnitFactors`, so that outflow trace is then unlimited linear. Other schemes use the same polynomial on inner and outflow faces. Override explicitly when needed:

```cpp
const auto tvd_reconstruction = cv::make_reconstruction(
    plan, rec::Tvd{}, data, rec::FirstOrder{}, 0); // First-order outflow.
```

An explicit outflow scheme in this factory must not require `OtherValues`. For QUICK or a larger stencil, use `make_reconstruction(scheme, data, stencil, component)`. The supplied device-copyable stencil provides `face<Scheme>(f)` or `face(f)` with `left` = owner and `right` = neighbour sides, and `boundary_side(f)` with a valid one-sided outflow stencil. Sample indices address local reconstruction data, including any additional halo samples. The module performs no stencil search or halo enlargement. Validate QUICK weights/stencils before submission; there is no automatic fallback.

Scheme choice can happen once on the host while each kernel uses a concrete type:

```cpp
auto add_selected = [&](auto scheme) {
    const auto r = cv::make_reconstruction(plan, scheme, data);
    convection.add_residual_async(q, residual, boundary, status, flux, r);
};
status.begin_async();
switch (method) {
case Method::Linear: add_selected(rec::SecondOrderUpwind{}); break;
case Method::Muscl:  add_selected(rec::MusclKappa{}); break;
case Method::Tvd:    add_selected(rec::Tvd{}); break;
}
(void)status.check(comm);
```

Here `Method` is an application enum and `residual` is already initialized. The same dispatch works with `add_system_async()`.

Gradients, factors, Hessians and moments are never allocated or calculated by convection. Calculate required owned data, publish its ghosts, then submit the contribution. Preparing cell factors may itself need extended boundary samples; see `NUMERICS-RECONSTRUCTION.md` for limiter preparation, scheme contracts and boundary sampling. On large stencils, size checks for `n_cells` do not establish that every extra sample exists.

A custom reconstruction object can bypass the adapter by supplying device-callable `interior(InteriorFaceContext<Real>, bool from_owner)` and `boundary(boundary::FaceContext<Real>)` methods returning finite increments. The boundary method is used only for outflow. Custom descriptors and their accessed indices are the caller's responsibility.

## Inflow values and patch profiles

The default `FixedValueInflow` extracts a scalar only from `bc::FixedValue`. `NormalGradient`, `FixedFlux` and custom scalar conditions do not define a default convective inflow value. If such a condition is encountered with `m < 0`, supply an inflow policy/profile or `check()` rejects the batch. Outflow/zero-flux faces do not evaluate the inflow policy.

Diffusive `FixedFlux` is not a convective scalar value. Separate inflow data allows convection and diffusion to share a boundary manager while using the appropriate data for each operator. All manager conditions must still be assigned, including when profiles override their inflow values.

A custom inflow policy is device-copyable and callable as `inflow(context, typed_condition)`. It must compile for every condition alternative in the manager. For example, at namespace scope:

```cpp
struct TimedInflow {
    template<class Context, class Condition>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& c, const Condition&) const
    { return Real{2} + Real{0.1} * Kokkos::sin(c.time); }
};
```

Pass it as the argument after reconstruction, followed by a finite evaluation time:

```cpp
status.begin_async();
convection.add_residual_async(q, residual, boundary, status, flux,
    cv::NoReconstruction{}, TimedInflow{}, Real{1});
(void)status.check(comm);
```

For spatially varying data already stored in arrays, bind a **compact local patch profile**:

```cpp
// Patch ID 0 exists in this example; its local range may be empty.
const auto range = boundary.catalog().patches()[0].range;
Kokkos::View<float*, Space> profile("inflow/patch0",
    static_cast<std::size_t>(range.end - range.begin));
Kokkos::deep_copy(exec, profile, 5.0F);
auto inflows = cv::make_inflow_profiles(boundary);
inflows.set(0, profile);

status.begin_async();
convection.add_residual_async(q, residual, boundary, status, flux,
    cv::NoReconstruction{}, inflows);
(void)status.check(comm);
```

Profile entry `k` supplies face `range.begin + k`; this is not a full face/cell array. Extent must equal the **local** patch size exactly, including zero on ranks without its faces. Float/double, const and strided rank-one views are supported. The same registry may mix float and double profiles.

`make_inflow_profiles(boundary, fallback)` optionally selects a custom fallback; otherwise unbound patches use `FixedValueInflow`. `set_unlabelled(view)` binds the compact unlabelled range and requires that the catalog reports unlabelled faces. `clear(patch_id)` / `clear_unlabelled()` restore fallback handling without editing profile buffers. Changing buffer values is visible on the next contribution without rebinding; use `set()` again after changing its pointer/stride.

The registry is host-only and tied to the **exact boundary catalog object**, including its local ranges. It stores raw pointer/stride bindings, not owning views. Keep that manager/catalog and all profile allocations alive until their users finish. Selection happens once per local patch on the host; kernels receive only a compact profile descriptor or fallback, with no per-face catalog lookup. Bindings may change between submissions; coordinate buffer writes with pending kernels.

## Reuse a fixed matrix

For repeated reconstruction updates, assemble the base upwind equation once, including inflow/source/other fixed terms, and save its RHS. Each outer iteration updates field/gradient/factor inputs, restores the base RHS and adds only the reconstruction difference:

```cpp
const auto b = Kokkos::subview(rhs.overwrite(exec), Kokkos::ALL(), 0);
Kokkos::deep_copy(exec, b, base_rhs); // Saved RHS assembled with NoReconstruction.
status.begin_async();
convection.add_correction_async(b, boundary, status, flux, reconstruction);
(void)status.check(comm);
rhs.commit();
// Solve using the existing committed matrix, then publish the new field externally.
```

This adds reconstructed RHS minus upwind RHS, including the outflow increment. It does not repeat prescribed inflow, sources or other base terms, and never touches matrix coefficients/revision. Do not accumulate corrections from previous iterations. Rebuild the matrix/base RHS if mass flux changes; refresh the base RHS when prescribed inflow or other base terms change. Use inputs/time consistent with that base equation. The application controls outer convergence; one successful linear solve only solves the current frozen system.

## Diagnostics, ownership and asynchrony

Use one `AssemblyStatus` batch for all contributions to an equation: `begin_async()` resets diagnostic flags, contributions accumulate errors, and `check(comm)` completes kernels and checks errors across ranks. A valid later contribution cannot erase an earlier failure. The convection plan/operator/profile checks use `mpi::require`; invalid mass fluxes, used inflow values, increments and nonfinite/unrepresentable additions are reported through device flags. A failed batch is not rolled back. This diagnostic does not check discrete mass conservation, stencil accuracy, global solvability or limiter quality. `has_diffusion_anchor` concerns diffusion only; convection does not set it.

| Operation | Completion and communication |
| --- | --- |
| `ConvectionPlan` construction | Blocking local geometry/connectivity validation; no collective. |
| `make_reconstruction()` | Local descriptor construction; no kernel launch, data copy or fence. |
| Profile registry construction / `set()` / `clear()` | Local host metadata operations; no upload, data copy, fence or collective. |
| `make_system_view()` | Opens native access; may synchronize native data. Does not commit. |
| `status.begin_async()` | Queues flag reset on the selected execution instance. Does not clear equation arrays. |
| Operator `add_*_async()` | Queue local kernels; no fence, halo exchange, MPI collective or algebra commit. |
| `status.check(comm)` | Copies flags to host, fences its instance and performs an MPI all-reduce. Does not commit algebra wrappers. |
| Matrix/vector `commit()` | Completes pending access on its recorded instance and closes the access batch; no collective. |
| Algebra `solve()` | Blocking and collective; application publication is separate. |

Use the same execution instance for status and contributions. Order producers/consumers on that instance; establish completion before consuming data produced on another instance or by external MPI. All ranks, including empty ranks, enter collective checks/solves in the same order. Keep read/write buffers live and stable until completion and avoid concurrent overlapping assembly writes. Destroying a pending status fences but does not perform its collective error check.

Plans/operators borrow mesh/connectivity arrays and the communicator; they own no field, graph, matrix or face-work arrays. Keep their underlying owners alive, including host arrays borrowed on CPU. `FaceAssemblyPlan` separately owns shared graph/entry-map allocations. `SystemView` references existing storage; retain its matrix/RHS or custom owners. `AssemblyStatus` allocates a small device flag array and host mirror.

`FaceMassFlux` stores its view by value: managed views retain allocations, unmanaged views borrow. Reconstruction packs and patch profile descriptors borrow pointers. A custom provider's ownership follows its members. The operator performs no field-to-algebra copies; the explicit `copy_from_component()` / `copy_to_component()` calls in the solve example copy the owned prefix and convert to the destination scalar type.
