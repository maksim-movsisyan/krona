# Scalar convection

`krona::transport::scalar::convection` adds finite-volume convection of one scalar to a linear system or a matrix-free residual. The application supplies the face mass flux and, optionally, reconstruction inputs. The module does not compute velocity, gradients, limiters or a solution.

## Equation and signs

The spatial equation uses the outward integrated flux sum:

$$
\sum_{f\in\partial K} m_{Kf}\,\phi_f = S_K V_K,
\qquad r=b-A\phi.
$$

`m` is **integrated mass per unit time**, signed along the stored owner-outward unit normal. It already includes density and face area. On an inner face, positive `m` selects the owner; negative `m` selects the neighbour. On a physical boundary, positive `m` is outflow and negative `m` is inflow, including reverse flow on an outlet patch.

The matrix is always donor-cell upwind. A reconstructed donor value is `phi_donor + increment`; the increment enters the RHS explicitly. Thus:

- Inner owner row: diagonal `max(m, 0)`, neighbour coefficient `min(m, 0)`; reconstruction RHS `-m * increment`.
- Inner neighbour row: opposite flux sign, with the same physical donor and increment.
- Boundary outflow: diagonal `m`, reconstruction RHS `-m * increment`.
- Boundary inflow: RHS `-m * prescribed_value`.

Residual contributions are the **negative outward flux sum**, without cell-volume division. A `FixedValue` is used on inflow; it is not imposed on convective outflow. A mass flux depending on the unknown is treated as frozen: its derivatives do not enter this matrix.

## Objects and dependencies

| Object | Responsibility |
| --- | --- |
| `ConvectionPlan<Exec, Real>` | Validates and borrows device mesh/owned `CellFaces` descriptors, communicator and execution instance. |
| `ConvectionOperator<Exec, Real>` | Adds system, residual or reconstruction-only contributions. Stores the plan by value. |
| `FaceMassFlux{view}` | Reads mass flux by local face ID. The accessible rank-one view must cover `n_faces`; strides are respected. |
| `UniformMassFlux<Real>{value}` | Returns the same signed mass flux on every face. This is not a uniform velocity field. |
| Custom flux callable | Implements `flux(context)` for both inner and boundary contexts. |
| `FixedValueInflow` | Default: samples the provider in a scalar `FixedValue`. |
| `NoReconstruction` | Default: first-order donor-cell values. |
| `Reconstruction` / `make_reconstruction` | Adapts one component of reconstruction data to convection. |
| `AssemblyStatus<Exec>` | Collects device diagnostics for one additive assembly batch. |

The `krona::transport_scalar_convection` target needs mesh-device, scalar-boundary, reconstruction, fields and MPI support; it does not depend on Trilinos. The optional `assembly.hpp` adapter and algebra solve require linear algebra. Convection and diffusion keep separate plans: convection does not need diffusion distances or interpolation metrics.

## Prepare the mesh and boundary conditions

The snippets assume active MPI/Kokkos and a prepared host partition `host`. Use one execution instance for inputs, contributions and status.

```cpp
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/host/aux_connectivity.hpp>
#include <krona/transport/scalar/convection/operator.hpp>

using namespace krona;
namespace ts = transport::scalar;
namespace cv = ts::convection;
namespace bc = boundary::scalar;
namespace rec = numerics::reconstruction;
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Space = typename Exec::memory_space;

Exec exec{};
const auto comm = mpi::CommView::world();
mesh::device::MeshStorage<Exec> mesh(
    host, {.data = mesh::device::MeshData::FiniteVolume}, exec);
const auto m = mesh.view();
const auto host_conn = mesh::host::build_aux_connectivity(
    host, mesh::AuxConnType::CellFaces);
mesh::device::AuxConnectivityStorage<Exec> conn(host_conn, exec);

cv::ConvectionPlan<Exec, Real> plan(m, conn.view(), comm, exec);
cv::ConvectionOperator<Exec, Real> convection(plan);
ts::AssemblyStatus<Exec> status(exec);

boundary::BoundaryLayout<Exec, Real> layout(host, m, exec);
bc::Manager<Exec, Real> boundary(layout);
for (std::size_t p = 0; p < host.patches.size(); ++p)
    boundary.set(static_cast<mesh::PatchId>(p), bc::FixedValue<Real>{2});
boundary.set_unlabelled(bc::FixedValue<Real>{2});
boundary.finalize();
```

The plan requires grouped faces, `FaceCells`, `FaceGeometry` and complete owned `CellFaces`. Construction validates geometry/connectivity locally and waits for that validation. It needs neither cell geometry nor auxiliary geometry. Boundary-layout construction and mesh-based reconstruction have their own cell-geometry requirements; the full mesh above satisfies them.

The communicator must describe the same partition; device descriptors contain no rank metadata to verify that agreement. The boundary manager must use the same owner/face-geometry arrays as the plan, not just matching sizes. If the plan includes cell geometry, its cell-centre arrays must also match the boundary geometry.

Assign every labelled patch, including patches empty on this rank, and explicitly handle unlabelled faces when present. Call `finalize()` after assignments. Layout, mesh/auxiliary owners and communicator must outlive their users. Rebuild dependent descriptors after topology, numbering or geometry changes.

A uniform velocity along x can supply mass flux as follows; define the functor outside a function:

```cpp
struct XVelocityFlux {
    Real density = 1, velocity = 3;
    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& c) const
    { return density * velocity * c.nx * c.area; }
};
// At the call site:
const auto flux = XVelocityFlux{Real{1}, Real{3}};
```

Inner contexts contain `face`, `owner`, `neighbour`, face position `x/y/z`, unit normal `nx/ny/nz`, `area` and `time`. Boundary callables receive `boundary::FaceContext<Real>`. MPI interface copies must represent the same physical mass flux with their respective normal signs; convection does not reconcile them.

## Matrix-free residual

```cpp
Kokkos::View<Real*, Space> q("q", m.sizes.n_cells);
Kokkos::View<Real*, Space> residual("residual", m.sizes.n_own);
Kokkos::deep_copy(exec, q, Real{2}); // Including real MPI ghosts.
Kokkos::deep_copy(exec, residual, Real{0});

status.begin_async();
convection.add_residual_async(q, residual, boundary, status, flux);
(void)status.check(comm);
```

`q` must cover all real cells, including current MPI ghosts. Output must cover the owned prefix; its tail is untouched. Both may be strided rank-one component views. Input and output storage must not overlap, including partial/strided aliases. The runtime catches an identical starting pointer, not every possible overlap.

The operator and its standard reconstruction adapter do not read extended boundary-field slots. Boundary values are sampled from the conditions. An external gradient/limiter algorithm may still require filled boundary slots.

## Assemble and solve

```cpp
#include <krona/transport/scalar/assembly.hpp>
#include <krona/linalg/solver.hpp>

// Requires an active Tpetra runtime.
linalg::FaceAssemblyPlan<Exec> assembly(host, comm, exec);
linalg::CsrMatrix<SolverReal, Exec> matrix(assembly.pattern());
linalg::Vector<SolverReal, Exec> rhs(matrix.distribution());
const auto target = ts::make_system_view(matrix, rhs, assembly, exec);
Kokkos::deep_copy(exec, target.values, SolverReal{0});
Kokkos::deep_copy(exec, target.rhs, SolverReal{0});

status.begin_async();
convection.add_system_async(target, boundary, status, flux);
// Add diffusion, transient terms and sources to this same initialized target.
(void)status.check(comm);
matrix.commit();
rhs.commit();

linalg::Vector<SolverReal, Exec> x(matrix.distribution());
x.copy_from_component(q, exec); // Copies the owned initial guess and waits.
linalg::SolverOptions options;
options.method = "GMRES";
options.relative_tolerance = 1e-6;
options.preconditioner.type = "RELAXATION";
linalg::Solver<decltype(matrix)> linear(matrix, options);
const auto solved = linear.solve(rhs, x);
if (solved.converged)
    x.copy_to_component(q, exec); // Owned values only; waits.
// Publish q's MPI ghosts externally before its next spatial use.
```

All contributions are additive: initialize storage once before the batch. The assembly plan must use the convection mesh's **same local cell and face numbering**. Matching sizes alone do not prove this.

`make_system_view()` accepts scalar CSR and a selected diagonal-component slice of BSR. BSR entry positions identify blocks; the selected equation/variable component is updated, not the entire block. Its RHS selects the matching point component. The matrix must share the assembly graph and the RHS must share its distribution. Commit native algebra access before solving.

Without Trilinos, construct a `ts::SystemView{values, rhs, entries}` over existing storage. `values` and `rhs` are writable, accessible rank-one views, possibly strided. RHS extent is exactly `n_own`. The entry map contains:

| Array | Meaning |
| --- | --- |
| `diagonal(cell)` | Native coefficient/block entry for each owned diagonal. |
| `owner_entries(face)` | Owner row → neighbour column, for `face < n_inner_faces`. |
| `neighbour_entries(face)` | Neighbour row → owner column, for `face < n_internal_faces` (owned/owned). |

All entry positions must be valid for `values`; duplicate connecting faces may share an entry. The caller supplies a correct graph/map. Convection does not allocate or reorder it, impose a reference constraint, or test global solvability.

## Boundary profiles

Profiles belong to **boundary conditions**. Convection samples them through `FixedValueInflow`; there is no separate convection profile registry.

For a compact array on one patch, register its concrete condition type when creating the manager:

```cpp
using PatchValue = bc::FixedValue<
    Real, boundary::profile::PatchValues<float, Space>>;
bc::Manager<Exec, Real, PatchValue> profiled_boundary(layout);
for (std::size_t p = 0; p < host.patches.size(); ++p)
    profiled_boundary.set(static_cast<mesh::PatchId>(p), bc::FixedValue<Real>{2});
profiled_boundary.set_unlabelled(bc::FixedValue<Real>{2});

const auto range = profiled_boundary.catalog().patches()[0].range;
Kokkos::View<float*, Space> values("patch0", range.end - range.begin);
Kokkos::deep_copy(exec, values, 5.0F);
profiled_boundary.set(0, bc::fixed_value<Real>(
    boundary::profile::patch_values(range, values)));
profiled_boundary.finalize();

status.begin_async();
convection.add_residual_async(q, residual, profiled_boundary, status, flux);
(void)status.check(comm);
```

Entry `k` supplies local face `range.begin + k`. The extent must equal the **local** patch size, including zero on ranks without its faces. This is neither global patch order nor cell/boundary-slot order. Const and strided views work; their scalar type is converted to `Real`. Register each distinct profiled condition type in the manager's template parameters. For unlabelled faces, use `catalog().unlabelled()` and `set_unlabelled()`.

`PatchValues` borrows pointer/stride metadata. Keep its allocation alive and synchronize producers before reads. Updating the values is visible without reassignment; after replacing its pointer/stride, call `set()` and `finalize()` again. A constant or device-callable `provider(context)` can be used instead of `PatchValues`; the same BC/provider interface is described in `BOUNDARY-SCALAR.md`.

Default inflow handling accepts only `FixedValue<Real, Provider>`. A prescribed normal gradient or diffusive flux does not determine a convective inflow value. On actual inward flow, such a condition requires a custom `inflow(context, condition)` callable; otherwise `status.check()` rejects the batch. Outflow and zero mass flux do not sample inflow data. A boundary profile used by both convection and diffusion retains the same physical BC meaning for both.

A custom operator-specific policy can interpret another condition type without a second catalog:

```cpp
struct MyInflow {
    template<class Context, class Condition>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& c,
                                           const Condition& condition) const
    {
        if constexpr (bc::is_fixed_value<Condition>)
            return cv::FixedValueInflow{}(c, condition);
        else
            return Real{2} + c.time; // Application's explicit fallback.
    }
};
// At the call site, with ready inputs:
convection.add_residual_async(q, residual, boundary, status, flux,
    cv::NoReconstruction{}, MyInflow{}, Real{0.5});
```

The callable must be device-copyable and compile for every condition alternative in that manager. It is evaluated only on inflow faces.

## Reconstruction

Compute inputs externally, publish all accessed ghosts, then adapt the selected scalar component:

```cpp
// dq has components (dphi/dx, dphi/dy, dphi/dz) for every real cell.
// q and dq are already populated and remain alive.
const auto input = rec::data(fields::pack(q), rec::gradient_pack(dq.view()));
const auto reconstruction = cv::make_reconstruction(
    plan, rec::SecondOrderUpwind{}, input);

status.begin_async();
convection.add_residual_async(q, residual, boundary, status, flux,
    reconstruction);
(void)status.check(comm);
```

For several packed scalars, pass the selected component as the last argument to `make_reconstruction(plan, scheme, data, component)`. Gradient component order is `3*c + axis`. `Limited<...>` additionally needs previously computed factors in `data`; TVD computes its ratio limiter during reconstruction. Convection computes none of these inputs. See `NUMERICS-RECONSTRUCTION.md` for limiter preparation, Hessians and moments.

The mesh adapter uses both sides for inner faces. At outflow, schemes requiring `OtherValues` (such as MUSCL/TVD) default to one-sided `LimitedLinear`: no nonexistent exterior sample is read, and absent factors mean unit factors. Other schemes retain their own polynomial. An explicit one-sided outflow policy can be selected:

```cpp
const auto reconstruction = cv::make_reconstruction(
    plan, rec::Tvd<>{}, input, rec::SecondOrderUpwind{});
```

QUICK and larger/custom stencils use `make_reconstruction(scheme, data, stencil, component)`, with explicit `face(f)` (or `face<Scheme>(f)`) and `boundary_side(f)` methods. The stencil owns the choice of sample indices/geometry. All accessed samples must exist and be current; a check against `n_cells` cannot establish the validity of a larger stencil.

Choose a concrete scheme on the host before submission, for example with a `switch` calling a templated lambda. No runtime scheme dispatch is needed inside face kernels. Reconstruction inputs must not overlap written coefficients, RHS or residual storage.

## Reuse a fixed upwind matrix

For deferred reconstruction updates, first assemble the equation with `NoReconstruction`, including prescribed inflow, sources and other fixed terms. Save that **base RHS without reconstruction** and commit the matrix. Each outer iteration updates the reconstruction inputs, restores the saved base RHS and adds the current reconstruction difference:

```cpp
const auto b = Kokkos::subview(rhs.overwrite(exec), Kokkos::ALL(), 0);
Kokkos::deep_copy(exec, b, base_rhs);
status.begin_async();
convection.add_correction_async(b, boundary, status, flux, reconstruction);
(void)status.check(comm);
rhs.commit();
// Solve with the existing committed matrix; publish the updated field externally.
```

This gives the same RHS as full reconstructed assembly with those inputs. It includes inner-face and outflow increments, repeats no prescribed inflow/source contribution and touches no matrix coefficient or revision. Do not accumulate previous iterations' corrections. Rebuild the matrix/base equation if mass flux changes; refresh the base RHS when prescribed inflow or other fixed terms change. Use time and inputs consistent with that base equation. Outer convergence belongs to the application; one linear solve only solves its current frozen system.

## Asynchrony, diagnostics and storage

| Operation | Completion contract |
| --- | --- |
| Plan construction | Blocking local validation; no collective. |
| Operator construction | Copies descriptors; no kernel submission or synchronization. |
| BC `set()` / `finalize()` | Local host metadata/provider validation; no field upload or collective. |
| `status.begin_async()` | Enqueues diagnostic reset; does not clear equation arrays. |
| `add_system_async()`, `add_residual_async()`, `add_correction_async()` | Enqueue local kernels; no fence, halo exchange, collective or algebra commit. |
| `status.check(comm)` | Waits for its execution instance and performs collective diagnostic reduction. Does not commit algebra access. |

The first two contribution methods default to `NoReconstruction{}`, `FixedValueInflow{}` and `time = 0`. Pass preceding policy arguments explicitly to supply time. `add_correction_async()` requires a reconstruction and has no inflow argument.

Use `begin_async()` once per batch and `check(comm)` once after all contributions. Every participating rank must call the collective check, including empty ranks. The status and operators must use the same execution instance. Complete producers on other instances before submission; keep borrowed storage valid until consumers finish. The status destructor waits if a batch remains pending, but does not replace an explicit collective check.

Invalid host contracts use `mpi::require`/`mpi::guarded`. Device diagnostics reject nonfinite mass fluxes, used inflow values, increments and invalid additions. A nonzero flux used in matrix assembly must also stay finite and nonzero when converted to the matrix scalar type. Zero flux skips scalar/reconstruction/inflow evaluation. Failed batches are not rolled back. `has_diffusion_anchor` is a diffusion report; convection does not set it.

Plans/operators allocate no field, graph, matrix or face-work buffers. Row kernels capture only the arrays they use. `SystemView` borrows existing equation storage; `FaceAssemblyPlan` separately owns its shared graph/entry maps. `AssemblyStatus` allocates small diagnostic buffers. `FaceMassFlux` stores its view by value (managed views retain storage); reconstruction packs and `PatchValues` borrow their backing buffers. Custom providers follow the ownership of their members.