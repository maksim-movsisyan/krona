# Euler transport

`krona::transport::euler` evaluates the inviscid finite-volume residual of the compressible Euler equations on a static 3-D mesh. It reconstructs primitive variables, prepares thermodynamics through an EOS and calls a Riemann solver. It also assembles a compact inviscid Jacobian for conservative increments. Link `krona::transport_euler`. The native BSR adapter requires Trilinos. The operator leaves viscous terms, CFL selection and time integration to the caller.

## Input and output

The component orders are fixed:

| Pack | Components `0 ... 4` |
| --- | --- |
| Primitive input | `p, u, v, w, T` |
| Conservative state / residual | `rho, rho*u, rho*v, rho*w, rho*E` |

`E = e + |velocity|^2/2` is specific total energy. The operator reconstructs **only `p, u, v, w, T`**. It derives density, specific internal energy and sound speed from the reconstructed `p,T` through the state policy/EOS; those derived quantities need no reconstruction arrays.

The contribution is additive:

$$
R_i\mathrel{-}=\sum_{f\in i} A_f\,\widehat F_{i,f}^{\mathrm{out}}.
$$

The Riemann solver receives a unit normal and returns flux **per unit area**. Euler multiplies by area once and performs **no volume division**. On an inner face the owner receives `-A*F`, an owned neighbour receives `+A*F`; ghost output rows are never written.

Primitive storage covers real cells, including MPI ghosts. When the closure consumes prepared boundary values, the pack also includes one slot per boundary face:

```cpp
extended_index = n_cells + face - n_inner_faces;
```

These are primitive face samples, not exterior ghost cells. Output needs only the owned-cell prefix. All packs have exactly five components; their components share a scalar type/memory space, but input and output scalar types may differ. Accessible rank-one views, including strided components, are supported.

Accessed base primitives must have finite positive `p,T`, finite velocity and an admissible finite EOS/conservative state after conversion to computation `Real`. Initialize residuals to finite values before additive contributions. A custom pack supplies `components`, `value_type`, `memory_space`, `size()` and device-callable `(cell, component)` access. For boundary residuals, the output accessor must return a mutable scalar reference for the atomic additions.

## Main pipeline

Assume active MPI/Kokkos, a prepared `mesh::host::MeshPart host`, and matching `comm`/`exec`. Run host setup under the project's `mpi::guarded` error boundary. The boundary example assumes patch `0` is an inlet facing negative x, patch `1` is an outlet and patch `2` contains slip walls; replace assignments for the actual mesh.

```cpp
#include <krona/numerics/reconstruction/reconstruction.hpp>
#include <krona/numerics/reconstruction/stencil_data.hpp>
#include <krona/transport/euler/operator.hpp>
#include <krona/boundary/physical/manager.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/host/aux_connectivity.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>

using namespace krona;
namespace eu = transport::euler;
namespace ph = boundary::physical;
namespace rec = numerics::reconstruction;
namespace ri = numerics::riemann;
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Space = typename Exec::memory_space;

mesh::device::MeshStorage<Exec> mesh(host, {}, exec); // FiniteVolume, including Halo.
const auto m = mesh.view();
const auto host_conn = mesh::host::build_aux_connectivity(host, mesh::AuxConnType::CellFaces);
mesh::device::AuxConnectivityStorage<Exec> conn(host_conn, exec);
eu::EulerPlan<Exec, Real> plan(host, m, conn.view(), comm, exec);

thermodynamics::IdealGas<Real> gas;
using Model = ph::Compressible<decltype(gas), ph::PerfectGasCharacteristics>;
const boundary::BoundaryLayout<Exec, Real> boundary_layout(host, m, exec);
ph::Manager<Model, Exec, Real> boundary(boundary_layout, Model{gas, {}});
boundary.set(0, ph::TotalConditionsInlet<Real>{110000, 300, {1, 0, 0}});
boundary.set(1, ph::PressureOutlet<Real>{100000, {}});
boundary.set(2, ph::SlipWall<Real>{});
// If unlabelled faces exist, assign them explicitly with set_unlabelled().
boundary.finalize();
ph::Status<Exec> boundary_status(exec);

fields::FieldStorage<StorageReal, 5, Exec> q(
    "primitive", boundary.required_field_size(), fields::FieldLocation::Custom, exec);
fields::FieldStorage<AccumulationReal, 5, Exec> r(
    "Euler residual", static_cast<std::size_t>(m.sizes.n_own),
    fields::FieldLocation::Custom, exec);
const auto values = fields::field_pack(q.view());
const auto output = fields::field_pack(r.view());
const auto physical = ph::primitive_components(q.view().component(0),
    q.view().component(1), q.view().component(2), q.view().component(3), q.view().component(4));
// Example finite initial state. Real applications derive owned primitives from their current U.
Kokkos::deep_copy(exec, q.view().component(0), StorageReal{100000});
Kokkos::deep_copy(exec, q.view().component(1), StorageReal{100});
Kokkos::deep_copy(exec, q.view().component(2), StorageReal{0});
Kokkos::deep_copy(exec, q.view().component(3), StorageReal{0});
Kokkos::deep_copy(exec, q.view().component(4), StorageReal{300});

halo::Exchange<StorageReal, Exec> primitive_halo(m, mesh.halo_plan(), comm, 81, exec);
primitive_halo.add_field(q.view());
primitive_halo.prepare();
const rec::StencilData<Space, Real> stencil(m, conn.view());
const auto first_order = eu::Reconstruction{rec::face_profiles(stencil, values, rec::FirstOrder{})};
const eu::EulerOperator euler{plan, eu::Thermodynamics{gas}, ri::Hllc<>{}};
eu::AssemblyStatus<Exec> status(exec);

const Real time = 0;
primitive_halo.exchange();
boundary_status.begin_async();
boundary.apply_async(physical, boundary_status, krona::boundary::scalar::NoFluxLaw{}, ph::Corrections<>{}, time);
boundary_status.check();
r.fill_async(exec, AccumulationReal{0});
status.begin_async();
euler.add_residual_async(first_order, output, boundary, status, eu::PreparedExterior{}, time);
const auto report = status.check(); // Completes kernels and checks this rank.
```

No heat-flux law is needed for this slip-wall example.

The field packs and `physical` binding reference the same allocations. The former supplies five indexed components to Euler; the latter supplies named primitive components to the physical manager. For separate existing fields, use `eu::primitive_fields(p.view(), velocity.view(), T.view())` for Euler and `ph::primitive_views(p.view(), velocity.view(), T.view())` for the manager. Residual fields can be bound with `eu::conservative_fields(rho_residual.view(), momentum_residual.view(), energy_residual.view())`. These adapters copy descriptors only.

Every labelled patch must have a condition, including patches empty on this rank. Unlabelled faces need an explicit condition when present. The manager must use the same mesh arrays and patch IDs/ranges as the plan. Its model must be both compressible and thermal, with thermodynamics consistent with the Euler state policy. The characteristic policy in this example enables total-condition inlets and pressure outlets; see `BOUNDARY-PHYSICAL.md` for their contracts/backflow data.

`EulerPlan` requires grouped `FaceCells`, `CellGeometry`, `FaceGeometry` and complete owned `CellFaces`. It builds a host patch catalog and checks the host and device boundary owners. Each patch already provides a contiguous range of faces. Construction is local and blocking. Preserve topology, local numbering, patch ranges and borrowed geometry. Rebuild the plan after changes.

## Configure higher-order reconstruction

Allocate gradients/factors once, alongside the fields above. This example uses limited linear reconstruction with one factor shared by all five variables:

```cpp
#include <krona/numerics/gradient/gradient_operator.hpp>

const auto cells = static_cast<std::size_t>(m.sizes.n_cells);
fields::FieldStorage<StorageReal, 15, Exec> g(
    "gradient/p_u_v_w_T", cells, fields::FieldLocation::Custom, exec);
fields::FieldStorage<StorageReal, 1, Exec> theta(
    "limiter/shared", cells, fields::FieldLocation::Custom, exec);
numerics::gradient::GradientOperator<Exec> gradient(m, conn.view(), {}, {}, exec);
const auto gradients = rec::gradient_pack(g.view()); // Component 3*c+axis.
const auto unlimited = rec::face_profiles(stencil, values, rec::Linear{gradients});
const auto factors = fields::pack(theta.view());
const auto high_order = eu::Reconstruction{rec::limited(unlimited, factors)};

halo::Exchange<StorageReal, Exec> reconstruction_halo(m, mesh.halo_plan(), comm, 82, exec);
reconstruction_halo.add_field(g.view());
reconstruction_halo.add_field(theta.view());
reconstruction_halo.prepare();

// Repeat after owned primitives change, e.g. at every time-integration stage.
primitive_halo.exchange();
boundary_status.begin_async();
boundary.apply_async(physical, boundary_status, krona::boundary::scalar::NoFluxLaw{}, ph::Corrections<>{}, time);
boundary_status.check();
gradient.apply_async(q.view(), g.view()); // Computes 15 owned derivative components.
rec::compute_limiters_async<Real>(exec, unlimited, factors, rec::BarthJespersen{}, rec::SharedMinimum{});
reconstruction_halo.exchange(); // Gradients and factors for real MPI ghosts.

r.fill_async(exec, AccumulationReal{0});
status.begin_async();
euler.add_residual_async(high_order, output, boundary, status, eu::EvaluatedExterior<>{}, time);
const auto high_order_report = status.check();
```

Gradient input/output use the same storage scalar. `g` contains derivatives of `p,u,v,w,T`, in that order; it does not contain gradients of conservative variables. Default least squares needs a nondegenerate 3-D stencil. This configuration needs no auxiliary diffusion geometry.

For independent factors, allocate `theta` with five components and select `rec::Componentwise{}` instead. The descriptor created by `rec::limited(profiles, factors)` reads the provided factors. Prepare them with the unlimited profiles or fill them with ones to use those profiles at full strength.

Choose the interior and physical-face formulas when binding the profile. For example, use TVD between cells and a one-sided linear trace at each physical face.

```cpp
const auto raw_tvd = rec::face_profiles(stencil, values, rec::Tvd{gradients}, rec::Linear{gradients});
rec::compute_limiters_async<Real>(exec, raw_tvd, factors, rec::BarthJespersen{}, rec::SharedMinimum{});
// Exchange factors and derivatives before evaluating remote traces.
const auto tvd = eu::Reconstruction{rec::limited(raw_tvd, factors)};
```

Keep the sampled fields, stencil, and prescribed boundary inputs current for the launch. The chosen formulas supply the interior traces. The closure below derives the exterior state. For explicit stencils, formula extension, and preparation contracts, see [Reconstruction](NUMERICS-RECONSTRUCTION.md).

## Physical faces and exterior states

Two states enter the boundary Riemann problem: the reconstructed **interior trace** and a temporary **exterior state**. The exterior state is local to the flux calculation. Physical-face primitive slots hold prepared boundary samples for closures and can also provide gradient or limiter samples.

| Closure | Exterior-state construction |
| --- | --- |
| `PreparedExterior{}` — default | Uses current stored face primitives for total-condition/supersonic inlet and pressure-outlet conditions. Mechanical walls mirror the reconstructed interior normal velocity relative to wall velocity. Other conditions, including symmetry/supersonic outlet/custom types, use `physical::evaluate(...).exterior`. |
| `EvaluatedExterior<>{}` | Re-evaluates the physical condition from the reconstructed interior trace. Mechanical walls/symmetry keep their inviscid mirror treatment. It accepts optional law/correction members for coupled/custom conditions. |
| Custom closure | Device-callable `(context, typed_condition, inside, prepared_face, model) -> physical::Primitive<Real,true>`. Must compile for all manager alternatives and return an admissible exterior state. |

`EvaluatedExterior` can work with a cell-only pack when the reconstruction fits in that pack. Prepare face slots separately if the gradient or limiter requires them. With `PreparedExterior`, the application may overwrite a prepared inlet or outlet profile after `boundary.apply_async()` and Euler will use those stored values.

Wall temperature/heat-flux data describe thermal/viscous traces; they do not add conduction to this inviscid operator. Stationary mechanical walls have no inviscid mass/energy flux. If `boundary.apply_async()` must fill a `HeatFluxWall`, supply its thermal flux law there, even though the Euler mechanical mirror bypasses that thermal law. Use the same evaluation time for preparation and residual calls.

## EOS, solvers and fallback behaviour

`Thermodynamics{eos}` checks finite positive `p,T` and finite velocity before EOS evaluation. The EOS supplies host `validate()` and device `from_pT(p,T)` results compatible with `riemann::from_thermodynamics()` (`density`, `pressure`, `internal_energy`, `sound_speed`). A custom EOS with a narrower domain should supply `Thermodynamics{eos, domain}`: device-callable `domain(eos, primitive)` must reject inadmissible candidates **before** the EOS is called. Default `PositivePT` adds no model-specific restriction. This guard covers Euler state preparation, not separate EOS calls inside physical boundary evaluation. Generic EOS evaluation retains its own error policy; an EOS abort cannot be converted into a reconstruction fallback afterward.

A custom state policy can replace `Thermodynamics`: device-callable `(physical::Primitive<Real,true>) -> riemann::State<Real>`, with optional host `validate()`. Use consistent EOS/energy-reference conventions across this policy, the physical manager and the Riemann solver.

Pass `Rusanov`, `Hll<>`, `Hllc<>` or `Roe{gas.gamma}` to `EulerOperator{plan, state_policy, solver}`; HLLC is the default. Roe additionally requires states compatible with its calorically perfect-gas convention; arbitrary EOS compatibility is not implied. Include the selected solver header. Runtime selection can dispatch a templated launch on the host; each operator/kernel then holds a concrete solver type. Custom solvers follow `(left_state, right_state, unit_normal) -> riemann::Result<Real>` and return flux density, not area-integrated flux. See `NUMERICS-RIEMANN.md`.

Both built-in closures accept profiled physical conditions dispatched by the manager. The evaluated closure samples the condition at the current face and time. The prepared inlet and outlet paths use the published face values. The mechanical wall path samples wall velocity independently of any temperature or heat-flux profile when the provider exposes it.

Custom policies must be device-copyable and return consistent results for fixed inputs. On MPI interfaces, publish every accessed primitive/reconstruction sample and use matching physics parameters; Euler does not reconcile rank-local model/data disagreement.

For each evaluated side:

1. Invalid base cell thermodynamics is an error; reconstruction cannot repair it.
2. An inadmissible reconstructed primitive/EOS state falls back to the **whole base side state**, not selected components. Boundary interior traces have the same fallback.
3. An invalid exterior state is an error. The chosen Riemann solver may separately use its own fallback; a failed result is an error.

`eu::AssemblyStatus` is separate from scalar transport status. `check()` returns this rank's `reconstruction_fallbacks` and `riemann_fallbacks`, and terminates on invalid cell/exterior states, rejected fluxes, invalid Jacobian contributions or nonfinite/unrepresentable residual additions. Reduce the two counters explicitly if a global report is needed. Counters count evaluations, **not unique faces**: owned/owned faces are evaluated from both rows, and each side can fall back.

## Split calls and range contracts

```cpp
r.fill_async(exec, AccumulationReal{0});
status.begin_async();
euler.add_inner_residual_async(high_order, output, status,
    {0, m.sizes.n_internal_faces}); // Owned/owned faces.
// Complete any remaining producers/exchanges before accessing their data.
euler.add_inner_residual_async(high_order, output, status,
    {m.sizes.n_internal_faces, m.sizes.n_inner_faces}); // MPI interfaces.
euler.add_boundary_residual_async(high_order, output, boundary, status,
    eu::EvaluatedExterior<>{}, time);
(void)status.check();
```

`FaceRange` is half-open; default end `-1` selects all inner faces (owned/owned plus owned/ghost). `CellRange` is a half-open owned-row interval; default end selects `n_own`. It restricts output rows, not which neighbour/stencil values they may read. Output storage must still cover all owned rows. Unselected output/tail entries are untouched. Split ranges must not overlap unless repeated addition is intended. Readiness must cover the **complete selected reconstruction stencil**, including wider samples; an owned/owned face does not automatically guarantee an entirely owned stencil.

Inner-only validation requires primitive storage for all real cells. Full and boundary calls require extended storage for `PreparedExterior`, custom closures and custom boundary Jacobians. The reconstruction can also require face samples for a wider stencil. Required gradient/factor/Hessian/moment providers cover at least real cells when they expose sizes; custom indices and wider-stencil coverage remain caller contracts. Inputs and residual components must not overlap, including partial/strided aliases. Checks catch duplicate residual starting pointers and some exact input/output aliases, not arbitrary memory overlap.

## Compact implicit assembly

Initialize Tpetra on the assembly communicator before constructing the native graph. Use the full Euler residual on the right-hand side and solve for the conservative density increment
$\delta U=(\delta\rho,\delta(\rho u),\delta(\rho v),\delta(\rho w),\delta(\rho E))$.
The matrix methods add the Jacobian of the **outward flux sum**, with sign opposite to the derivative of the residual. Add the physical or pseudo-time diagonal separately when the integration method calls for it.

```cpp
#include <krona/transport/euler/assembly.hpp>
#include <krona/linalg/field_transfer.hpp>

// Set up once, with the same host cell/face numbering as EulerPlan.
const linalg::FaceAssemblyPlan assembly(host, comm, exec);
linalg::BsrMatrix<SolverReal, Exec> matrix(assembly.pattern(), 5);
linalg::Vector<SolverReal, Exec> rhs(matrix.distribution());

// Repeat after halo, boundary-slot and reconstruction preparation.
const auto closure = eu::EvaluatedExterior<>{};
const auto inner = eu::FrozenRusanov{gas};
const auto target = eu::make_jacobian_view(matrix, assembly, exec);
Kokkos::deep_copy(exec, target.blocks, SolverReal{0});
r.fill_async(exec, AccumulationReal{0});
status.begin_async();
euler.add_residual_async(first_order, output, boundary, status, closure, time);
euler.add_jacobian_async(target, first_order, boundary, status, inner, closure, time);
(void)status.check();

linalg::copy_from_fields_async(rhs, output, exec);
matrix.commit();
rhs.commit();
```

The same sequence accepts a higher-order reconstruction in both calls. Add a time residual or matrix contribution as needed before committing. The target borrows native blocks addressed as `(entry, equation, variable)`. The adapter checks that the matrix shares the assembly graph and has block width five. The caller keeps the assembly plan's local numbering and halo topology consistent with the Euler plan. Keep input and matrix storage disjoint. An application with another backend may provide `JacobianView{blocks, entries}` with the same native-row and entry-map contract.

### Inner faces

`FrozenRusanov{eos}` uses cell-centred states. For a unit normal it returns
$D_L = (A_n(U_L)+\lambda I)/2$ and
$D_R = (A_n(U_R)-\lambda I)/2$, where
$A_n = \partial F_n/\partial U$ and $\lambda=\max(|u_{n,L}|+c_L, |u_{n,R}|+c_R)$.
The face speed is recomputed at assembly and held fixed in the derivative. The stencil contains the cell and its face neighbours. Reconstruction and limiter data stay in the full residual.

Supply any device-callable `linearization(left, right, unit_normal)` returning `eu::FaceJacobians<Real>{left_block, right_block}` to replace the inner linearization. Both blocks contain derivatives of the flux density with respect to conservative densities. The assembler multiplies by area. In the owner row, the blocks go to the diagonal and neighbour column with a positive sign. In an owned neighbour row, they go to the owner column and diagonal with a negative sign. Across an MPI interface, each rank writes its owned row. A custom flux linearization must be consistent under reversal of the normal and exchange of the two states.

`add_inner_jacobian_async(target, values, status, inner, faces, cells)` accepts the primitive pack directly and shares the residual's owned-row and inner-face ranges. `add_boundary_jacobian_async(target, reconstruction, boundary, status, closure, time, cells)` adds only physical boundary terms. These calls and the combined `add_jacobian_async` are additive and leave clearing, synchronization and commit to the caller.

### Physical boundaries

Use the same reconstruction, numerical flux, closure and time for the residual and boundary linearization. The default `FiniteDifferenceBoundary` differentiates the complete numerical boundary flux in the five primitive components of the interior trace, then applies the analytic $\partial W_{\mathrm{cell}}/\partial U_{\mathrm{cell}}$ at the owner cell. The reconstruction increment and any prepared face values supplied to the closure are held fixed. A smooth branch uses central differences. A valid one-sided stencil is used when a probe leaves the admissible state or boundary domain. Near a change of characteristic or Riemann branch, the result is a local numerical slope. Probes are evaluated on the same device execution space as the residual.

With `EvaluatedExterior`, the characteristic outlet/inlet and mirror responses to the interior trace are included. Each flux evaluation uses the original condition with the same face context and time. A profile provider should return the same prescribed values throughout these evaluations. With `PreparedExterior`, a published inlet/outlet face state stays fixed throughout the derivative. Use `EvaluatedExterior` for coupled boundary states whose response is needed in the matrix.

The boundary policy is independently replaceable via the final argument of either boundary assembly call. It receives `(context, condition, model, closure, states, state_law, solver)` and returns a `eu::BoundaryJacobian<Real>{block, valid}`. The states include the owner's thermodynamic state, current primitive interior trace and prepared face slot. Return the unit-area derivative by owner conservative density. It is added to the owner diagonal. This also supports a direct analytic boundary-flux derivative.

`krona/numerics/riemann/state_jacobians.hpp` exposes `primitive_from_conservative_jacobian`, `conservative_from_primitive_jacobian` and `physical_flux_jacobian`. They use the EOS derivatives in density and specific internal energy. `IdealGas` already supplies them. The default boundary policy reads the EOS from `state_law.eos`. State policies without this interface can use a custom boundary derivative. All EOS and boundary domain checks apply to the perturbed states too.

## Completion, storage and the next stage

`EulerPlan` owns the host patch catalog and borrows the mesh/connectivity arrays and communicator. `EulerOperator` stores the plan, state policy and solver by value. Inner faces gather by owned cell row. Each boundary kernel runs over one patch with one iteration per face and atomically adds to the owner's five residual components or the 25 entries of its diagonal BSR block. Patch launches are ordered on one execution instance. No face-work buffer is allocated.

Boundary additions use the destination scalar precision. When several faces update the same cell within a patch, accumulation order can vary and affect floating-point round-off.

Reconstruction/component packs borrow data pointers; field owners must outlive work. Managed members in custom providers retain their allocations, unmanaged ones borrow. `AssemblyStatus` owns a small device buffer and host mirror. `Real` controls state/flux arithmetic, `Accumulator` defaults to `AccumulationReal` for row sums, and the output pack controls stored residual precision.

| Operation | Completion |
| --- | --- |
| Plan / physical-manager construction | Local and blocking; validates/prepares maps/geometry. No MPI collective. |
| `boundary.apply_async()`, gradient/limiter calls, Euler `add_*_async()` | Queue kernels. Euler performs no reset, fence, MPI/halo exchange or time update. |
| `FaceAssemblyPlan`, BSR construction | Collective and blocking. The native view adapter can synchronize native storage access. |
| `halo.exchange()` | Blocking convenience exchange; explicit ownership remains with the application. |
| `status.begin_async()` | Queues diagnostic reset; does not clear residuals. |
| `status.check()` | Copies diagnostics to host, fences its instance and checks the flags on this rank. |

Use the same execution instance for status and contributions. Order producers/consumers on that instance; synchronize other instances and external MPI before use. Complete valid field/boundary/model setup before submission; a failed batch is not rolled back. A pending status destructor fences but leaves error reporting to `check()`.

After successful completion, a time integrator may update conservative **densities**, for example `U_new = U_old + dt * R / V` for explicit Euler. **Do not add conservative residual components directly to `p,u,v,w,T`.** Recover velocity and specific internal energy from the updated `rho, rho*u, rho*E`, invert the EOS for `p,T`, then repeat primitive publication/boundary/reconstruction preparation for the next stage. Time-step choice, conservative-state admissibility and all updates remain application responsibilities.