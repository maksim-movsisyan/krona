# Euler transport

`krona::transport::euler` evaluates the inviscid finite-volume residual of the compressible Euler equations on a static 3-D mesh. It reconstructs primitive variables, prepares thermodynamics through an EOS and calls a Riemann solver. It provides no matrix/Jacobian assembly, viscous terms, CFL calculation or time integrator. Trilinos is not required.

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

Primitive storage covers real cells, including MPI ghosts. Boundary/full calls additionally require one physical-face slot per boundary face:

```cpp
extended_index = n_cells + face - n_inner_faces;
```

These are primitive face samples, not exterior ghost cells. Output needs only the owned-cell prefix. All packs have exactly five components; their components share a scalar type/memory space, but input and output scalar types may differ. Accessible rank-one views, including strided components, are supported.

Accessed base primitives must have finite positive `p,T`, finite velocity and an admissible finite EOS/conservative state after conversion to computation `Real`. Initialize residuals to finite values before additive contributions. A custom pack supplies `components`, `value_type`, `memory_space`, `size()` and device-callable `(cell, component)` access, writable for outputs.

## Main pipeline

Assume active MPI/Kokkos, a prepared `mesh::host::MeshPart host`, and matching `comm`/`exec`. Run host setup under the project's `mpi::guarded` error boundary. The boundary example assumes patch `0` is an inlet facing negative x, patch `1` is an outlet and patch `2` contains slip walls; replace assignments for the actual mesh.

```cpp
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
ph::Manager<Model, Exec, Real> boundary(host, m, Model{gas, {}}, exec);
boundary.set(0, ph::TotalConditionsInlet<Real>{110000, 300, {1, 0, 0}});
boundary.set(1, ph::PressureOutlet<Real>{100000, {}});
boundary.set(2, ph::SlipWall<Real>{});
// If unlabelled faces exist, assign them explicitly with set_unlabelled().

fields::FieldStorage<StorageReal, 5, Exec> q(
    "primitive", boundary.required_field_size(), fields::FieldLocation::Custom, exec);
fields::FieldStorage<AccumulationReal, 5, Exec> r(
    "Euler residual", static_cast<std::size_t>(m.sizes.n_own),
    fields::FieldLocation::Custom, exec);
const auto values = rec::field_pack(q.view());
const auto output = rec::field_pack(r.view());
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
rec::MeshStencil<Space, Real> stencil(m, conn.view());
const auto first_order = eu::Reconstruction{rec::FirstOrder{}, rec::data(values), stencil};
const eu::EulerOperator euler{plan, eu::Thermodynamics{gas}, ri::Hllc<>{}};
eu::AssemblyStatus<Exec> status(exec);

const Real time = 0;
primitive_halo.exchange();
boundary.apply_async(physical, krona::boundary::scalar::NoFluxLaw{}, ph::Corrections<>{}, time);
r.fill(exec, AccumulationReal{0});
status.begin_async();
euler.add_residual_async(first_order, output, boundary, status, eu::PreparedExterior{}, time);
const auto report = status.check(comm); // Completes kernels and checks all ranks.
```

No heat-flux law is needed for this slip-wall example.

The field packs and `physical` binding reference the same allocations. The former supplies five indexed components to Euler; the latter supplies named primitive components to the physical manager. For separate existing fields, use `eu::primitive_fields(p.view(), velocity.view(), T.view())` for Euler and `ph::primitive_views(p.view(), velocity.view(), T.view())` for the manager. Residual fields can be bound with `eu::conservative_fields(rho_residual.view(), momentum_residual.view(), energy_residual.view())`. These adapters copy descriptors only.

Every labelled patch must have a condition, including patches empty on this rank. Unlabelled faces need an explicit condition when present. The manager must use the same mesh arrays and patch IDs/ranges as the plan. Its model must be both compressible and thermal, with thermodynamics consistent with the Euler state policy. The characteristic policy in this example enables total-condition inlets and pressure outlets; see `BOUNDARY-PHYSICAL.md` for their contracts/backflow data.

`EulerPlan` requires grouped `FaceCells`, `CellGeometry`, `FaceGeometry` and complete owned `CellFaces`. It uses host patch metadata/face owners to construct reusable boundary incidence grouped by `(patch, owned cell)`, uploads that map and validates it against device ownership. Construction is local and blocking. This host input has a role beyond count checks. Preserve topology, local numbering, patch ranges and borrowed geometry; rebuild after changes.

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
const auto unlimited = rec::data(values, gradients);
const auto data = rec::data(values, gradients, rec::field_pack(theta.view()));
const auto high_order = eu::Reconstruction{rec::LimitedLinear{}, data, stencil};

halo::Exchange<StorageReal, Exec> reconstruction_halo(m, mesh.halo_plan(), comm, 82, exec);
reconstruction_halo.add_field(g.view());
reconstruction_halo.add_field(theta.view());
reconstruction_halo.prepare();

// Repeat after owned primitives change, e.g. at every time-integration stage.
primitive_halo.exchange();
boundary.apply_async(physical, krona::boundary::scalar::NoFluxLaw{}, ph::Corrections<>{}, time);
gradient.apply(q.view(), g.view()); // Computes 15 owned derivative components.
rec::compute_limiters_async<Real>(exec, stencil, unlimited, rec::field_pack(theta.view()),
    rec::BarthJespersen{}, rec::SecondOrderUpwind{}, rec::SharedMinimum{});
reconstruction_halo.exchange(); // Gradients and factors for real MPI ghosts.

r.fill(exec, AccumulationReal{0});
status.begin_async();
euler.add_residual_async(high_order, output, boundary, status, eu::EvaluatedExterior<>{}, time);
const auto high_order_report = status.check(comm);
```

Gradient input/output use the same storage scalar. `g` contains derivatives of `p,u,v,w,T`, in that order; it does not contain gradients of conservative variables. Default least squares needs a nondegenerate 3-D stencil. This configuration needs no auxiliary diffusion geometry.

For independent factors, allocate `theta` with five components and select `rec::Componentwise{}` instead. If factors are omitted from `Data`, `UnitFactors` returns `1`: wrapping a scheme in `Limited` alone does not calculate or activate limiting.

| Interior scheme | Required data / boundary choice |
| --- | --- |
| `FirstOrder` | Primitive values only. |
| `SecondOrderUpwind` | Gradients; unlimited linear trace. |
| `LimitedLinear`, `Limited<Scheme>` | Required scheme inputs plus externally prepared factors. Prepare factors using that same **unlimited** scheme. |
| `MusclKappa`, `Tvd` | Gradients and other-cell values. Explicitly choose the intended boundary scheme. |
| `Quadratic` | Gradients, true Hessians and central cell moments. |
| `Quick` / custom wider scheme | Explicit side stencils, valid samples/weights and sufficient halo data; no automatic stencil search. |

`eu::Reconstruction{scheme, data, stencil}` uses the same configured scheme/parameters on inner faces and boundary interior traces. Unlike the scalar convection factory, **Euler does not automatically replace MUSCL/TVD at the boundary**. To avoid reading a boundary slot as the scheme's `other` sample, choose a one-sided trace explicitly:

```cpp
const auto tvd = eu::Reconstruction{rec::Tvd<>{}, data, stencil, rec::LimitedLinear{}};
```

Here TVD handles inner faces and the prepared cell factor limits the one-sided linear boundary trace. A custom stencil supplies `face<Scheme>(f)` or `face(f)` with owner/neighbour sides and `boundary_side(f)` suitable for the selected boundary scheme. Keep scheme geometry, factors and all accessed samples consistent; see `NUMERICS-RECONSTRUCTION.md`.

Accessed reconstruction inputs must be finite, factors must lie in `[0,1]`, and indices/geometry must satisfy the scheme's stencil contract. The admissibility fallback below does not replace storage/stencil validation.

Prefer extending schemes/data/stencils inside `eu::Reconstruction`. Replacing that adapter requires its full protocol: `scheme_type`, `boundary_scheme_type`, `.data` (including `values` and `factors`), `validate<Exec>(cells, extended, with_boundary)`, `interior<Real>(face)` returning five left/right values and `boundary<Real>(face)` returning five interior values.

## Physical faces and exterior states

Two states enter the boundary Riemann problem: the reconstructed **interior trace** and a temporary **exterior state**. Euler does not allocate/store exterior ghost cells. Fill physical-face primitive slots before a full/boundary call; they also provide gradient/limiter samples.

| Closure | Exterior-state construction |
| --- | --- |
| `PreparedExterior{}` — default | Uses current stored face primitives for total-condition/supersonic inlet and pressure-outlet conditions. Mechanical walls mirror the reconstructed interior normal velocity relative to wall velocity. Other conditions, including symmetry/supersonic outlet/custom types, use `physical::evaluate(...).exterior`. |
| `EvaluatedExterior<>{}` | Re-evaluates the physical condition from the reconstructed interior trace. Mechanical walls/symmetry keep their inviscid mirror treatment. It accepts optional law/correction members for coupled/custom conditions. |
| Custom closure | Device-callable `(context, typed_condition, inside, prepared_face, model) -> physical::Primitive<Real,true>`. Must compile for all manager alternatives and return an admissible exterior state. |

`EvaluatedExterior` changes the closure calculation; it does not write face slots or remove the API's extended-storage requirement. Keep the face preparation step. A prepared inlet/outlet face profile may be overwritten by application code after `boundary.apply_async()` and consumed by `PreparedExterior`; re-evaluation generally derives the state from the condition instead.

Wall temperature/heat-flux data describe thermal/viscous traces; they do not add conduction to this inviscid operator. Stationary mechanical walls have no inviscid mass/energy flux. If `boundary.apply_async()` must fill a `HeatFluxWall`, supply its thermal flux law there, even though the Euler mechanical mirror bypasses that thermal law. Use the same evaluation time for preparation and residual calls.

## EOS, solvers and fallback behaviour

`Thermodynamics{eos}` checks finite positive `p,T` and finite velocity before EOS evaluation. The EOS supplies host `validate()` and device `from_pT(p,T)` results compatible with `riemann::from_thermodynamics()` (`density`, `pressure`, `internal_energy`, `sound_speed`). A custom EOS with a narrower domain should supply `Thermodynamics{eos, domain}`: device-callable `domain(eos, primitive)` must reject inadmissible candidates **before** the EOS is called. Default `PositivePT` adds no model-specific restriction. This guard covers Euler state preparation, not separate EOS calls inside physical boundary evaluation. Generic EOS evaluation retains its own error policy; an EOS abort cannot be converted into a reconstruction fallback afterward.

A custom state policy can replace `Thermodynamics`: device-callable `(physical::Primitive<Real,true>) -> riemann::State<Real>`, with optional host `validate()`. Use consistent EOS/energy-reference conventions across this policy, the physical manager and the Riemann solver.

Pass `Rusanov`, `Hll<>`, `Hllc<>` or `Roe{gas.gamma}` to `EulerOperator{plan, state_policy, solver}`; HLLC is the default. Roe additionally requires states compatible with its calorically perfect-gas convention; arbitrary EOS compatibility is not implied. Include the selected solver header. Runtime selection can dispatch a templated launch on the host; each operator/kernel then holds a concrete solver type. Custom solvers follow `(left_state, right_state, unit_normal) -> riemann::Result<Real>` and return flux density, not area-integrated flux. See `NUMERICS-RIEMANN.md`.

Custom policies must be device-copyable and return consistent results for fixed inputs. On MPI interfaces, publish every accessed primitive/reconstruction sample and use matching physics parameters; Euler does not reconcile rank-local model/data disagreement.

For each evaluated side:

1. Invalid base cell thermodynamics is an error; reconstruction cannot repair it.
2. An inadmissible reconstructed primitive/EOS state falls back to the **whole base side state**, not selected components. Boundary interior traces have the same fallback.
3. An invalid exterior state is an error. The chosen Riemann solver may separately use its own fallback; a failed result is an error.

`eu::AssemblyStatus` is separate from scalar transport status. `check(comm)` returns global `reconstruction_fallbacks` and `riemann_fallbacks`, and terminates on invalid cell/exterior states, rejected fluxes or nonfinite/unrepresentable residual additions. Counters count evaluations, **not unique faces**: owned/owned faces are evaluated from both rows, and each side can fall back.

## Split calls and range contracts

```cpp
r.fill(exec, AccumulationReal{0});
status.begin_async();
euler.add_inner_residual_async(high_order, output, status,
    {0, m.sizes.n_internal_faces}); // Owned/owned faces.
// Complete any remaining producers/exchanges before accessing their data.
euler.add_inner_residual_async(high_order, output, status,
    {m.sizes.n_internal_faces, m.sizes.n_inner_faces}); // MPI interfaces.
euler.add_boundary_residual_async(high_order, output, boundary, status,
    eu::EvaluatedExterior<>{}, time);
(void)status.check(comm);
```

`FaceRange` is half-open; default end `-1` selects all inner faces (owned/owned plus owned/ghost). `CellRange` is a half-open owned-row interval; default end selects `n_own`. It restricts output rows, not which neighbour/stencil values they may read. Output storage must still cover all owned rows. Unselected output/tail entries are untouched. Split ranges must not overlap unless repeated addition is intended. Readiness must cover the **complete selected reconstruction stencil**, including wider samples; an owned/owned face does not automatically guarantee an entirely owned stencil.

Inner-only validation requires primitive storage for all real cells. Full/boundary validation requires extended storage even with first-order or evaluated closure. Required gradient/factor/Hessian/moment providers cover at least real cells when they expose sizes; custom indices and wider-stencil coverage remain caller contracts. Inputs and residual components must not overlap, including partial/strided aliases. Checks catch duplicate residual starting pointers and some exact input/output aliases, not arbitrary memory overlap.

## Completion, storage and the next stage

`EulerPlan` owns its small host patch metadata and managed device boundary-incidence arrays; copies share device maps. Mesh/connectivity arrays and communicator are borrowed. `EulerOperator` stores the plan, state policy and solver by value, with no state or face-flux arrays. Inner row gathering recomputes shared face fluxes; boundary work gathers faces per `(patch, owned cell)`, with patch launches ordered on one execution instance. No residual atomics or per-call face-work allocation are needed; diagnostic flags/counters use atomics.

Reconstruction/component packs borrow data pointers; field owners must outlive work. Managed members in custom providers retain their allocations, unmanaged ones borrow. `AssemblyStatus` owns a small device buffer and host mirror. `Real` controls state/flux arithmetic, `Accumulator` defaults to `AccumulationReal` for row sums, and the output pack controls stored residual precision.

| Operation | Completion |
| --- | --- |
| Plan / physical-manager construction | Local and blocking; validates/prepares maps/geometry. No MPI collective. |
| `boundary.apply_async()`, gradient/limiter calls, Euler `add_*_async()` | Queue kernels; Euler performs no reset, fence, MPI/halo exchange or time update. |
| `halo.exchange()` | Blocking convenience exchange; explicit ownership remains with the application. |
| `status.begin_async()` | Queues diagnostic reset; does not clear residuals. |
| `status.check(comm)` | Copies diagnostics to host, fences its instance and performs an MPI all-reduce. |

Use the same execution instance for status and contributions. Order producers/consumers on that instance; synchronize other instances and external MPI before use. All ranks, including empty ranks, enter collective checks in the same order. Complete valid field/boundary/model setup before submission; a failed batch is not rolled back. A pending status destructor fences but does not perform the collective error check.

After successful completion, a time integrator may update conservative **densities**, for example `U_new = U_old + dt * R / V` for explicit Euler. **Do not add conservative residual components directly to `p,u,v,w,T`.** Recover velocity and specific internal energy from the updated `rho, rho*u, rho*E`, invert the EOS for `p,T`, then repeat primitive publication/boundary/reconstruction preparation for the next stage. Time-step choice, conservative-state admissibility and all updates remain application responsibilities.
