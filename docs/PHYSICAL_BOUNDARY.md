# Physical boundary conditions

`krona::boundary::physical` evaluates primitive states at physical boundary-face
centroids. Its manager follows the scalar boundary module's host patch dispatch
and asynchronous Kokkos execution. One shared TOML file contains patch metadata,
physical data and independent scalar sections.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| `Manager<Exec,Real,CustomConditions...>` | Owns its patch catalogue and typed conditions; borrows geometry and submits kernels |
| Condition types | Prescribed mechanical/thermal data and typed face-state operation |
| `CompressibleClosure<EOS,OpenBoundary>` | Completes compressible face states using the supplied EOS and open-boundary policy |
| `IncompressibleClosure<Real,Thermal>` | Constant-density face-state completion, optionally with temperature |
| `thermodynamics::IdealGas<Real>` | Reusable calorically perfect thermodynamics, without boundary types or formulas |
| Field adapters | Borrow existing primitive buffers and convert storage/arithmetic precision |

The manager stores no EOS, closure or field adapter. A closure is supplied to each
`apply_async` or `evaluate_async` call. `for_each_face_async` needs neither fields
nor a closure, so gradient and matrix kernels can use the same patch dispatch.
Each scalar manager and the main-flow manager own independent catalogues.
The physical module uses condition-independent geometry descriptors in
`boundary/geometry_view.hpp` and does not depend on scalar conditions or their parser.

## Storage and execution contracts

- Faces are grouped into owned-owned, owned-MPI-ghost and physical boundary
  intervals. Each local patch has one contiguous absolute interval `[begin,end)`.
  The shared `PatchCatalog` scans host labels once and stores no per-face patch map.
- Patch IDs index `MeshPart::patches`; names must match exactly. Every global
  labelled patch needs a condition, including patches empty on the current rank.
  Unlabelled physical faces require an explicit condition.
- `requirements()` returns `MeshData::FaceCells | MeshData::Geometry`. Face owners
  are owned cells, normals are outward unit vectors, and face areas and normal
  owner-to-face distances are positive.
- Fields contain at least `n_cells + n_faces - n_inner_faces` entries. `n_cells`
  includes owned cells and MPI ghosts. Face `f` uses extended slot
  `n_cells + f - n_inner_faces`.
  **This slot contains the value at the face centroid, not a mirrored ghost cell.**
- `apply_async` writes only physical boundary slots. Owned cells, MPI ghosts and
  extra trailing storage are unchanged. Primitive component buffers must not
  overlap. No boundary storage is allocated by the manager.
- Geometry setup is local and blocking. Submission uses the manager's execution
  instance and inserts no fence or MPI communication. Some CPU backends may
  execute synchronously; asynchronous submission does not promise overlap.
- Borrowed mesh, field, condition and callback buffers must outlive submitted
  work. Use the same execution instance for dependent kernels, or synchronize
  explicitly. `manager.fence()` completes work; destruction does not synchronize.
- Descriptors are copied into submitted kernels. Host `set` calls between
  submissions affect subsequent work; they do not alter already submitted work.
- Arithmetic uses `Real`, normally `ComputeReal`; storage may use another
  floating-point type. Positions, displacements, area and normal distance use
  `GeometryReal`. Coordinate differences are computed before conversion to fluid
  arithmetic. The normal and time use fluid arithmetic precision.
- Before filling fields, the manager checks closure parameters and condition
  capabilities on the host, including globally configured empty local patches.
  This performs no collective or application-time device reduction.

MPI halo exchange and global diagnostic reductions belong to the caller. Input
primitives must be admissible and consistent with the EOS within storage precision;
output values must be representable in their destination storage type.

## States and conditions

`State<Real,true>` contains `density`, `pressure`, `velocity` (`Vector3<Real>`) and
`temperature`. `State<Real,false>` has no temperature member. Conditions return
`Evaluation<state_type>` containing `state` and `status`; `valid()` means
`status == Status::Success`.

Let `n` point outward, `u_n = u dot n`, and `P` denote the face owner. Negative
normal velocity is inflow. Compressible regime checks use normal Mach number;
large tangential speed alone does not make a face supersonic.

| Condition | Face-state behavior |
| --- | --- |
| `VelocityInlet<Real>` | Prescribed velocity and optional static temperature; pressure comes from the selected closure |
| `TotalInlet<Real>` | Stagnation pressure/temperature and unit inflow direction; requires a total-inlet closure |
| `SupersonicInlet<Real>` | Prescribed static pressure, temperature and velocity; requires supersonic normal inflow |
| `PressureOutlet<Real>` | Prescribed static pressure for subsonic outflow; remaining primitives follow the selected closure |
| `InletOutlet<Real>` | Pressure outlet with explicitly prescribed static inflow data on detected backflow |
| `SupersonicOutlet` | Copies the owner state for strict supersonic normal outflow |
| `AdiabaticWall<Real>` | No-slip wall velocity, zero by default; extrapolated/corrected temperature |
| `IsothermalWall<Real>` | No-slip wall velocity and fixed positive static temperature |
| `SlipWall<Real>` | `u_f = u_P - n*((u_P-u_wall) dot n)`; optional thermal data |
| `Symmetry` | `u_f = u_P - n*(u_P dot n)` with pressure extrapolation and thermal completion |
| `Extrapolate` | Copies the admissible owner primitive state without a Mach restriction |

Walls extrapolate owner pressure. A compressible closure recomputes density through
its EOS when temperature changes. A moving-wall velocity does not move mesh geometry
or implement ALE fluxes. Slip preserves the owner's tangential velocity; specifying
wall tangential velocity alone does not create a no-slip condition.

`Temperature<Real>` used by velocity inlets, slip walls and backflow data supports
`ThermalKind::Extrapolated` and `FixedTemperature`. Extrapolation uses
`T_P + context.temperature_tangential_increment`. Adiabatic walls and symmetry use
the same correction; isothermal walls ignore it. The default increment is zero,
giving first-order temperature extrapolation.

For zero normal temperature derivative on a skew face, use

```cpp
const auto correction = bc::TemperatureGradientCorrection{gx, gy, gz};
boundary.apply_async(primitive, closure, time, correction);
```

The views contain owned-cell temperature gradients. This adds
`grad(T)_P dot ((F-C_P) - distance*n)`. Gradients must be initialized before use;
this does not bootstrap a coupled gradient reconstruction. A custom transform
implements `Context<Real>(Context<Real>) const` and is accepted by all three
submission APIs after `time`. It must preserve face/owner/storage indices;
validity of changed auxiliary data and borrowed buffers belongs to the transform.

## Thermodynamics and boundary closures

`thermodynamics::IdealGas<Real>` defaults to `gamma = 1.4` and
`gas_constant = 287.05`. Its API provides `density(p,T)`, `pressure(rho,T)`,
`internal_energy(T)`, `enthalpy(T)`, inverse temperature conversions, `cv()`, `cp()`,
`sound_speed(rho,T)`, `valid(p,T)` and `valid_parameters()`.

The gas is calorically perfect: `rho = p/(R*T)`, `cv = R/(gamma-1)` and
`cp = gamma*cv`. Pressure and temperature are absolute. In SI units, pressure is
Pa, temperature K and the specific gas constant J/(kg K). EOS methods have no
boundary dependency and perform no MPI calls. Parameter/domain predicates report
validity; callers must not evaluate thermodynamic formulas outside their domains.

The generic compressible closure requires an EOS with `real_type`, `valid(p,T)`,
`density(p,T)` and `sound_speed(rho,T)`. `valid_parameters()` is optional. This
protocol assumes density can be completed from prescribed pressure/temperature;
additional mixture variables or other state representations need a custom closure
and appropriate field adapters.

### Open-boundary policies

The policy is selected explicitly as a template argument:

```cpp
using EOS = krona::thermodynamics::IdealGas<Real>;
bc::CompressibleClosure<EOS> prescribed{eos};
bc::CompressibleClosure<EOS, bc::PerfectGasCharacteristics> characteristic{eos};
```

| Policy | Velocity inlet | Subsonic pressure outlet | Total inlet |
| --- | --- | --- | --- |
| `PrescribedOpenBoundary` (default) | Copies owner pressure; prescribes velocity and optional temperature | Prescribes pressure; copies owner velocity and temperature; recomputes density | Unsupported |
| `PerfectGasCharacteristics` | Linearized outgoing acoustic compatibility | Preserves owner entropy and outgoing acoustic invariant | Solves stagnation/characteristic compatibility |

Both compressible policies diagnose backflow at a pressure outlet and extrapolate
the entire state when the owner has normal Mach at least one. Explicit supersonic
conditions require strict supersonic flow and diagnose sonic states.

For `PerfectGasCharacteristics`, define `g = gamma-1`,
`a = sqrt(gamma*R*T)` and `J+ = u_n + 2*a/g`. Its velocity inlet uses

`p_f = p_P + rho_P*a_P*(u_n,P - u_n,f)`.

Its subsonic pressure outlet uses

```text
T_f   = T_P*(p_f/p_P)^(g/gamma)
rho_f = p_f/(R*T_f)
u_n,f = u_n,P + 2*(a_P-a_f)/g
```

Tangential velocity is copied. For a stagnation inlet, write `u_f = V*d` with a unit
inward direction, `mu = d dot n < 0`. Given `p0,T0`, solve

`mu*V + 2*sqrt(g*(cp*T0 - V*V/2))/g = J+_P`,

then compute `T_f = T0 - V*V/(2*cp)` and
`p_f = p0*(T_f/T0)^(gamma/g)`. A rationalized quadratic root handles weak inflow
without device iteration; the unsquared equation is checked. Owner and resulting
normal states must be subsonic. Incompatible data are diagnosed rather than clamped.

This policy requires a calorically perfect EOS exposing `calorically_perfect`,
`gamma` and `cp()`. These formulas belong to the boundary policy, not the EOS.
Other EOS models can supply another open-boundary policy. A policy implements
`supports_total_inlet`, host `validate(eos)` and device methods
`velocity_inlet(closure,owner,u,T,n)`, `pressure_outlet(closure,owner,p,n)` and
`total_inlet(closure,owner,p0,T0,direction,n)`, returning `Evaluation`.

### Incompressible flow

`IncompressibleClosure<Real,false>` supplies constant density with no temperature.
`IncompressibleClosure<Real,true>` additionally carries temperature. A velocity
inlet copies owner pressure; a pressure outlet copies owner velocity. Gauge
pressure can be negative or zero and is not divided by density.

Isothermal conditions are rejected for athermal flow. Stagnation and supersonic
conditions require compressible flow. Projection and pressure-correction equations
remain the solver's responsibility.

## Shared TOML file

The existing `[[boundary_condition]]` array, `patch_id`, `name` and top-level
physical `type` are retained. Physical parameters live in the entry's `physical`
table; each scalar manager reads only its own `scalar.<field>` table.

```toml
[[boundary_condition]]
patch_id = 0
name = "inlet"
family = "fluid"
source_type = "BCInflow"
type = "INLET_TOTAL_TP"

[boundary_condition.physical]
total_pressure = 110000.0
total_temperature = 300.0
direction = [1.0, 0.0, 0.0]

[boundary_condition.scalar.tracer]
type = "FIXED_VALUE"
value = 0.25

[[boundary_condition]]
patch_id = 1
name = "outlet"
type = "OUTLET_PRESSURE"
[boundary_condition.physical]
pressure = 100000.0
[boundary_condition.scalar.tracer]
type = "ZERO_GRADIENT"

[[boundary_condition]]
patch_id = 2
name = "wall"
type = "WALL_ISOTHERMAL"
[boundary_condition.physical]
temperature = 350.0
[boundary_condition.scalar.tracer]
type = "ZERO_GRADIENT"

[unlabelled]
type = "SYMMETRY"
[unlabelled.scalar.tracer]
type = "ZERO_GRADIENT"
```

`read_config(path)` and `parse_config(text)` return only physical configuration.
There is no general parser, named `flow` section or renamed patch-ID key.
Unrelated import metadata and scalar tables are ignored; keys inside `physical`
are strict. Scalar-only unlabelled data do not configure physical unlabelled faces.

| Top-level `type` | Required physical parameters | Optional physical parameters |
| --- | --- | --- |
| `INLET_VELOCITY` | `velocity` | `temperature` |
| `INLET_TOTAL_TP` | `total_pressure`, `total_temperature`, `direction` | None |
| `INLET_SUPERSONIC` | `pressure`, `temperature`, `velocity` | None |
| `OUTLET_PRESSURE` | `pressure` | None |
| `INLET_OUTLET` | `pressure`, `velocity` | `temperature` |
| `OUTLET_SUPERSONIC` | None | None |
| `WALL_ADIABATIC` | None | `velocity` |
| `WALL_ISOTHERMAL` | `temperature` | `velocity` |
| `WALL_SLIP` | None | `velocity`, `temperature` |
| `SYMMETRY`, `EXTRAPOLATE` | None | None |

Vectors contain three finite numbers; wall velocity defaults to zero. For
`INLET_OUTLET`, velocity/temperature describe the backflow branch. An entry's ID and
name are mandatory, entries may be reordered, and every mesh patch must appear
exactly once. Unknown physical types, including unconfigured `USER_DEFINED`
placeholders, are rejected by the built-in parser. Custom types use programmatic
assignment. A globally empty catalogue uses `boundary_condition = []`.

The parser reads locally on each caller, without MPI communication. All ranks
should receive compatible configuration. Parser and setup errors terminate through
`mpi::fatal`.

## Field filling

```cpp
namespace bc = krona::boundary::physical;
using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;
using EOS = krona::thermodynamics::IdealGas<Real>;

Exec exec;
EOS eos;
const auto config = bc::read_config("bc.toml");
bc::Manager<Exec, Real> boundary{mesh_part, mesh_storage.view(), config, exec};
bc::CompressibleClosure<EOS, bc::PerfectGasCharacteristics> closure{eos};
const auto primitive = bc::make_fields<Real>(
    rho.component(0), velocity.component(0), velocity.component(1),
    velocity.component(2), pressure.component(0), temperature.component(0));
boundary.apply_async(primitive, closure, time);
```

The names denote existing field views with extended boundary storage.
`make_fields` supports rank-one Kokkos Views and component subviews, including
`LayoutStride`. Its argument order is density, three velocity components, pressure
and optional temperature. It allocates no buffers.

For a configured athermal incompressible catalogue:

```cpp
bc::IncompressibleClosure<Real> closure{Real{1000}};
const auto primitive = bc::make_incompressible_fields<Real>(
    closure.density, velocity.component(0), velocity.component(1),
    velocity.component(2), pressure.component(0));
boundary.apply_async(primitive, closure);
```

No density or temperature array is required. A final temperature view is optional
for thermal incompressible flow. Compressible output cannot use the constant-density
adapter. Reader state, closure state and manager arithmetic types must agree.

## General face API

| API | Execution and contract |
| --- | --- |
| `for_each_patch(visitor)` | Host `visitor(PatchRange,condition)` for nonempty local intervals |
| `visit(patch_id,visitor)` | Host callback for a configured labelled patch, including an empty interval |
| `visit_unlabelled(visitor)` | Host callback for a configured unlabelled interval |
| `for_each_face_async(function,time,transform)` | Device `function(context,condition)`; no automatic field read or state evaluation |
| `evaluate_async(reader,closure,sink,time,transform)` | Device `sink(context,owner,Evaluation,condition)` |
| `apply_async(fields,closure,time,transform)` | Evaluates states and fills extended face slots |

`Context<Real>` exposes `face`, `owner`, `boundary_index`, `extended_index`,
`position`, `displacement`, `tangential_displacement`, `normal`, `distance`, `area`,
`time` and `temperature_tangential_increment`. `boundary_index` indexes the compact
physical-face block; `extended_index` indexes an extended field array.

A named functor can select work at compile time by condition type:

```cpp
struct WallKernel {
    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const bc::Context<Real>& c,
                                          const Condition& condition) const
    {
        if constexpr (std::is_same_v<Condition, bc::IsothermalWall<Real>>) {
            // Use condition.temperature, c.owner, c.normal, c.distance and c.area
            // to fill gradient data or assemble the equation's boundary contribution.
        }
    }
};
boundary.for_each_face_async(WallKernel{}, time);
```

The callback owns indexing, output precision and write-conflict handling. Several
boundary faces may share an owner; scatter into cell or matrix data needs atomics
or another conflict-safe scheme. Host visitors expose arbitrary launch policies
through `geometry_view()` and `execution_space()`.

`evaluate_async` reads `reader(owner_index) -> closure.state_type` and sends each
result to the caller's sink without a temporary state array, hidden field filling,
reduction or fence. It can record diagnostics or perform application-specific
primitive/conservative conversion. A custom field adapter for `apply_async` adds
`store(extended_index,state)` and optionally host `validate<Exec>(count)`.

## Extension and failure handling

A custom condition implements
`operator()(context,owner,closure) const -> Evaluation<Closure::state_type>`.
Register its type in `Manager<Exec,Real,CustomConditions...>` and assign it with
`set(patch_id,condition)`. `set_unlabelled(condition)` configures unlabelled faces.
Custom descriptors own parameter and backing-storage validation. Device operators
must be `const`; referenced buffers must outlive submitted work.

A custom closure provides `real_type`, `state_type`, `compressible` and
`has_temperature`, plus the built-in condition protocol for those capabilities.
Variant dispatch instantiates kernels for every built-in condition type, including
conditions not assigned at run time; compressible closures therefore also need a
`total_inlet` method, which may return `UnsupportedCondition`.
`validate()` and `validate_condition(condition)` are optional host hooks.
Built-in conditions use `valid`, `make_state`, `wall_state`, `velocity_inlet`,
`pressure_outlet`, and for compressible cases `sound_speed` and `total_inlet`.
The standard state adapters target the supplied primitive `State` types; other
states require their own reader/writer and compatible condition operations.

No per-face variant lookup, virtual call, string lookup or allocation occurs.
Condition variants are dispatched on the host; device kernels receive concrete
condition and closure types.

Built-ins return `Success`, `InvalidOwner`, `InvalidParameters`,
`InvalidThermodynamics`, `UnsupportedCondition`, `RegimeMismatch`, `Backflow` or
`NoSubsonicSolution`. Failed evaluation states must not be consumed.
`evaluate_async` gives the sink control over fallback and diagnostic handling.
`apply_async` calls `Kokkos::abort` on failed device evaluations; this is a local
process/device failure, not a wrapper-controlled collective MPI abort. For an
MPI-wide policy, reduce recorded statuses at a caller-selected synchronization
point and report through `mpi::fatal`.

`InletOutlet` supplies an explicit static inflow state when the outlet closure
reports backflow; its compressible inflow must be subsonic. Characteristic formulas
are local approximations, not a general multidimensional nonreflecting boundary.
Physical face values alone do not impose viscous traction, heat flux, projection
boundary equations or implicit matrix terms. Using a face state as an arbitrary
exterior-cell Riemann state does not by itself enforce impermeability.
