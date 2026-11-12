# Scalar boundary conditions

`krona::boundary::scalar` assigns conditions to exterior faces, fills face-value slots in extended fields, and supplies affine value/derivative relations to numerical operators. It performs no halo exchange or equation solve.

## Objects and ownership

| Object | Responsibility |
| --- | --- |
| `BoundaryLayout<Exec, Real>` (`boundary/layout.hpp`) | Owns one host patch catalog, borrows compact mesh geometry, stores the execution instance and field sizes. Share it between scalar managers on the same mesh. |
| `Manager<Exec, Real, ExtraConditions...>` | Borrows the layout; owns one condition per labelled patch and an optional unlabelled-face condition. Never binds or owns a field. |
| `FixedValue`, `NormalGradient`, `FixedFlux` | Describe the mathematical condition. Their parameter can be a number or a provider evaluated at the current face/time. |
| `Relation<Real>` | Affine face value and outward normal derivative, without a transport coefficient or face area. |
| `Status<Exec>` | Owns small device/host diagnostic buffers, reusable across managers and submissions on the same execution instance. |

Keep the layout and mesh allocations alive while managers use them. Keep fields, provider buffers, coefficient/gradient buffers and callback outputs alive until their kernels finish. Providers returned by `profile::patch_values()` borrow storage and retain no allocation ownership.

## Prepare, assign, finalize, apply

Assume active MPI/Kokkos and a prepared `host_mesh`. The host mesh and device view must describe the same local numbering and geometry. Faces must be grouped; select at least `FaceCells | Geometry`.

```cpp
#include <krona/boundary/scalar/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>

namespace bc = krona::boundary;
namespace scalar = bc::scalar;
namespace grad = krona::numerics::gradient;
using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;
using Space = typename Exec::memory_space;

Exec exec{};
krona::mesh::device::MeshStorage<Exec> mesh(host_mesh,
    {.data = krona::mesh::device::MeshData::FaceCells |
             krona::mesh::device::MeshData::Geometry}, exec);
bc::BoundaryLayout<Exec, Real> layout(host_mesh, mesh.view(), exec);
scalar::Manager<Exec, Real> boundary(layout);

for (const auto& patch : layout.catalog().patches())
    boundary.set(patch.range.patch_id, scalar::FixedValue<Real>{300});
if (layout.catalog().needs_unlabelled())
    boundary.set_unlabelled(scalar::NormalGradient<Real>{0});
boundary.finalize();

krona::fields::FieldStorage<Real, 1, Exec> temperature(
    "temperature", layout.required_field_size(),
    krona::fields::FieldLocation::CellExtnd, exec);
scalar::Status<Exec> status(exec);

temperature.fill_async(exec, Real{350});
status.begin_async();
boundary.apply_async(temperature.view(), status);
status.check(); // Local blocking completion and fatal-error check.
```

Assign every labelled patch, including patches empty on this rank. Unlabelled faces require an explicit condition if they exist locally or according to global mesh counts. There is no implicit condition.

`finalize()` checks completeness and provider metadata and caches whether a flux law is needed. It reads no device values, submits no kernels, and performs no MPI communication. `set()`, `set_unlabelled()` and `configure()` invalidate it. Finalize again after changing a condition or provider binding. Changing time or updating an already-bound buffer's contents does not invalidate it.

Conditions are copied into submitted kernels. Later assignments affect later submissions; they do not change already-queued condition objects. Updating borrowed buffers still requires correct execution ordering.

## Field and geometry contracts

Exterior faces occupy `[n_inner_faces, n_faces)`; MPI interface faces are outside this range. Every exterior face must have an owned owner cell, an outward unit normal, positive area and positive projected owner-to-face distance. Layout construction validates these in `Real` and completes before returning.

| Extended field interval | Contents |
| --- | --- |
| `[0, n_own)` | Owned cells. |
| `[n_own, n_cells)` | MPI ghost cells. |
| `[n_cells, n_cells + n_faces - n_inner_faces)` | Face-centred boundary values. |

For exterior face `f`, the output index is `n_cells + (f - n_inner_faces)`. These are face values, not reflected ghost-cell values.

`apply_async(field, status, ...)` accepts a writable scalar `FieldView`. `apply_component_async(view, status, ...)` accepts a writable rank-one Kokkos view, including a strided component. It requires floating-point storage, accessible memory and at least `required_field_size()` entries. Only boundary slots are written. Owned/ghost cells, other components and trailing entries are preserved.

Storage precision may differ from `Real`. Computation uses `Real`; assignment converts to the storage type and checks for a finite result. Rounding and underflow are not diagnosed. Owner values must be finite where read. Fixed-value filling reads neither owners nor the correction.

## Conditions and affine relations

Let `h = (x_f - x_P)·n > 0`, `t = x_f - x_P - h*n`, and `delta` be the explicit correction. The convention is:

`phi_f = phi_P + h*g_n + delta`, where `g_n = d(phi)/dn` uses the outward normal.

| Condition | Parameter at the face | Face value | Normal derivative |
| --- | --- | --- | --- |
| `FixedValue<Real>{v}` | Prescribed `v` | `v` | `(v - phi_P - delta)/h` |
| `NormalGradient<Real>{g}` | Prescribed `g` | `phi_P + h*g + delta` | `g` |
| `FixedFlux<Real>{q}` | Outward physical flux density `q` | Uses the derivative returned by the flux law | `law.normal_gradient(context, q)` |

Zero gradient is `NormalGradient<Real>{0}`; with a correction its face value can differ from the owner value.

`Relation<Real>` stores:

```cpp
phi_f = relation.value_owner * phi_P + relation.value_constant;
g_n   = relation.gradient_owner * phi_P + relation.gradient_constant;
```

Use `face_value(phi_P)` and `normal_gradient(phi_P)` to avoid reading a mathematically unused owner value when the corresponding slope is zero.

A flux condition requires a device-callable `law.normal_gradient(context, flux)` for `apply_*` and `evaluate_async()`, even if its patch is locally empty. Value/gradient conditions do not use this law. For isotropic conduction, `q = -k*g_n`: positive heat flux leaves the domain. Face area is not included.

```cpp
#include <krona/transport/diffusion/isotropic.hpp>
const auto law = krona::transport::IsotropicDiffusion{
    krona::transport::UniformCoefficient<Real>{Real{2}}};
// Assign scalar::FixedFlux<Real>{4} to obtain g_n = -2.
status.begin_async();
boundary.apply_async(temperature.view(), status, law);
status.check();
```

Coefficients must satisfy their law's contract; isotropic conversion requires finite, positive conductivity. Validity checks for other law-specific inputs remain the law's responsibility.

## Profiles: same condition, another parameter source

Providers live in `boundary/profiles.hpp`, independently of scalar conditions. They describe a parameter, not an inlet or outlet. Use the same mechanism for a wall temperature, a prescribed derivative or a heat-flux distribution.

| Provider | Sampling |
| --- | --- |
| Plain number | Constant on the patch. Existing `FixedValue<Real>{300}` syntax remains valid. |
| `profile::Constant<T>{value}` | Returns `value`; supports scalars and `math::Vector3`. |
| Device functor or Kokkos lambda | `provider(context)`; can use face coordinates, time or borrowed inputs. |
| `profile::patch_values(range, view)` | Reads a borrowed value at `view(context.face - range.begin)`, preserving its stride. |

Scalar condition factories are `fixed_value<Real>(provider)`, `normal_gradient<Real>(provider)` and `fixed_flux<Real>(provider)`. List the resulting additional condition types in the manager template; dispatch remains once per patch, with no device-side virtual calls.

For example, define an analytic temperature provider at namespace scope:

```cpp
struct TemperatureProfile {
    KOKKOS_INLINE_FUNCTION Real operator()(const bc::FaceContext<Real>& c) const
    { return Real{300} + Real{2}*c.y + Real{5}*c.time; }
};
```

Create a manager supporting it; this fragment requires labelled patch `0`:

```cpp
const auto prescribed = scalar::fixed_value<Real>(TemperatureProfile{});
scalar::Manager<Exec, Real, decltype(prescribed)> profiled(layout);
for (const auto& patch : layout.catalog().patches())
    profiled.set(patch.range.patch_id, scalar::NormalGradient<Real>{0});
if (layout.catalog().needs_unlabelled())
    profiled.set_unlabelled(scalar::NormalGradient<Real>{0});
profiled.set(0, prescribed);
profiled.finalize();

status.begin_async();
profiled.apply_async(temperature.view(), status, scalar::NoFluxLaw{},
                     grad::NoCorrection{}, Real{1.5});
status.check();
```

To use a compact patch array instead, replace `prescribed` above with:

```cpp
const auto range = layout.catalog().patches()[0].range;
Kokkos::View<Real*, Space> patch_data("wall_temperature", range.end - range.begin);
Kokkos::deep_copy(exec, patch_data, Real{300});
const auto prescribed = scalar::fixed_value<Real>(bc::profile::patch_values(range, patch_data));
```

The array must have **exactly the local patch face count**, including zero for an empty patch. It is neither indexed by global face ID nor by the boundary-slot index. Its bound range/patch ID must match the assigned patch. Use `layout.catalog().unlabelled()` for an unlabelled profile. Const and strided views are supported; no copy, allocation or fence occurs in `patch_values()`.

Update `patch_data` on `exec` before the next apply; no rebinding or refinalization is needed. Keep it alive until the last consumer completes. Do not alias profile inputs with outputs modified by the same operation.

A custom provider must be device-copyable/callable, return a value convertible to `Real`, and capture accessible storage. An optional host `validate<Exec>(PatchRange)` hook checks its metadata; it must not submit work or synchronize. Providers should return the same parameter for the same context and current input data. Each built-in condition samples its provider once per face per evaluation.

The common provider layer also supports vector values for physical-condition integration.

## Gradient correction

Use `numerics/gradient/correction.hpp`.

```cpp
const auto correction = grad::GradientCorrection{grad::cell_gradients(gx, gy, gz)};
status.begin_async();
boundary.apply_async(temperature.view(), status, law, correction);
status.check();
```

`gx`, `gy`, `gz` are rank-one views, including strided views, with at least `n_own` entries. Alternatively, `cell_gradients(gradient_field.view(), component)` selects the gradient module's `3*component + axis` ordering. Projection is `delta = t·grad(phi_P)`; no coefficient or face area is applied. `grad::NoCorrection{}` returns zero and is the default.

Complete gradient producers before reading them. Boundary filling uses owned gradients only; any halo requirements of their computation belong to the application.

## Face context

| Members | Meaning |
| --- | --- |
| `face`, `owner` | Absolute local face index and owned-cell index. |
| `boundary_index`, `extended_index` | `face - n_inner_faces`, and `n_cells + boundary_index`. |
| `x, y, z`; `nx, ny, nz` | Face centroid and outward unit normal. |
| `distance`; `tx, ty, tz` | Positive normal distance `h` and tangential displacement `t`. |
| `area`, `time` | Face area and the supplied evaluation time; time defaults to zero. |

## Relations, callbacks and custom conditions

| Method | Callback and purpose |
| --- | --- |
| `evaluate_async(sink, status, law, correction, time)` | Device `sink(context, Relation<Real>)`; finite relations are checked before delivery. Use for implicit assembly or boundary derivatives. |
| `for_each_face_async(sink, time)` | Device `sink(context, typed_condition)`; raw dispatch without relation/law/correction/status handling. |
| `for_each_patch(visitor)` | Host `visitor(PatchRange, typed_condition)` for each nonempty local patch, including unlabelled faces. Launch your own kernels from it. |

Your callback must compile for every condition type registered in the manager, even if the current configuration uses only some of them.
Call `finalize()` before using `for_each_patch()` or `for_each_face_async()`. Both methods pass the condition to your callback. For `FixedFlux`, the callback can use the prescribed flux directly. It needs a flux law only if it converts that flux to a normal gradient.

For example, define this sink at namespace scope:

```cpp
struct SaveGradient {
    Kokkos::View<Real*, Space> owner_values, normal_gradient;
    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<Real>& c,
                                          const scalar::Relation<Real>& r) const
    {
        const Real owner = r.gradient_owner == Real{0} ? Real{0} : owner_values(c.owner);
        normal_gradient(c.boundary_index) = r.normal_gradient(owner);
    }
};
```

Then launch:

```cpp
Kokkos::View<Real*, Space> derivatives("boundary_derivatives",
    host_mesh.n_faces - host_mesh.n_inner_faces);
status.begin_async();
boundary.evaluate_async(SaveGradient{temperature.view().component(0), derivatives}, status, law);
status.check();
```

The manager checks the relation, not a custom sink's reads, arithmetic or writes. Validate the sink's buffers and results yourself. Multiple faces can share one owner: accumulation into owner-cell arrays needs atomics or another safe strategy.

For a new mathematical condition, implement device `operator()(const FaceContext<Real>&, Real increment)` returning `Relation<Real>`, and list that type in the manager template. An optional `validate<Exec>(PatchRange)` hook checks host metadata. For a law-aware custom closure, define `make_relation(condition, context, increment, law)` in the condition's namespace (found by argument-dependent lookup), and specialize `scalar::requires_flux_law<YourCondition>`. Its evaluation must also compile with `NoFluxLaw` for unassigned alternatives. Direct/custom launches must enforce their own input and numerical checks.

## Configuration

`read_config(path, "temperature", comm)` reads TOML on communicator rank zero, broadcasts the text and parses the selected scalar on every rank. `parse_config(text, "temperature")` parses locally. To load several scalars from one file, call `toml_config::read_broadcast(path, comm)` once and pass the text to `parse_config` for each field.

```toml
[[boundary_condition]]
patch_id = 0
name = "wall"
[boundary_condition.scalar.temperature]
type = "FIXED_VALUE"
value = 300.0

[unlabelled.scalar.temperature]
type = "ZERO_GRADIENT"
```

Supported types are `FIXED_VALUE` (`value`), `NORMAL_GRADIENT` (`gradient`), `ZERO_GRADIENT` (only `type`), and `FIXED_FLUX` (`flux`). Every labelled mesh patch must appear exactly once with matching ID and exact name. With no labels, write `boundary_condition = []` explicitly. Unknown scalar keys and invalid parameters are rejected. `Configuration::field` selects the parsed scalar section.

For numeric conditions, construct `Manager(layout, config)` or call `configure(config)`, then `finalize()`.

For a tabulated condition, put the CSV path in the corresponding parameter. This patch entry prescribes an inlet temperature that varies with `y`.

```toml
[[boundary_condition]]
patch_id = 0
name = "inlet"

[boundary_condition.scalar.temperature]
type = "FIXED_VALUE"
value = "profiles/inlet.csv"
```

The file `profiles/inlet.csv` contains the tabulated temperatures.

```csv
y,value
0.0,300.0
1.0,340.0
```

**KRONA interpolates the profile at each local face centroid.** For example, a face whose centroid has `y = 0.25` gets a temperature of `310.0` by linear interpolation. Only the axes listed in the CSV are used.

**Choose the CSV coordinates independently of the boundary mesh.** The sample points can differ from the face centroids. A centroid outside the table bounds on any active axis causes a configuration error.

Relative CSV paths use rank zero's working directory. The header contains `value` and one to three of `x`, `y`, and `z`. Each active axis needs at least two distinct coordinates, and the rows must cover every combination of active coordinates exactly once.

Include `<krona/boundary/scalar/tabulated_config.hpp>` and configure the manager on every rank, including ranks with empty local patches:

```cpp
const auto config = scalar::read_config("bc.toml", "temperature", comm);
scalar::TabulatedManager<Exec, Real> boundary(layout);
scalar::configure_tabulated(boundary, config, host_mesh, comm);
boundary.finalize();
```

Pass the host mesh used to build the layout and identical configurations on all ranks. The manager's conditions own the sampled buffers. Complete prior boundary kernels before rebinding conditions. For custom C++ providers, use `set()` or `set_unlabelled()`.

## Asynchrony and errors

| Operation | Completion |
| --- | --- |
| `BoundaryLayout(...)` | Local blocking geometry validation and fence; copies host catalog metadata. |
| `Manager(...)`, `set*()`, `configure()`, `finalize()` | Host work only. No device submission, fence or MPI communication. |
| `parse_config()` | Local, blocking TOML parsing. |
| `read_config()` | Blocking collective TOML load, followed by local parsing. |
| `configure_tabulated()` | Collective CSV loads, local interpolation and completed host-to-device copies. |
| `Status(...)` | Allocates small diagnostic buffers without initializing their device values. |
| `status.begin_async()` | Queues a reset on its execution instance; requires the previous batch to have been checked. |
| `apply_async()`, `apply_component_async()`, `evaluate_async()` | Host checks and kernel submission; no final fence. Require a begun status on the layout's execution instance. |
| `for_each_face_async()` | Submits callback kernels without a final fence. |
| `for_each_patch()` | Finishes host dispatch; does not wait for visitor-launched kernels. |
| `status.check()` | Copies diagnostics and fences the instance. Calls `mpi::require` on local failure; no MPI collective. |

Host contract violations terminate through `mpi::require`; unexpected host exceptions are handled by `mpi::guarded`. Standard fill/evaluation kernels record nonfinite results in the status and skip the affected write/sink call. No rollback is performed; a failed batch must not be consumed.

Use one status for several submissions/managers on the same execution instance, then check once. Queue dependent producers/consumers on that instance or synchronize explicitly across instances. A fence alone completes kernels but does not inspect recorded errors. Synchronous backends may execute immediately; `_async` methods provide no portable completion guarantee.

Layout/manager destruction does not fence. Status destruction drains a still-pending batch to protect its buffers, but does not replace `check()` or protect buffers already destroyed by the caller. Check before releasing any referenced storage or finalizing Kokkos. Custom `for_each_*` kernels own their numerical error handling.