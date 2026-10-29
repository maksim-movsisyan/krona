# Scalar boundary conditions

`krona::boundary::scalar` assigns scalar conditions to external mesh patches. It can fill boundary-face values in an extended cell field, provide value/gradient relations to a solver, or dispatch your own face kernels. It does not solve an equation or exchange MPI ghosts.

## Architecture

| Part | Responsibility |
| --- | --- |
| `Configuration` (`scalar/config.hpp`) | Host data for one scalar: patch IDs, names, built-in conditions, and optional unlabelled-face condition. Parsed from TOML or constructed in C++. |
| `PatchCatalog` (`boundary/patches.hpp`) | Copies patch names and their local face intervals from a grouped host mesh. Records whether unlabelled external faces need a condition. |
| `GeometryView` / `FaceContext` | Borrowed mesh geometry and the geometry of one external face, respectively. |
| `FixedValue`, `NormalGradient`, `FixedFlux`, custom condition types | Describe the boundary condition independently of field storage and kernel scheduling. |
| `Relation<Real>` | Expresses face value and outward normal derivative as affine functions of the owner-cell value. |
| `Manager<Exec, Real, CustomConditions...>` | Stores one condition per patch, dispatches its concrete type on the host, and submits Kokkos work to `Exec`. Defaults: project execution space and `ComputeReal`. |

The manager owns the patch catalog and condition objects. It borrows the mesh geometry; it does not own the scalar field. Pass the target field explicitly on each application. `Configuration::field` identifies the TOML section; the manager does not use it to find or check a field by name.

## Mesh and field contracts

Use the host mesh and device view of the **same prepared mesh**. Faces must be grouped (`faces_grouped == true`), and geometry must already be computed. Select at least `MeshData::FaceCells | MeshData::Geometry` in `MeshStorage`.

External faces occupy `[n_inner_faces, n_faces)`. MPI interface faces precede this interval and are not processed by the boundary manager. Each external face has an owned owner cell, a unit normal pointing out of that cell, positive area, and positive projected owner-to-face distance. Construction validates these properties in the selected computation precision.

Boundary values are stored **at face centroids**, in slots appended after all local cells:

| Field interval | Contents |
| --- | --- |
| `[0, n_own)` | Owned-cell values. |
| `[n_own, n_cells)` | MPI ghost-cell values. |
| `[n_cells, n_cells + n_bfaces)` | One value per external face; `n_bfaces = n_faces - n_inner_faces`. |

For an external face `f`, its slot is `n_cells + (f - n_inner_faces)`. These appended slots are not additional mesh cells or values at reflected ghost-cell centres.

`required_field_size()` returns `n_cells + n_bfaces`. `apply_async()` accepts a writable, one-component `FieldView`. `apply_component_async()` accepts a writable rank-one Kokkos view, including a strided component of an AoS field. The selected component must:

- contain at least `required_field_size()` entries;
- use a floating-point value type and memory accessible to `Exec`;
- contain valid owner values wherever the condition needs them.

Field storage may use a different floating-point type from the manager's `Real`: values are converted for computation and on assignment. The field's location tag is not checked; `CellExtnd` expresses the intended layout. Both application methods write only appended boundary slots. Owned cells, MPI ghosts, other components, and any trailing extra entries are preserved.

For a selected component, use `manager.apply_component_async(vector.view().component(2), law)` instead of `apply_async()`.

## Fill boundary values

Sequence: prepare the mesh → create the manager → assign every condition → allocate an extended field → update owner values → apply the boundary conditions.

The following fragments assume MPI and Kokkos are initialized and `host_mesh` is a prepared `mesh::host::MeshPart`. Use these headers and aliases:

```cpp
#include <krona/boundary/scalar/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/transport/diffusion/isotropic.hpp>
#include <type_traits>

namespace boundary = krona::boundary;
namespace scalar = boundary::scalar;
namespace device = krona::mesh::device;
namespace fields = krona::fields;
using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;
using Space = typename Exec::memory_space;
```

This example assigns `300` to every labelled patch and zero normal gradient to any unlabelled faces:

```cpp
Exec exec{};
device::MeshStorage<Exec> mesh{
    host_mesh,
    {.data = device::MeshData::FaceCells | device::MeshData::Geometry},
    exec};
scalar::Manager<Exec, Real> manager{host_mesh, mesh.view(), exec};

for (const auto& patch : manager.catalog().patches())
    manager.set(patch.range.patch_id, scalar::FixedValue<Real>{300});
if (manager.catalog().needs_unlabelled())
    manager.set_unlabelled(scalar::NormalGradient<Real>{0});

fields::FieldStorage<Real, 1, Exec> temperature{
    "temperature", manager.required_field_size(),
    fields::FieldLocation::CellExtnd, exec};

krona::execution::complete(exec, [&] {
    temperature.fill(exec, Real{350}); // Example initial state.
    manager.apply_async(temperature.view());
}); // The fill and boundary kernels have completed.
```

Each labelled patch must have a condition, **including patches with no faces on this rank**. If unlabelled external faces exist locally or according to the global mesh counts, call `set_unlabelled()` or provide the corresponding configuration. There is no implicit boundary-condition default.

Reuse the manager and field. After changing owner values, call `apply_async()` again. Use `set(patch_id, condition)` to change a condition for subsequent submissions; no separate preparation or registration-closing step exists. Condition objects are copied into submitted kernels, so a later `set()` does not change an earlier submission.

Basic filling reads owner values only. If your gradient reconstruction or custom providers need current MPI ghosts, complete the required halo exchange before using those inputs. This manager performs no halo exchange itself.

## Conditions and their meaning

Let `phi_P` be the owner-cell value, `phi_f` the face value, and `g_n` the outward normal derivative. Geometry and an optional correction define:

- `h = (x_f - x_P) · n > 0`: projected normal distance, not the Euclidean distance;
- `t = x_f - x_P - h*n`: tangential displacement;
- `delta`: supplied explicit reconstruction increment; zero by default.

The reconstruction convention is `phi_f = phi_P + h*g_n + delta`.

| C++ condition | TOML type and parameter | Face value `phi_f` | Outward derivative `g_n` |
| --- | --- | --- | --- |
| `FixedValue<Real>{v}` | `FIXED_VALUE`, `value = v` | `v` | `(v - phi_P - delta) / h` |
| `NormalGradient<Real>{g}` | `NORMAL_GRADIENT`, `gradient = g` | `phi_P + h*g + delta` | `g` |
| `NormalGradient<Real>{0}` | `ZERO_GRADIENT`, no parameter | `phi_P + delta` | `0` |
| `FixedFlux<Real>{q}` | `FIXED_FLUX`, `flux = q` | Uses the derivative returned by the supplied flux law. | `law.normal_gradient(context, q)` |

`FixedFlux` specifies **outward physical flux density**, not flux integrated over the face. It carries no conductivity, diffusivity, or area factor. Positive flux means outward transport; conversion to a derivative belongs to the supplied law.

For isotropic diffusion, KRONA provides `q_n = -k*g_n`:

```cpp
const auto law = krona::transport::IsotropicDiffusion{
    krona::transport::UniformCoefficient<Real>{Real{2}}};
manager.apply_async(temperature.view(), law);
exec.fence("Boundary values ready");
```

Here a prescribed flux of `4` gives `g_n = -2`. A custom law supplies a device-callable `normal_gradient(const FaceContext<Real>&, Real flux)` returning a value convertible to `Real`. Built-in value and gradient conditions do not access this law. Any configured `FixedFlux` requires a law for `apply_*` and `evaluate_async()`, even when that patch is empty locally.

Law coefficients must be valid when kernels execute. In particular, the isotropic flux-to-gradient conversion requires finite, strictly positive `k`. The manager does not scan coefficient buffers during application.

### Optional gradient correction

Pass `boundary::GradientCorrection{gx, gy, gz}` as the correction argument to compute `delta = t · grad(phi_P)`. Each input is an accessible rank-one view with at least `n_own` entries, indexed by owner cell. Gradients must already be available when the boundary kernels execute. A custom correction is a device-callable object returning an increment from `FaceContext<Real>`.

Without a correction, `delta = 0`. With one, zero normal gradient generally gives `phi_f = phi_P + delta`, not simply `phi_P`. `apply_*` fills an exact built-in `FixedValue` directly without reading the owner or evaluating the correction; `evaluate_async()` evaluates the correction to form the derivative relation as well.

## Read conditions from TOML

Example for a mesh with patch `0` named `inlet`, patch `1` named `outlet`, and patch `2` named `wall`:

```toml
[[boundary_condition]]
patch_id = 0
name = "inlet"
[boundary_condition.scalar.temperature]
type = "FIXED_VALUE"
value = 300.0

[[boundary_condition]]
patch_id = 1
name = "outlet"
[boundary_condition.scalar.temperature]
type = "ZERO_GRADIENT"

[[boundary_condition]]
patch_id = 2
name = "wall"
[boundary_condition.scalar.temperature]
type = "FIXED_FLUX"
flux = 4.0

[unlabelled.scalar.temperature]
type = "NORMAL_GRADIENT"
gradient = 0.0
```

Replace the programmatic condition assignment with configuration-based construction:

```cpp
const auto config = scalar::read_config("boundary.toml", "temperature");
scalar::Manager<Exec, Real> configured{host_mesh, mesh.view(), config, exec};
configured.apply_async(temperature.view(), law);
exec.fence("Configured boundary values ready");
```

Alternatively, `parse_config(text, "temperature")` parses an existing string, and `manager.configure(config)` replaces the current assignments.

Configuration must describe every mesh patch exactly once, with its exact ID and name; entry order does not matter. Every entry must contain the selected scalar's table. Type strings are case-sensitive. The table accepts only `type` and the parameter belonging to that type. Numbers must be finite and, when bound, representable in the manager's `Real`. `ZERO_GRADIENT` accepts no parameter. Multiple scalar tables can share the same patch entries. With no labelled patches, use `boundary_condition = []` explicitly.

`read_config()` reads the file on each calling rank. It does not broadcast or synchronize. For root-only I/O, broadcast the text yourself, then parse it on each rank. Parsing accepts only the built-in condition types; assign custom types in C++. The mesh-generated BC template contains import/physical metadata: add the scalar tables shown above before using it here.

## Use relations in your own computation

`Relation<Real>` contains four coefficients:

```text
phi_f = value_owner    * phi_P + value_constant
g_n   = gradient_owner * phi_P + gradient_constant
```

Use `relation.face_value(phi_P)` and `relation.normal_gradient(phi_P)` to evaluate them. These helpers avoid using `phi_P` when the corresponding coefficient is zero. The derivative relation already accounts for the supplied correction; do not subtract it again. Relations do not multiply the derivative by a material coefficient or face area.

Choose the entry point matching what your computation needs:

| Method | Callback and behaviour |
| --- | --- |
| `evaluate_async(sink, law, correction, time)` | Submits face kernels calling `sink(context, relation)`. Suitable for derivative output or implicit assembly; it neither reads a scalar field nor fills one automatically. |
| `for_each_face_async(sink, time)` | Submits face kernels calling `sink(context, typed_condition)`. Performs no relation construction or flux conversion. Useful when consuming a prescribed flux directly. |
| `for_each_patch(visitor)` | Calls `visitor(range, typed_condition)` on the host for each nonempty local range. Launch your own kernels with the supplied concrete condition type. |

All three require complete condition assignments. Empty local ranges are skipped. An unlabelled range uses `patch_id == -1`. Typed callbacks must compile for every condition alternative in the manager, not only the types currently assigned.

### Example: launch your own fill-and-gradient kernel

This kernel fills face values and stores their outward derivatives in a separate boundary-sized array. It uses the same conditions as the manager; no type switches or TOML parsing occur on the device. Define it at namespace scope after the aliases above:

```cpp
template<class Condition, class Law>
struct FillAndGradient {
    boundary::GeometryView<Space> geometry;
    fields::FieldView<Real, 1, Space> field;
    Kokkos::View<Real*, Space> normal_gradient;
    Condition condition;
    Law law;
    Real time;

    KOKKOS_INLINE_FUNCTION void operator()(krona::LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const auto r = scalar::make_relation(condition, c, Real{0}, law);
        const Real owner = (r.value_owner != Real{0} || r.gradient_owner != Real{0})
            ? field(static_cast<std::size_t>(c.owner)) : Real{0};
        field(c.extended_index) = r.face_value(owner);
        normal_gradient(c.boundary_index) = r.normal_gradient(owner);
    }
};
```

Launch it using the existing `manager`, `temperature`, and `law`:

```cpp
const auto geometry = manager.geometry_view();
const auto field = temperature.view();
const auto boundary_count = manager.required_field_size()
    - static_cast<std::size_t>(host_mesh.n_cells);
Kokkos::View<Real*, Space> derivatives("boundary_normal_gradient", boundary_count);
const Real time = Real{1.5}; // Must be finite.
using Law = std::remove_cvref_t<decltype(law)>;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<krona::LocalIndex>>;

manager.for_each_patch([&](boundary::PatchRange range, const auto& condition) {
    using Condition = std::remove_cvref_t<decltype(condition)>;
    Kokkos::parallel_for("custom_boundary_fill",
        Policy(manager.execution_space(), range.begin, range.end),
        FillAndGradient<Condition, Law>{
            geometry, field, derivatives, condition, law, time});
});
manager.execution_space().fence("Custom boundary results ready");
```

`range.begin` and `range.end` are absolute local face indices, with an exclusive end. Each face has a unique output slot. If instead you accumulate contributions into owner cells, several faces can share an owner: use atomics or another safe assembly strategy.

The example explicitly chooses zero correction. To include reconstruction, pass a correction object into the kernel and use `correction(c)` instead of `Real{0}` in `make_relation()`. This manual launch bypasses the manager's `apply_*` input checks: validate field/output extents, time, and provider inputs yourself. Unlike direct `FixedValue` filling, this example also evaluates normal derivatives, so its owner values must be valid even on `FixedValue` patches.

### Add a custom condition

A custom scalar condition implements a device-callable `operator()(context, increment)` returning `Relation<Real>`. It can use face position and time. For example:

```cpp
struct TimeDependentValue {
    Real base = 300, x_slope = 2, time_slope = 5;

    KOKKOS_INLINE_FUNCTION scalar::Relation<Real> operator()(
        const boundary::FaceContext<Real>& c, Real increment) const
    {
        const Real value = base + x_slope*c.x + time_slope*c.time;
        return scalar::FixedValue<Real>{value}(c, increment);
    }
};
```

Add its type to the manager's template arguments; built-in conditions remain available. This example requires labelled patch `0`:

```cpp
scalar::Manager<Exec, Real, TimeDependentValue> custom{host_mesh, mesh.view(), exec};
for (const auto& patch : custom.catalog().patches())
    custom.set(patch.range.patch_id, scalar::NormalGradient<Real>{0});
if (custom.catalog().needs_unlabelled())
    custom.set_unlabelled(scalar::NormalGradient<Real>{0});
custom.set(0, TimeDependentValue{Real{300}, Real{2}, Real{5}});
custom.apply_async(temperature.view(), scalar::NoFluxLaw{},
                   boundary::NoCorrection{}, Real{1.5});
exec.fence("Time-dependent boundary values ready");
```

Here the second argument to the condition's `operator()` is the reconstruction increment; time is `c.time`. Custom parameters, captured buffers, and returned coefficients are caller contracts: the manager does not validate them. Keep condition/provider objects device-callable, with accessible, valid backing storage, and produce finite results.

## Face context reference

| Member | Meaning |
| --- | --- |
| `face`, `owner` | Absolute local face index and owned-cell index. |
| `boundary_index` | `face - n_inner_faces`; index into boundary-only arrays. |
| `extended_index` | `n_cells + boundary_index`; index into an extended cell field. |
| `x, y, z` | Face centroid coordinates. |
| `nx, ny, nz` | Outward unit normal. |
| `distance` | Positive projected distance `h`. |
| `tx, ty, tz` | Tangential displacement `t`. |
| `area` | Face area; apply it explicitly when integrating flux density. |
| `time` | Time supplied to the application/iteration method; defaults to zero. |

## Asynchrony and lifetime

| Operation | Completion behaviour |
| --- | --- |
| `Manager(...)` | Blocking setup: copies host metadata, validates external geometry, and fences its execution-space instance before returning. |
| `read_config()`, `parse_config()`, `configure()`, `set()`, `set_unlabelled()` | Execute on the host. They do not submit boundary kernels or wait for previously submitted kernels. File reading/parsing completes before return. |
| `apply_async()`, `apply_component_async()` | Validate host-visible inputs and submit boundary-fill kernels. No final fence. |
| `evaluate_async()`, `for_each_face_async()` | Submit callback kernels. No final fence. |
| `for_each_patch()` | Completes host callback dispatch before return; does not wait for kernels launched by the visitor. |
| `execution_space().fence()` | Waits for work submitted to that execution-space instance. |

These operations perform no MPI communication or barrier. A local fence waits for local Kokkos work, not for other ranks. Kernels may finish immediately on a synchronous backend; methods without a final fence provide no portable completion guarantee.

Queue dependent producers, boundary kernels, and consumers on the **same execution-space instance** for ordering. Synchronize explicitly when passing data between different instances, to host code, or to MPI. Finish boundary work before any host/MPI access that conflicts with its reads or writes.

Keep mesh geometry, fields, correction/coefficient buffers, and custom callback storage valid and unchanged wherever the queued kernels read them. The manager has no destructor fence: complete outstanding work before releasing its backing data or finalizing Kokkos. A new condition assignment copies new parameters for later kernels; it does not synchronize changes to buffers referenced by those parameters.
