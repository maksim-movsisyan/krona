# Scalar boundary conditions

The scalar boundary module describes face values and outward normal derivatives. It does not own solution fields, select material properties, or assemble a particular equation. Physical patch types and scalar conditions are independent: a physical wall does not implicitly prescribe a condition for every scalar.

`krona::boundary` provides the shared patch catalogue, face geometry and reconstruction corrections. `krona::boundary::scalar` provides conditions, configuration and `Manager`. Constitutive adapters, such as `krona::transport::IsotropicDiffusion`, live outside the boundary module.

## Objects and ownership

| Object | Role |
| --- | --- |
| `PatchCatalog` | Owns host patch names and contiguous face ranges, including the unlabelled range. |
| `GeometryView<Space>` | Borrows the compute mesh arrays needed to construct a face context. |
| `FaceContext<Real>` | Geometry, indices and time for one exterior face; passed by value into device operations. |
| `Relation<Real>` | Affine face value and normal derivative; contains no material coefficient or area. |
| `scalar::Manager<Exec, Real, CustomConditions...>` | Owns condition parameters and the catalogue; stores an execution-space instance and borrowed geometry. |

The manager is neither copyable nor movable. It has no permanent field buffers or host field mirrors. Patch selection and `std::variant` dispatch happen on the host, once per patch; kernels receive a concrete condition type. Each nonempty local patch produces one kernel submission.

## Mesh and field contracts

The mesh must have grouped faces. Exterior faces occupy `[n_inner_faces, n_faces)`, and every labelled patch, as well as unlabelled faces, occupies at most one contiguous interval. `MeshView` must include `MeshData::FaceCells | MeshData::Geometry` and agree with `MeshPart` counts. Exterior owners are owned cells.

Patch IDs index `MeshPart::patches`. Every labelled patch needs an explicit condition, including patches with no faces on this rank. Unlabelled faces need an explicit condition when present locally or indicated by global boundary counts. There is no implicit fallback condition.

For a face `f`, the context exposes:

| Member | Meaning |
| --- | --- |
| `face` | Absolute mesh face index `f`. |
| `owner` | Owned-cell index. |
| `boundary_index` | Compact exterior-face index `f - n_inner_faces`. |
| `extended_index` | Field slot `n_cells + boundary_index`. |
| `x, y, z` | Face-centre coordinates. |
| `nx, ny, nz` | Outward unit normal. |
| `distance` | Positive normal distance from owner centre to face centre. |
| `tx, ty, tz` | Tangential part of the owner-to-face displacement. |
| `area`, `time` | Face area and the time supplied to the operation. |

Boundary filling requires at least `manager.required_field_size()` entries: owned cells, MPI ghost cells, then one exterior-face slot per boundary face. These last slots represent **face values**, not values at mirrored ghost-cell centres. Filling modifies only these slots; owned cells and MPI ghosts are preserved. Strided component views are supported.

## Relations and reconstruction

Let `P` be the owner, `f` the face centre, `n` its outward unit normal, and

\[
h=(x_f-x_P)\cdot n>0,\qquad t=x_f-x_P-hn.
\]

The optional explicit increment is `tau = t dot grad(phi)_P`. `NoCorrection` supplies zero; `GradientCorrection{gx, gy, gz}` reads the three owner-cell gradient components. It does not compute gradients or exchange them.

`Relation<Real>` represents

\[
\phi_f=a\phi_P+b,\qquad g_n=c\phi_P+d.
\]

The coefficients are `value_owner`, `value_constant`, `gradient_owner`, and `gradient_constant`. Evaluate them with `face_value(phi_P)` and `normal_gradient(phi_P)`.

| Condition | Face value | Outward normal derivative |
| --- | --- | --- |
| `FixedValue<Real>{V}` | `V` | `(V - phi_P - tau) / h` |
| `NormalGradient<Real>{G}` | `phi_P + h*G + tau` | `G` |
| `FixedFlux<Real>{Q}` | Uses `NormalGradient{law.normal_gradient(context, Q)}` | Supplied by the explicit flux law |

Thus zero normal gradient can still give a face value different from the owner value when `tau != 0`. Relation methods treat an exactly zero owner coefficient as owner-independent, avoiding `0 * NaN`; this is a structural coefficient check, not a tolerance comparison.

## Setup and boundary filling

The following uses an already loaded `host_mesh`. MPI and Kokkos must be initialized. The TOML example below supplies conditions for its two example patches.

```cpp
#include <krona/boundary/scalar/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/transport/isotropic_diffusion.hpp>

#include <cstddef>
#include <type_traits>

namespace bc = krona::boundary;
namespace scalar = bc::scalar;
namespace transport = krona::transport;
using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;

Exec exec;
krona::mesh::compute::MeshOptions options;
options.data = krona::mesh::compute::MeshData::FaceCells |
               krona::mesh::compute::MeshData::Geometry;
krona::mesh::compute::MeshStorage<Exec> mesh(host_mesh, options, exec);

const auto config = scalar::read_config("boundary.toml", "temperature");
scalar::Manager<Exec, Real> temperature_bc(host_mesh, mesh.view(), config, exec);

krona::fields::FieldStorage<Real, 1, Exec> temperature(
    "temperature", temperature_bc.required_field_size(),
    krona::fields::FieldLocation::CellExtnd, exec);
temperature.fill(exec, Real{300});

const auto law = transport::IsotropicDiffusion{
    transport::UniformCoefficient<Real>{Real{2}}};
temperature_bc.apply_async(temperature.view(), law);
// Subsequent work submitted to the same exec may consume the face values.
```

Alternatively, construct the manager without `config` and call `set(patch_id, condition)` for **every** label. Use `set_unlabelled(condition)` when needed. For example, `set(0, scalar::FixedValue<Real>{300})` binds label zero. `NormalGradient<Real>{0}` is the C++ zero-gradient condition.

`apply_async` accepts a scalar `FieldView` (`components == 1`). To fill one component of a vector or tensor, pass its rank-one Kokkos view:

```cpp
temperature_bc.apply_component_async(vector_field.component(0), law);
```

Here `vector_field` is an existing `FieldView` with the full extended extent. This scalar operation does not implement a coupled vector boundary condition.

## Flux laws and coefficients

`FixedFlux` stores an **outward flux density**, without multiplication by area. The scalar module imposes no constitutive sign or coefficient. A flux law must provide a device-callable, const method:

```cpp
law.normal_gradient(const bc::FaceContext<Real>& context, Real outward_flux)
```

The result must be convertible to `Real`. The supplied isotropic adapter uses `Q = -D * g_n`, so positive outward flux gives a negative normal derivative for positive `D`. For temperature and heat flux, `D` is thermal conductivity, not thermal diffusivity.

No face-sized coefficient array is required. A provider may evaluate a model or read an existing field:

```cpp
const auto law = transport::IsotropicDiffusion{
    transport::OwnerCellCoefficient{conductivity.component(0)}};
temperature_bc.apply_async(temperature.view(), law);
```

`conductivity` is an existing scalar field with valid owned-cell values. `OwnerCellCoefficient` explicitly chooses owner-cell extrapolation; other face-property models belong in other providers. Providers may also capture additional state through views.

Only `FixedFlux` conversion accesses the law; fixed values and normal gradients do not. `apply_*` and `evaluate_async` require a compatible law if **any configured** built-in `FixedFlux` exists, even on a locally empty patch. There is no implicit unit coefficient. Coefficient extents, accessibility and numerical validity remain caller contracts; conversion by `IsotropicDiffusion` requires finite `D > 0`. Its device assertions are not a release-mode validation pass.

## TOML configuration

Scalar conditions are added to the physical patch entries generated by the partitioner:

```toml
[[boundary_condition]]
patch_id = 0
name = "heated_wall"
type = "WALL_ADIABATIC"

[boundary_condition.scalar.temperature]
type = "FIXED_FLUX"
flux = 1000.0

[[boundary_condition]]
patch_id = 1
name = "outlet"

[boundary_condition.scalar.temperature]
type = "ZERO_GRADIENT"

[unlabelled.scalar.temperature]
type = "FIXED_VALUE"
value = 300.0
```

Supported scalar types are `FIXED_VALUE` / `value`, `NORMAL_GRADIENT` / `gradient`, `FIXED_FLUX` / `flux`, and `ZERO_GRADIENT` with no parameter. Condition keys are strict; unrelated physical metadata and other scalar tables are ignored. In particular, the physical `type` above does not determine the temperature condition.

Every labelled entry must contain the requested scalar table. Binding checks exact patch IDs and names against the mesh. With no labelled patches, use `boundary_condition = []` at the document root. `parse_config(text, field)` parses text; `read_config(path, field)` reads locally on each caller. Neither performs MPI communication or broadcasts configuration.

## Choosing an operation

| API | Callback or effect | When to use it |
| --- | --- | --- |
| `apply_async(field, law, correction, time)` | Fills scalar boundary slots. | Standard face-value reconstruction. |
| `apply_component_async(view, law, correction, time)` | Fills a rank-one Kokkos view's boundary slots. | A selected field component or direct Kokkos view. |
| `evaluate_async(sink, law, correction, time)` | Device `sink(context, Relation<Real>)`. | Consume uniform affine relations, e.g. gradients or equation coefficients. |
| `for_each_face_async(sink, time)` | Device `sink(context, concrete_condition)`. | Interpret condition types directly, without mandatory relation construction. |
| `for_each_patch(visitor)` | Host `visitor(range, concrete_condition)`. | Submit a custom patch operation with a different traversal or execution policy. |

The optional arguments default to `NoFluxLaw`, `NoCorrection`, and time zero where applicable. `evaluate_async` computes the relation with the supplied law and correction before calling the sink; it does not read a solution field or store results itself. The sink decides which relation coefficients and fields to use.

`for_each_face_async` supplies the original typed condition. It does not convert fluxes, apply a correction, or require a flux law. Its sink must compile for every registered variant alternative, including types not currently assigned to a patch. Use overloads or `if constexpr` to distinguish them. This is compile-time branching inside typed kernels, not a device-side variant lookup.

### Raw visitation: collect prescribed integrated flux

This operation only collects explicitly prescribed fluxes; it is not a complete equation boundary assembly. Declare the functor at namespace scope:

```cpp
template<class Out>
struct CollectPrescribedFlux {
    Out out;

    template<class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(
        const bc::FaceContext<Real>& c, const Condition& condition) const
    {
        Real integrated = 0;
        if constexpr (std::is_same_v<Condition, scalar::FixedFlux<Real>>)
            integrated = condition.flux * c.area;
        out(c.boundary_index) =
            static_cast<typename Out::non_const_value_type>(integrated);
    }
};
```

```cpp
Kokkos::View<Real*, typename Exec::memory_space> prescribed_flux(
    "prescribed_flux", host_mesh.n_faces - host_mesh.n_inner_faces);
temperature_bc.for_each_face_async(CollectPrescribedFlux{prescribed_flux});
```

No flux law is needed because this operation never requests a gradient. The output contains `Q * area` for `FixedFlux` and zero for other condition types. It is indexed by `boundary_index`, not by absolute face index.

### Relation evaluation: reconstruct boundary gradients

A scalar condition generally does not determine the entire gradient vector. One explicit reconstruction is to preserve the owner gradient's tangential part and replace its normal component:

\[
\nabla\phi_f=\nabla\phi_P+
\left(g_n^{BC}-n\cdot\nabla\phi_P\right)n.
\]

This choice is useful for demonstrating `evaluate_async`; a solver may require a different tangential reconstruction, particularly for prescribed face values. Declare this functor at namespace scope:

```cpp
template<class Q, class X, class Y, class Z>
struct ReconstructBoundaryGradient {
    Q q;
    X gx;
    Y gy;
    Z gz;

    KOKKOS_INLINE_FUNCTION void operator()(const bc::FaceContext<Real>& c,
                                           const scalar::Relation<Real>& r) const
    {
        const auto p = static_cast<std::size_t>(c.owner);
        const Real px = static_cast<Real>(gx(p));
        const Real py = static_cast<Real>(gy(p));
        const Real pz = static_cast<Real>(gz(p));
        const Real owner_value = r.gradient_owner == Real{0} ? Real{0} : static_cast<Real>(q(p));
        const Real normal = r.normal_gradient(owner_value);
        const Real delta = normal - (c.nx*px + c.ny*py + c.nz*pz);
        gx(c.extended_index) = static_cast<typename X::non_const_value_type>(px + c.nx*delta);
        gy(c.extended_index) = static_cast<typename Y::non_const_value_type>(py + c.ny*delta);
        gz(c.extended_index) = static_cast<typename Z::non_const_value_type>(pz + c.nz*delta);
    }
};
```

```cpp
// gradient is an existing three-component FieldView with the full extended
// extent. Its owned-cell gradients have already been computed on exec.
const auto q = temperature.view().component(0);
const auto gx = gradient.component(0);
const auto gy = gradient.component(1);
const auto gz = gradient.component(2);

temperature_bc.evaluate_async(
    ReconstructBoundaryGradient{q, gx, gy, gz}, law,
    bc::GradientCorrection{gx, gy, gz});
```

The correction uses owner gradients to obtain `tau`; the sink then consumes the resulting normal derivative without applying that correction a second time. Reads use owned cells, writes use boundary slots, so the same gradient allocation can serve both roles without overlapping reads and writes. AoS component strides are retained. Scalar values and owned/MPI ghost gradient slots are unchanged.

This example does not require previously filled scalar boundary values. It does require valid owner values and gradients before submission. If gradient reconstruction itself depends on boundary data, initialization and any correction iterations belong to the solver; the boundary manager does not resolve that dependency cycle.

## Custom conditions

Register additional condition types as `Manager<Exec, Real, MyCondition, ...>` and bind them through `set` or `set_unlabelled`. For relation-based operations, each custom condition must have a device-callable const `operator()(const FaceContext<Real>&, Real increment)` returning `Relation<Real>`. The context carries the supplied time for time-dependent conditions.

The generic `make_relation` forwards the context and increment to the condition; automatic flux-law conversion applies only to built-in `FixedFlux`. Custom conditions manage any additional state and parameter validation themselves. The TOML parser supports built-ins only; custom types are bound in C++.

## Asynchrony, validation and lifetime

Manager construction is **blocking and local**. It builds host patch metadata and validates borrowed exterior geometry with a device reduction whose result is read on the host. Setup completes work on its execution-space instance before returning. It performs no halo exchange or MPI collective, although invalid input terminates through KRONA's MPI error handling.

After setup, `apply_*`, `evaluate_async` and `for_each_face_async` perform host dispatch and lightweight checks, then submit kernels on the stored `exec`. They introduce no explicit fence, host-result reduction, or per-face temporary allocation. A host execution backend may still execute a kernel synchronously. `for_each_patch` itself is a synchronous host callback loop; the visitor determines whether its own work is asynchronous.

Order producers and consumers through the **same execution-space instance**, available as `manager.execution_space()`, or establish an explicit dependency between different instances. Fence before external host access or releasing storage still used by queued work. The manager's destructor does not fence. Keep mesh backing arrays, field owners and all buffers referenced by laws, corrections and sinks alive until their work completes. Host mesh arrays may also be borrowed by `MeshStorage` on CPU.

Condition objects are copied into submitted operations. Changing a binding affects later submissions; it does not change already submitted scalar parameters. Views inside copied objects still refer to shared backing memory, whose updates must be ordered separately. Host configuration changes are not thread-safe concurrent operations.

Setup checks counts and grouping, required geometry extents, owned exterior owners, finite geometry, approximately unit normals, positive normal distance and positive area. It is not a full mesh validator: topology, geometric closure, all cell volumes and consistency between separately supplied host/compute arrays remain outside this check.

Applications check binding completeness and finite time. Filling checks output extent, accessibility and writable floating-point storage; built-in `GradientCorrection` checks the three input view descriptors over owned cells. They do not scan solution values, validate arbitrary sink outputs, inspect a generic law's backing storage, or prove custom functor correctness. These checks preserve the asynchronous application path. There is no separate asynchronous validation-result API.

## Current limits and possible extensions

- Geometry and patch ranges are attached once. Mesh motion or adaptation requires an explicit refresh/rebuild design, including lifetimes, revalidation and field resizing; no such update API exists yet.
- The provided flux adapter is isotropic. Anisotropic or coupled constitutive laws may need additional gradient information through a custom law; coefficient updates and implicit linearization remain solver responsibilities.
- Advective inflow/outflow selection, periodic mappings and coupled physical boundary conditions are not implemented by these scalar relations. Matrix assembly must handle owner-cell write conflicts and equation-level issues such as a pure-Neumann nullspace.
- One kernel per nonempty patch keeps dispatch simple. Batching small patches is a potential optimization if profiling shows launch overhead matters.
