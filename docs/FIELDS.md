# Fields

`krona::fields` stores scalar and multicomponent fields in Kokkos memory. A field has a name, an entity count, a fixed component count and a location tag. An **entity** is one cell, face, node or custom array entry.

| Header / object | Purpose |
| --- | --- |
| `krona/fields/fields_manager.hpp` — `FieldsManager<Exec>` | Owns a registry of named fields and uses one execution-space instance for initialization, fills and transfers. |
| `krona/fields/field_storage.hpp` — `FieldStorage<T, Components, Exec, Layout>` | Owns one allocation in `Exec::memory_space`. Can also be used without a manager. |
| `krona/fields/fields_view.hpp` — `FieldView<T, Components, Space, Layout>` | Non-owning descriptor passed by value into kernels. `Space` is a memory space. |
| `krona/fields/transfer.hpp` — `copy_to_dofs[_async]()`, `copy_from_dofs[_async]()` | Copies fields/packs to and from flat cell-major DOFs without a linear-algebra dependency. |
| `krona/fields/pack.hpp` — `pack(...)`, `field_pack(...)` | Combines fields/components into one non-owning component pack. Preserves strides and supports mixed numeric types. |

The module does not depend on a mesh object. It does not perform MPI communication, halo exchanges, boundary-condition updates or automatic resizing after mesh changes.

## Create fields, upload values and use a kernel

```cpp
#include <krona/fields/fields_manager.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/mpi/error.hpp>
#include <Kokkos_Core.hpp>
#include <cstddef>

int main(int argc, char** argv) {
    krona::mpi::Runtime runtime{argc, argv};
    return krona::mpi::guarded([&] {
        Kokkos::ScopeGuard kokkos{argc, argv};
        using namespace krona::fields;
        using Real = krona::StorageReal;
        using Exec = krona::execution::ExecutionSpace;
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        Exec exec;

        // Example local counts: 8 owned cells and 2 MPI ghosts.
        FieldsManager<Exec> fields{
            {.own_cells = 8, .cells = 10, .faces = 14, .bfaces = 4, .nodes = 12},
            exec};
        const auto density = fields.add_field_async<Real>(
            "density", FieldLocation::Cell, FieldInitialization::Uninitialized);
        const auto velocity = fields.add_field_async<Real, 3>(
            "velocity", FieldLocation::Cell, FieldInitialization::Uninitialized);

        auto initial = fields.create_host_buffer<Real>("density");
        for (std::size_t i = 0; i < initial.extent(0); ++i)
            initial(i) = Real{1};

        krona::execution::complete(exec, [&] {
            fields.copy_from_host_async<Real>("density", initial);
            fields.fill_field_async<Real, 3>("velocity", Real{0});
            Kokkos::parallel_for("update_owned_cells",
                Policy(exec, 0, fields.sizes().own_cells),
                KOKKOS_LAMBDA(std::size_t cell) {
                    density(cell) += Real{1};
                    velocity.x(cell) = density(cell);
                });
        }); // Waits for this group of operations, including on failure.

        auto snapshot = fields.copy_to_host_async<Real>("density");
        fields.fence(); // The host may now read the snapshot.
        krona::mpi::require(snapshot(0) == Real{2}, "Unexpected owned value");
        krona::mpi::require(snapshot(9) == Real{1}, "Ghost slot was overwritten");
        return 0;
    });
}
```

The upload, fill and kernel use the **same execution-space instance**, so no fence is needed between those dependent operations. `execution::complete` waits at the end and drains submitted work if submission throws. The later host snapshot requires its own completion wait before CPU access.

Ghost values in this example are just local array entries. Updating them from other MPI ranks requires a separate halo exchange. Keep the application workflow inside `mpi::guarded` so escaping exceptions become fatal job errors.

## Sizes and locations

`FieldSizes` contains local counts supplied by the caller. A manager stores a fixed copy and requires `own_cells <= cells`.

| `FieldLocation` | Number of entities | Intended entries |
| --- | --- | --- |
| `OwnCell` | `own_cells` | Owned cells only. |
| `Cell` | `cells` | Owned cells followed by MPI ghost cells. |
| `CellExtnd` | `cells + bfaces` | Cell entries followed by one additional slot per boundary face. |
| `Face` | `faces` | All local faces, including partition interfaces and external boundaries. |
| `Node` | `nodes` | All local nodes. |
| `Custom` | Explicit count | Scratch arrays or other caller-defined entries. |

For a validated, grouped `host::MeshPart`, use `n_own`, `n_cells`, `n_faces`, `n_faces - n_inner_faces` and `n_nodes` respectively for `FieldSizes::own_cells`, `cells`, `faces`, `bfaces` and `nodes`. Convert these nonnegative counts to `size_t`. No mesh header is needed by the fields module itself.

The intended `CellExtnd` ranges are `[0, own_cells)` for owned cells, `[own_cells, cells)` for ghosts and `[cells, cells + bfaces)` for appended boundary slots. The manager allocates these slots; the caller defines their mapping to boundary faces and their values. It does not turn them into mesh cells or apply boundary conditions.

Counts must be consistent with the caller's indexing. In particular, `cells + bfaces` must fit `size_t`; the current `CellExtnd` addition is not overflow-checked.

## Register and find fields

Creation supports a location-derived size or an explicit count:

```cpp
auto pressure = fields.add_field_async<Real>("pressure", FieldLocation::Cell);
auto scratch = fields.add_field_async<Real, 3>(
    "scratch", 64, FieldLocation::Custom, FieldInitialization::Uninitialized);
```

The names and aliases in snippets are those from the first example. Creation returns a writable view. The default scalar type is `StorageReal`, component count is 1, layout is `Kokkos::LayoutLeft`, and initialization is `FieldInitialization::Zero`.

- `Zero` queues a fill with `T{}` over **all entities and components**, including ghost and appended boundary entries. For numeric types, this is zero. It does not wait for completion.
- `Uninitialized` skips the fill. Write every value that will be read before using it.
- Names must be nonempty and unique within the manager, regardless of field type.
- `Custom` requires the explicit-count overload. For other locations, an explicit count must match the corresponding `FieldSizes` count.
- Owned storage requires non-const, trivially copyable `T`, a positive compile-time component count and either `LayoutLeft` or `LayoutRight`. Allocation byte-count overflow is checked separately from the `CellExtnd` addition above.
- Zero-sized fields are valid and remain registered. Adding fields does not invalidate existing field views.

Lookup uses the exact scalar type, component count and layout used at registration. It performs no value conversion. Resolve names on the host before launching kernels; kernel views perform no registry lookup or virtual dispatch.

| Method | Result |
| --- | --- |
| `get_field_view<T, N, Layout>(name)` | Typed view; aborts for a missing name or type/components/layout mismatch. A const manager returns read-only values. |
| `find_field<T, N, Layout>(name)` | Typed storage pointer, or `nullptr` for a missing name or mismatch. |
| `get_field(name)` | `const AbstractField*` metadata, or `nullptr`. Provides `name()`, `location()`, `size()`, `component_count()` and `byte_size()`. |
| `has_field(name)` | Whether the name exists. |
| `get_field_size(name)` | Entity count; returns 0 for a missing name **or an existing empty field**. Use `has_field` to distinguish them. |
| `sizes()` / `execution_space()` | Stored counts / the manager's execution-space instance. |

`byte_size()` reports allocated value bytes, including any layout padding, and excludes registry/metadata overhead. The manager has no remove or resize operation; create new storage when entity counts change.

## Kernel access and layout

`FieldView::size()` counts entities, not scalar values. A scalar uses a rank-one Kokkos view; a multicomponent field uses a rank-two view with a static second extent. Entity and component indices must be in range.

| Field | Access |
| --- | --- |
| Scalar (`N = 1`) | `value(i)`, `value[i]` or `value(i, 0)`. |
| Any component count | `value(i, component)`. |
| Vector (`N = 3`) | `value.x(i)`, `.y(i)`, `.z(i)`. |
| Full tensor (`N = 9`) | `value(i, row, column)`, where row and column are in `[0, 3)`. Component order: `xx, xy, xz, yx, yy, yz, zx, zy, zz`. |
| One component | `value.component(c)` returns a rank-one Kokkos subview sharing the data. |
| Interoperability | `value.kokkos_view()` returns the unmanaged Kokkos descriptor; `value.data()` returns its base pointer. |

Aliases `ScalarView<T, Space, Layout>`, `VectorView<...>` and `TensorView<...>` select 1, 3 and 9 components.

| Layout | Logical arrangement | Fixed-component access |
| --- | --- | --- |
| `LayoutLeft` (default) | SoA: entity index varies fastest within each component. | `component(c)` has unit entity stride. |
| `LayoutRight` | AoS: components of an entity are adjacent. | A component subview is strided across entities. |

Pass the chosen layout consistently to creation and typed lookup/copy/fill methods. For example, `get_field_view<Real, 3, Kokkos::LayoutRight>(name)` retrieves an AoS vector field. Tensor component numbering stays the same for both layouts. Do not assume a component subview is a packed flat array; use its Kokkos indexing/strides. GPU pointers from `data()` cannot be dereferenced directly by the CPU.

### Read-only values

`const` on the descriptor does not make values read-only: the `const auto density` in the example is writable inside the kernel. Constness of `T` controls value access.

```cpp
const auto read_only = density.as_const();
const auto also_read_only = std::as_const(fields).get_field_view<Real>("density");
```

Include `<utility>` for `std::as_const`. These views share the original allocation; neither copies values. Mutable-to-const view conversion is supported, but const-to-mutable conversion is not. Use the registered, non-const scalar type as the lookup template argument, including when looking up through a const manager.

## Pack fields and components

Include `<krona/fields/pack.hpp>`. `pack(...)` accepts one or more field views, rank-one Kokkos views or existing packs. A multicomponent field contributes all its components in their original order. Arguments are appended from left to right.

Using `density` and `velocity` from the first example:

```cpp
#include <krona/fields/pack.hpp>

const auto state = krona::fields::pack(density, velocity);
static_assert(decltype(state)::components == 4);
// Order: density, velocity.x, velocity.y, velocity.z.

krona::execution::complete(exec, [&] {
    Kokkos::parallel_for("write_packed_fields",
        Policy(exec, 0, state.size()), KOKKOS_LAMBDA(std::size_t i) {
            state(i, 0) = Real{1};
            state(i, 1) = Real{2};
            state(i, 2) = Real{3};
            state(i, 3) = Real{4};
        });
});
```

These writes modify the original fields. Packing allocates no buffers, copies no values and retains no allocation ownership. SoA and AoS fields can be mixed: each component keeps its original entity stride.

Select, reorder or regroup components explicitly:

```cpp
const auto selected = krona::fields::pack(velocity.component(2), density.component(0));
const auto velocity_pack = krona::fields::field_pack(velocity); // Same as pack(velocity).
const auto combined = krona::fields::pack(density, velocity_pack);
// selected: velocity.z, density; combined: density, velocity.x, velocity.y, velocity.z.
```

### Access and mixed scalar types

All components must use the same memory space and arithmetic scalar types. Their scalar types, layouts, lengths and constness may differ.

| Access / property | Contract |
| --- | --- |
| `size()` / `empty()` | Common entity extent: the minimum input component length / whether it is zero. |
| `components` | Compile-time total component count. |
| `get<C>(i)` | Reference to component `C` in its exact scalar type; preserves that component's constness. `C` is a compile-time index. |
| `component<C>()` | Unmanaged rank-one Kokkos view with the original scalar type/stride, restricted to `size()` entities. |
| `operator()(i, c)` — same scalar type | Reference; writable only when every input component is mutable. |
| `operator()(i, c)` — mixed scalar types | Value converted to `std::common_type_t` of the scalar types. Use `get<C>()` or `component<C>()` for writes. |
| `homogeneous` / `writable` | Whether scalar types match, ignoring const / whether runtime-indexed access returns a writable reference. |
| `as_const()` | Another pack of the same buffers with every component read-only. |

For example, a mixed pack can combine floating-point data and integer flags:

```cpp
const auto samples = fields.add_field_async<float>("samples", FieldLocation::Cell);
const auto flags = fields.add_field_async<int>("flags", FieldLocation::Cell);
const auto mixed = krona::fields::pack(samples, density, flags);

krona::execution::complete(exec, [&] {
    Kokkos::parallel_for("write_mixed_fields",
        Policy(exec, 0, mixed.size()), KOKKOS_LAMBDA(std::size_t i) {
            mixed.get<0>(i) = 1.5f;    // float reference
            mixed.get<1>(i) = Real{2}; // density's original scalar type
            mixed.get<2>(i) = 7;       // int reference
            // mixed(i, c) reads a common-type value; it is not a writable reference.
        });
});
```

Runtime reads use ordinary numeric conversion, which may lose precision for some type/value combinations. Typed access performs no conversion. Constness of the pack descriptor itself does not make its data read-only; use read-only input views or `as_const()`. With a mixture of mutable and read-only inputs, `get<C>()` preserves each input's individual access rights, while same-type runtime access is read-only for the whole pack.

### Bounds, lifetime and execution

- Indices satisfy `i < size()` and `c < components`. A longer input's tail is outside the pack's range. Any empty input makes the pack empty; never index an empty pack.
- The caller ensures that entity `i` has the intended meaning in every component. Packing does not compare location tags or reconcile cell/face numbering.
- `pack`/`field_pack` run on the host and only construct descriptors. They perform no allocation, data transfer, kernel launch, fence or MPI operation. Invalid nonempty pointer/stride descriptors terminate through `mpi::require`.
- Pass a pack by value into a kernel whose execution space can access its memory space. Its element accessors, `component<C>()` and `as_const()` are device-callable and do not launch work.
- Complete producers before consuming values, or enqueue dependent work on the same execution-space instance. Packing itself does not wait for initialization or uploads.
- Keep all original allocations alive and at stable addresses until every user finishes. Passing an owning Kokkos view into `pack` does not transfer or retain ownership. Packed components may alias; the caller must avoid conflicting writes.

## Transfer fields to and from flat DOFs

Include `<krona/fields/transfer.hpp>`. These helpers work with rank-one Kokkos views; no linear-algebra backend is required.

```cpp
const auto packed = krona::fields::pack(density, velocity);
// dofs.extent(0) == owned_cells * decltype(packed)::components
krona::fields::copy_to_dofs_async(exec, packed.as_const(), dofs);
krona::fields::copy_from_dofs_async(exec, dofs, packed);
exec.fence("field/DOF transfers complete");
```

DOF order is `cell*components + component`, with components in pack order. The DOF view's extent selects the cell prefix and must be divisible by pack width. Every component must be at least that long. Field tails are preserved; field/component and DOF strides are respected.

A single field or rank-one component view can replace `packed`. Mixed numeric types are supported, and each assignment converts directly to the exact destination scalar type. Read-only inputs are allowed; every unpacked component must be writable. Conversion representability, actual entity/DOF order and non-overlapping source/destination storage are caller contracts. Unpack destinations must not overlap each other.

Each asynchronous operation submits one kernel and performs no buffer allocation or final fence. All buffers must be accessible to `exec` and remain alive through completion. Queue dependent producers and consumers on that instance or synchronize explicitly. `copy_to_dofs()` / `copy_from_dofs()` are blocking counterparts that fence only `exec`.

If the DOF view comes from an object with its own access-batch protocol, a fields-level fence does not reset that object's state. Follow its commit contract; the linalg transfer adapters do this for their blocking forms.

## Host buffers and transfers

`create_host_buffer<T, N, Layout>(name)` allocates an independent host Kokkos buffer with the matching shape and layout. Values are uninitialized; no field values are copied. The buffer is independent even when the field itself resides in `HostSpace`.

`copy_to_host_async<T, N, Layout>(name)` allocates such a buffer, queues a snapshot and returns the buffer. Keep both it and the field alive until completion. Read or modify the snapshot only after `fields.fence()`. Later field updates do not change a completed snapshot, and editing the snapshot does not modify the field.

`copy_from_host_async<T, N, Layout>(name, source)` queues an upload. The source must be a host-accessible Kokkos view with the same scalar type, rank and extents. Keep it alive and unchanged until completion. No type conversion occurs. Prefer `create_host_buffer` for a matching layout; other views must also satisfy Kokkos's memory-layout and deep-copy requirements.

To reuse an existing host buffer for repeated downloads, use the exposed descriptor instead of allocating a new snapshot each time:

```cpp
auto reusable = fields.create_host_buffer<Real>("density");
Kokkos::deep_copy(fields.execution_space(), reusable, density.kokkos_view());
fields.fence(); // Safe to read reusable now.
```

## Asynchronous operations and completion

The manager's optional `Exec exec = {}` selects the instance used for queued work. **Field creation with zero initialization, fills and transfers do not provide a general completion guarantee on return.** This differs from mesh storage constructors that fence their uploads. For fields, use `fields.fence()` when completed values are required on the host or before releasing buffers.

| Method | Completion behaviour |
| --- | --- |
| `FieldsManager(...)` | Host setup and validation only. Does not wait for previously submitted execution-space work. |
| `FieldStorage(...)` | Allocates uninitialized storage; no initialization kernel or completion fence is requested by KRONA. |
| `add_field_async(..., Zero)` | Allocates/registers storage and queues the `T{}` fill. Treat initialization as pending until completion. |
| `add_field_async(..., Uninitialized)` | Allocates/registers storage; no fill is submitted. Values remain undefined until written. |
| `fill_field_async(...)` / `FieldStorage::fill_async(exec, value)` | Queue a fill of every entity and component. No KRONA completion wait. |
| `copy_to_host_async(...)` | Allocates a host buffer and queues a download. Returned values may still be pending. |
| `copy_from_host_async(...)` | Queues an upload. The source must remain alive and unchanged until completion. |
| `create_host_buffer(...)` | Host allocation only; does not copy values or wait for field results. |
| Lookup, metadata, `.view()`, `.as_const()`, `.component()` | Host/descriptor operations only; no field-data transfer or completion wait. |
| `copy_to_dofs_async()` / `copy_from_dofs_async()` | Submit one copy/conversion kernel on the supplied instance; no final fence. |
| `copy_to_dofs()` / `copy_from_dofs()` | Submit the transfer and fence the supplied instance. |
| `pack(...)` / `field_pack(...)` | Host descriptor construction only; no allocation, field-data copy, submitted work or completion wait. |
| `FieldsManager::fence()` | **Blocking:** waits for work on the manager's execution-space instance. |
| Destruction | Releases owned allocations; the module does not provide an automatic wait-for-users contract. Complete all users first. |

These are completion contracts, not promises that a call never blocks the CPU. Allocation and Kokkos backend operations can take time or synchronize internally; some CPU or empty-field operations may finish synchronously. Do not rely on this incidental behaviour for correctness.

Dependent fills, copies and kernels submitted to the **same instance** can run in order without intermediate fences. Use `fields.execution_space()` for their execution policies, or pass the same `exec` when constructing the manager. Work on another instance needs explicit producer completion before consuming these arrays; the manager does not track other instances.

## Lifetime and standalone storage

Views, component subviews and descriptors returned by `kokkos_view()` do not own the field allocation. Keep the owning manager/storage alive until every kernel, copy and external user has finished. Manager and storage objects cannot be copied or moved. Take views from stable owners; copying a view does not extend the allocation's lifetime.

`FieldStorage` can be used directly when name-based lookup is unnecessary:

```cpp
FieldStorage<Real, 3, Exec> standalone{
    "standalone", 64, FieldLocation::Custom, exec};
const auto values = standalone.view();
krona::execution::complete(exec, [&] {
    standalone.fill_async(exec, Real{0});
});
```

The standalone constructor always leaves values uninitialized. Its location is metadata; it has no manager counts against which to validate the explicit size. Use `std::as_const(standalone).view()` for read-only access and `standalone.create_host_buffer()` for an independent uninitialized host buffer. For standalone fills and raw Kokkos copies, synchronize the execution-space instance you supplied.