# Fields architecture

KRONA stores field values directly in the memory space where they are used.
The fields module separates ownership, kernel access, and host-side name lookup.
It keeps no permanent host mirror and performs no halo exchange.

Examples assume initialized MPI and Kokkos. Owners must be destroyed before
Kokkos finalization. All field types below live in `krona::fields`.

## Objects and ownership

| Object | Responsibility |
| --- | --- |
| `FieldView<T, Components, Space, Layout>` | Typed, unmanaged descriptor passed by value to kernels. It does not own or copy values. |
| `FieldStorage<T, Components, Exec, Layout>` | Owns one allocation in `Exec::memory_space`; exposes mutable or read-only views and can enqueue fills. |
| `AbstractField` | Host-side name, location, and virtual metadata interface. |
| `FieldsManager<Exec>` | Owns field storage through a name registry and submits operations on its stored execution-space instance. |

Name lookup, RTTI, and virtual metadata calls remain on the host. Resolve field
names before dispatch and retain typed views for computation. Kernels receive
only descriptors; they do not access the registry. Host RTTI must be enabled.

Managers and storage objects are neither copyable nor movable. Adding fields
preserves existing allocations and views. There are no erase or resize methods.
Views, component subviews, and descriptors returned by `kokkos_view()` do not
extend storage lifetime. Finish every user of an allocation before destroying
its owner; destruction does not insert a fence.

## Locations and sizes

The module accepts local counts through `FieldSizes`; it has no mesh-type
dependency and does not derive or validate these counts against a mesh.

| `FieldLocation` | Number of entities |
| --- | --- |
| `OwnCell` | `own_cells` |
| `Cell` | `cells`, including owned and MPI ghost cells |
| `CellExtnd` | `cells + bfaces`, with additional entries for boundary-face values |
| `Face` | `faces`, including local internal, interface, and boundary faces |
| `Node` | `nodes` |
| `Custom` | An explicitly supplied count |

Cell fields follow the mesh's local cell numbering: owned cells first, then MPI
ghosts. For `CellExtnd`, boundary entries follow the cell portion. The caller
provides the mapping from boundary faces to these entries; the manager only
determines allocation size. A location does not provide communication or boundary
condition handling.

```cpp
#include <krona/fields/fields_manager.hpp>

namespace fields = krona::fields;
using Exec = krona::execution::ExecutionSpace;
using Real = krona::StorageReal;

Exec exec{};

// Illustrative local counts; obtain the actual counts from the local mesh.
fields::FieldSizes sizes{
    .own_cells = 100,
    .cells = 120,
    .faces = 340,
    .bfaces = 40,
    .nodes = 180
};
fields::FieldsManager<Exec> manager(sizes, exec);

auto density = manager.add_field<Real>("density", fields::FieldLocation::CellExtnd);
auto velocity = manager.add_field<Real, 3>("velocity", fields::FieldLocation::CellExtnd);
auto stress = manager.add_field<Real, 9>("stress", fields::FieldLocation::OwnCell);
auto flags = manager.add_field<int>("flags", fields::FieldLocation::Cell);
auto scratch = manager.add_field<Real>("scratch", 64, fields::FieldLocation::Custom);
```

The explicit-size overload also accepts mesh-associated locations, but its count
must equal the corresponding `FieldSizes` count. `Custom` requires this overload.

## Types, layouts, and access

The manager defaults to `StorageReal`, one component, and `Kokkos::LayoutLeft`.
Component count is a positive compile-time constant; other counts, such as a
five-component state, use the same implementation. Storage requires a mutable,
trivially copyable value type and either `LayoutLeft` or `LayoutRight`.

Scalars use a rank-one `Kokkos::View<T*>`. Multicomponent fields use
`Kokkos::View<T*[Components]>`, with a static second extent. For multicomponent
fields, `LayoutLeft` makes entity indices contiguous within a component;
`LayoutRight` makes components contiguous within an entity.

```cpp
auto state = manager.add_field<Real, 5>(
    "state", fields::FieldLocation::Cell,
    fields::FieldInitialization::Uninitialized);

auto packed_velocity = manager.add_field<Real, 3, Kokkos::LayoutRight>(
    "packed_velocity", fields::FieldLocation::OwnCell);
```

`FieldView` exposes `value_type`, `memory_space`, `array_layout`, `view_type`,
and the static `components` count. `FieldStorage` uses its `Exec` template
argument to select execution and memory placement. The aliases `ScalarView`,
`VectorView`, and `TensorView` select 1, 3, and 9 components respectively.

| Accessor | Meaning |
| --- | --- |
| `size()`, `empty()` | Entity count and emptiness. |
| `field(i, c)` | Component `c` of entity `i`, for every component count. |
| `field(i)`, `field[i]` | Scalar access, available for one component. |
| `field.x(i)`, `field.y(i)`, `field.z(i)` | Access for three-component vectors. |
| `field(i, row, column)` | Full 3x3 tensor access; component index is `row * 3 + column`. |
| `field.component(c)` | Unmanaged rank-one Kokkos subview; it may be strided. |
| `field.kokkos_view()` | Underlying unmanaged descriptor for interoperability. |
| `field.data()` | Base pointer; memory accessibility and strides still apply. |

Tensor components are ordered `xx, xy, xz, yx, yy, yz, zx, zy, zz`, independently
of physical layout. Do not assume that a component subview is contiguous or that
`data()` can be flattened in entity-major order. Value accessors may be used on
the host only when the field's memory is host-accessible; inspecting sizes or
copying descriptors does not dereference field memory.

## Lookup, constness, and metadata

Lookup requires the exact value type, component count, and layout used at
creation. For `fill_field`, `T` is explicitly specified or defaults to
`StorageReal`; the fill value does not deduce the field type.

```cpp
auto rho = manager.get_field_view<Real>("density");
auto u = manager.get_field_view<Real, 3>("velocity");
auto packed = manager.get_field_view<Real, 3, Kokkos::LayoutRight>("packed_velocity");

manager.fill_field<Real>("density", Real{1});
manager.fill_field<int>("flags", 0);
manager.fill_field<Real, 5>("state", Real{0});

auto read_only_u = u.as_const();
const auto& read_only_manager = manager;
auto read_only_rho = read_only_manager.get_field_view<Real>("density");

const fields::AbstractField* metadata = manager.get_field("velocity");
if (metadata) {
    auto count = metadata->size();
    auto components = metadata->component_count();
    auto bytes = metadata->byte_size();
    // Use the metadata on the host.
}
```

Constness of the descriptor does not make values read-only. Value constness is
encoded by `FieldView<const T, ...>`. Mutable-to-const conversion and `as_const()`
share the allocation without copying; const-manager lookup exposes const values,
including through `data()`.

`find_field<T, Components, Layout>()` returns typed storage or `nullptr` for a
missing field or type mismatch. `get_field()` returns immutable metadata or
`nullptr`. `has_field()` distinguishes absence from an empty field;
`get_field_size()` returns zero in both cases. `byte_size()` reports the value
buffer's span in bytes, including any layout padding, but excludes metadata.

Empty or duplicate names, strict lookup mismatches, and invalid sizes are
contract errors reported through the KRONA MPI error layer. These checks do not
validate numerical values or mesh consistency.

## Execution and lifetime

The manager retains the `Exec` instance passed to its constructor. Obtain it
through `execution_space()` and use that same instance for dependent kernels
and copies.

| Operation | Completion behavior |
| --- | --- |
| `add_field(..., Zero)` (default) | Allocates and registers storage, then queues a fill with `T{}`. No completion fence. |
| `add_field(..., Uninitialized)` | Allocates and registers storage without initializing values. |
| `fill_field(...)` | Queues a fill of all entities and components. |
| `create_host_buffer(...)` | Allocates independent, uninitialized host-accessible storage; copies no values. |
| `copy_to_host_async(...)` | Allocates a host buffer and queues a field-to-host copy. |
| `copy_from_host_async(...)` | Queues a host-to-field copy from the supplied View. |
| `fence()` | Waits for work on the manager's execution-space instance. |

Allocation and registration occur during `add_field`, but zero initialization
need not be complete when it returns. Work subsequently submitted to the same
instance is ordered after the fill, so dependent kernels need no intervening
fence. Directly constructed `FieldStorage` always leaves values uninitialized;
its `fill(exec, value)` queues initialization explicitly.

There is no automatic ordering between different execution-space instances.
Coordinate external producers and consumers explicitly. An uninitialized field
must be written before it is read. Keep the manager, standalone storage, and all
transfer buffers alive until their pending users complete.

Here, asynchronous means that operations use the explicit-execution-space
Kokkos API without an unconditional completion fence on the normal path. It does
not promise that every call returns immediately: allocations may block, CPU
backends may execute synchronously, and transfers involving ordinary pageable
host memory may block or require staging. Host buffers are not guaranteed to be
pinned. Initialization and snapshot failure paths fence before propagating an
exception so pending work can finish safely.

## Explicit host transfers

Host buffers are independent allocations even when fields already live on the
host. Modifying one does not update a field until an upload is submitted.

```cpp
auto host_velocity = manager.create_host_buffer<Real, 3>("velocity");
for (std::size_t i = 0; i < host_velocity.extent(0); ++i)
    for (int c = 0; c < 3; ++c)
        host_velocity(i, c) = Real{0};

manager.copy_from_host_async<Real, 3>("velocity", host_velocity);
auto snapshot = manager.copy_to_host_async<Real, 3>("velocity");

// The upload precedes the snapshot on the same instance.
manager.fence();
// snapshot is now readable; host_velocity can now be changed or released.
```

Do not read or write the snapshot before completion. An upload source must
remain alive and unchanged until completion. The field owner must also remain
alive. `copy_to_host_async` allocates a new buffer on every call; it does not
maintain or update a cached host mirror.

### Layout compatibility

`copy_from_host_async` checks that its source is a host-accessible Kokkos View
with the matching value type, rank, and extents. These checks do not guarantee
that every layout or strided source can be copied on every backend.

Kokkos can convert layouts entry by entry when an execution space can access
both arrays. For ordinary host memory and device-only GPU memory, direct
transfer instead requires compatible contiguous representations. Different
layouts or incompatible strides can require an intermediate allocation, which
this module does not create automatically. A buffer returned by
`create_host_buffer` preserves the field's layout and padding and is the
portable choice for uploads.

To import an external host View with a different layout, stage explicitly:

```cpp
// external_host_view has matching value type, rank, and extents.
auto compatible = manager.create_host_buffer<Real, 3>("velocity");
Kokkos::deep_copy(compatible, external_host_view); // Host-side layout conversion.
manager.copy_from_host_async<Real, 3>("velocity", compatible);
manager.fence();
```

The first copy uses the synchronous Kokkos overload. The host source must be
ready before that copy. The subsequent upload follows the normal lifetime
contract. See the official [Kokkos deep_copy requirements and layout conversion
example](https://kokkos.org/kokkos-core-wiki/API/core/view/deep_copy.html).

## Current limits and potential improvements

- **External host layouts:** layout conversion is caller-managed. A future
  explicit staging operation could support incompatible layouts and strides,
  with clear allocation and completion costs. More specific compatibility
  diagnostics could accompany it.
- **Repeated host transfers:** snapshots allocate each time. An overload that
  copies into a caller-owned reusable host buffer could avoid repeated
  allocations. A selectable pinned-memory path could improve GPU transfer
  overlap, subject to backend behavior and buffer lifetime requirements.
- **Changing mesh sizes or numbering:** counts are fixed at manager construction
  and fields cannot be resized or remapped. Mesh motion with unchanged entity
  numbering can retain allocations; adaptation or reordering needs a coordinated
  field-transfer/remapping design and explicit rules for existing views.
- **Communication and boundary conditions:** locations describe storage only.
  Halo exchange, boundary-value updates, and mappings for appended boundary
  entries belong to their respective consumers; the fields module provides no
  automatic updates.

These are extension points, not additional facilities in the current API.