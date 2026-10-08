# KRONA fields

`krona::fields` stores typed solution arrays in the selected Kokkos memory space. Names and metadata stay on the host; kernels receive lightweight field views. The manager performs no numerical updates, boundary conditions, or MPI exchange.

Mesh numbering and lifetime rules are described in [MESH-COMPUTE.md](MESH-COMPUTE.md).

## 1. Objects and field identity

| Object | Responsibility |
| --- | --- |
| `FieldsManager<Exec>` | Host registry, field creation, typed lookup, fills, and explicit transfers |
| `FieldStorage<T, Components, Exec, Layout>` | One owned field allocation; no permanent host mirror |
| `FieldView<T, Components, Space, Layout>` | Unmanaged descriptor passed by value to kernels |

Kokkos must be active. A manager stores one execution-space instance; use that instance for dependent kernels. Its counts are supplied through `FieldSizes` and remain fixed. Names must be nonempty and unique within the manager. Types and component counts are explicit: `T` is non-const and trivially copyable in storage; `Components` is a positive compile-time constant.

Defaults are `T = StorageReal`, one component, and `LayoutLeft`. Different fields may use different value types; arithmetic precision and units are the solver's responsibility.

| `FieldLocation` | Entity count and meaning |
| --- | --- |
| `OwnCell` | `own_cells`; cells advanced by this rank |
| `Cell` | `cells`; owned cells followed by MPI ghosts |
| `Face` | `faces`; local face copies, including cut and boundary faces |
| `Node` | `nodes`; all local node copies, without unique ownership implied |
| `Custom` | Explicit caller-provided size and indexing convention |

`Cell` contains no artificial boundary ghost cells. Mesh-associated explicit sizes must match `FieldSizes`. Location determines size/indexing only; it does not provide communication or FEM DOF numbering.

## 2. Typical use

The following fragment runs after Kokkos initialization, with a prepared `MeshView` named `mesh`:

```cpp
#include <krona/fields/fields_manager.hpp>

using Exec = krona::execution::ExecutionSpace;
using Index = krona::LocalIndex;
using Real = krona::StorageReal;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<Index>>;
using namespace krona::fields;

Exec exec{};
FieldsManager<> fields({
    .own_cells = static_cast<std::size_t>(mesh.sizes.n_own),
    .cells = static_cast<std::size_t>(mesh.sizes.n_cells),
    .faces = static_cast<std::size_t>(mesh.sizes.n_faces),
    .nodes = static_cast<std::size_t>(mesh.sizes.n_nodes)
}, exec);

auto density = fields.add_field<Real>("density", FieldLocation::Cell);
auto velocity = fields.add_field<Real, 3>("velocity", FieldLocation::Cell);

Kokkos::parallel_for("fields/initialize_owned",
    Policy(exec, 0, mesh.sizes.n_own), KOKKOS_LAMBDA(Index cell) {
        density(cell) = Real{1};
        velocity.x(cell) = Real{1};
        velocity.y(cell) = Real{0};
        velocity.z(cell) = Real{0};
    });
exec.fence("Fields ready for host access or destruction");
```

Resolve names before dispatch and cache views. `get_field_view<T, N, Layout>()` requires the exact type, component count, and layout used at creation. `find_field<T, N, Layout>()` returns `nullptr` for a missing or mismatched field. Registry lookup, RTTI, and virtual metadata access never enter kernels.

## 3. Layout and access

Scalars use `View<T*>`; multicomponent fields use `View<T*[Components]>`. Each field has one value allocation.

For indexing `field(entity, component)`:

| Layout | Organization | Component subview stride |
| --- | --- | --- |
| `LayoutLeft` — KRONA default | SoA; a fixed component is contiguous across entities | `1` |
| `LayoutRight` — supported alternative | AoS; an entity's components are adjacent | `Components` for this allocation |

`FieldView::size()` counts entities. The underlying Kokkos view's `size()` counts scalar values; `extent(0)` counts entities. Vector aliases have three components (`x/y/z`); tensor aliases store all nine entries in `xx,xy,xz,yx,yy,yz,zx,zy,zz` order, without symmetry compression.

`field.component(c)` returns a subview without copying values. `.data()` returns a raw pointer, not a host copy; obey memory-space access rules and the subview's stride. Prefer indexed view access. Layout can include padding, so obtain component pointers through subviews rather than hand-calculated offsets.

Read-only values require `FieldView<const T, ...>`, `view.as_const()`, or lookup through a const manager. A `const FieldView<T, ...>` still permits writes to values. Views, component subviews, and `kokkos_view()` do not keep allocations alive.

## 4. Initialization and transfers

Manager-created fields queue `T{}` initialization by default. `FieldInitialization::Uninitialized` skips it; initialize every entry before use. Direct `FieldStorage` construction also leaves values uninitialized. Numeric initialization does not make ghosts physically valid: exchange or explicitly supply ghost values before a stencil reads them. Whole-field host copies include ghost slots.

| Operation | Contract |
| --- | --- |
| `fill_field(...)` | Queues a fill of all entities/components on the manager's execution instance |
| `create_host_buffer(...)` | Allocates independent uninitialized host storage, even on CPU; copies nothing |
| `copy_to_host_async(...)` | Allocates an independent host snapshot and queues the copy |
| `copy_from_host_async(...)` | Queues a copy from caller-owned host storage with matching value type, rank, and extents |

```cpp
auto host = fields.copy_to_host_async<Real>("density");
fields.execution_space().fence("Host snapshot ready");
// host(i) now holds an independent value, including in CPU builds.
```

Operations on the same execution instance are ordered; they may be asynchronous depending on the backend. Synchronize before host reads or external MPI access. Upload sources must stay alive and unchanged until completion; download destinations must stay alive. Dependent work on another instance needs explicit synchronization. Manager operations do not retain caller-owned transfer buffers or complete MPI requests.

## 5. Lifetime and current scope

Adding fields preserves existing allocations and views. Erase, resize, rename, and mesh remapping are not implemented. Managers/storage are neither copyable nor movable. Complete all users before destruction and destroy owned buffers before Kokkos finalization. Destructors do not fence; on exceptions, drain pending work before unwinding owners.

Contract failures use `mpi::fatal`; successful manager operations perform no MPI calls. Registry mutation is a host control operation and is not designed for concurrent callers. Kernels may share views only with a race-free access pattern. Mesh adaptation, field checkpointing, and halo exchange require separate components.
