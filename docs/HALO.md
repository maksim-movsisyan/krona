# Halo exchange architecture

`krona::halo::Exchange<T, Exec>` exchanges registered cell-field components
through existing mesh halo maps. It gathers owned values into communication
buffers, exchanges one message per active peer and direction, and scatters
received values into ghost cells. Field names and physical meaning remain
outside the module.

Registration and synchronous preparation happen once. Subsequent cycles reuse
the allocated buffers, component descriptors, maps, and MPI request array.
There are no application-side buffer allocations in `start()`, `finish_async()`,
or `wait()`; Kokkos and MPI may still allocate internally.

## Setup and registration

Examples assume initialized MPI and Kokkos, a valid host mesh `host_mesh`, and
an existing `manager` containing `density` and `velocity`. The fields and mesh
storage use the same execution-space type. Fields include owned and ghost cells;
`CellExtnd` fields may also contain appended boundary entries.

```cpp
#include <krona/halo/exchange.hpp>
#include <krona/fields/fields_manager.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>

namespace compute = krona::mesh::compute;
namespace halo = krona::halo;
using Exec = krona::execution::ExecutionSpace;
using Real = krona::StorageReal;

const auto exec = manager.execution_space();
const auto comm = krona::mpi::CommView::world();
compute::MeshStorage<Exec> mesh_storage(
    host_mesh, {.data = compute::MeshData::Halo}, exec);

const auto density = manager.get_field_view<Real>("density");
const auto velocity = manager.get_field_view<Real, 3>("velocity");

// Reserve this tag against other traffic using the same communicator and peers.
halo::Exchange<Real, Exec> state_halo(
    mesh_storage.view(), mesh_storage.halo_plan(), comm, 700, exec);
state_halo.add_field(density);
state_halo.add_field(velocity);
state_halo.prepare();
```

An existing `MeshStorage` can be reused when `MeshData::Halo` is selected;
`MeshData::FiniteVolume` already includes it. The constructor accepts a borrowed
`mpi::CommView` and does not duplicate the communicator.

`add_field()` appends all components in ascending component order.
`add_component()` accepts a mutable rank-one Kokkos View, including a strided
component subview. Both require the exact scalar type `T`, memory accessible to
`Exec`, and at least `mesh.sizes.n_cells` entries per component. The default `T`
is `StorageReal`; another type must have a `krona::mpi::datatype<T>()` mapping.
Different scalar types use separate exchanger objects.

For example, register selected components of an AoS field in a separate group:

```cpp
const auto transport = manager.add_field<Real, 3, Kokkos::LayoutRight>(
    "transport_vector", krona::fields::FieldLocation::CellExtnd);

halo::Exchange<Real, Exec> selected_halo(
    mesh_storage.view(), mesh_storage.halo_plan(), comm, 701, exec);
selected_halo.add_component(transport.component(2));
selected_halo.add_component(transport.component(0));
selected_halo.prepare();
```

The unregistered component is unchanged by this exchange. Boundary entries
appended after the cell portion are unchanged for every component.
`component_count()` reports the number of registered scalar components: four
for `state_halo` and two for `selected_halo` above.

Each component descriptor stores a base pointer and an element stride. If all
registered strides are one, packing and scattering use a unit-stride
specialization. Otherwise they use the stored strides. No field-layout
conversion or temporary field copy is required for registration.

Registration closes after `prepare()`. Preparation requires at least one
component, including on ranks with zero cells or no peers. Each object can be
prepared only once and is neither copyable nor movable.

## Mesh and communication contracts

`MeshView` provides the gather/scatter maps and local cell counts. The host-only
`HaloPlanView` supplies peer ranks and offsets into those maps. Offsets count
cells, not scalar values or bytes.

| Input | Meaning |
| --- | --- |
| `send_owned_local` | Owned cell indices gathered for outgoing messages. |
| `recv_ghost_local` | Ghost cell indices receiving incoming values. |
| `nb_ranks[p]` | Peer rank in the supplied communicator. |
| `send_offsets[p:p+2]` | Range in the send map belonging to peer `p`. |
| `recv_offsets[p:p+2]` | Range in the receive map belonging to peer `p`. |

The exchanger requires the following contracts:

- Peer ranks are valid and unique in the supplied intracommunicator. Using a
  different rank numbering requires remapping the plan's peer ranks.
- For each communicating pair, the scalar type, tag, component count, component
  order, and component meaning agree. Field names are not transmitted.
- The sender's map order matches the receiver's map order for that peer, and
  the corresponding send and receive counts agree. Local mesh sizes and peer
  list order need not be identical on different ranks.
- Send indices belong to `[0, n_own)`; receive indices belong to
  `[n_own, n_cells)`. Receive destinations are unique across the local plan.
  Sending the same owned cell to different peers is allowed.
- Registered components do not overlap in the elements that unpacking writes.
  Checking duplicate base pointers does not detect arbitrary aliasing between
  offset or strided views.
- Simultaneously active groups with overlapping source/destination pairs use
  distinct tags or communicator contexts. Reserve these tags against other
  application messages as well.

Only registered components and receive-map entries are updated. The exchanger
does not supply boundary conditions, face/node communication, or accumulation
semantics: received values replace ghost values.

For each peer, the wire format is SoA regardless of source field layout. If
`width` components and `count` cells are sent, each component contributes one
contiguous plane of `count` scalar values. In the combined local buffer:

```text
offset = peer_cell_offset * width + component * count + slot_within_peer
```

Setup and exchange methods introduce no collective handshake or barrier.
`prepare()` is local; matching peers must nevertheless use compatible plans
and registration. Isolated ranks are supported.

## Execution and synchronization

The exchanger stores its `Exec` instance and exposes it through
`execution_space()`. Submit producers, independent work, and dependent ghost
consumers to that same instance to obtain execution ordering. Producers on
other instances must be synchronized before their data is packed.

| Method | Behavior before returning |
| --- | --- |
| `prepare()` | Validates local maps, allocates reusable storage, and completes metadata uploads. |
| `start()` | Posts receives, packs outgoing values, stages them if needed, waits for send-buffer readiness, then posts sends. MPI communication may remain active. |
| `finish_async()` | Waits for all MPI requests, then queues receive upload and ghost scatter without a final unpack fence. |
| `wait()` | Completes the current exchange; fences the instance when unpacking is pending. Safe to call again after completion. |
| `exchange()` | Runs `start()` followed by `wait()`. |

The normal overlapped sequence is:

```cpp
// Submit owned-value producers on exec before starting the exchange.
state_halo.start();

// Submit independent computation on exec; it must not require incoming ghosts.
state_halo.finish_async();

// Submit ghost-dependent computation on exec. Unpack is ordered before it.

// Complete the exchange before an external consumer accesses its ghost values.
state_halo.wait();
```

Place independent computation after `start()` when aiming to overlap it with
MPI. Work already queued before `start()` is included in its send-readiness
fence. Once `start()` returns, MPI reads the packed send buffer rather than the
field allocation, so later owned-value updates do not alter that outgoing
snapshot.

`finish_async()` is asynchronous with respect to receive upload/unpacking;
its `MPI_Waitall` can block the host. `start()` also waits for packing and
optional send staging. These waits establish the dependencies required by the
current MPI/Kokkos protocol, including in `DeviceDirect` mode.

The state sequence is `Idle` → `Communicating` → `Unpacking` → `Idle`. With no
incoming payload, MPI completion returns directly to `Idle`. Starting while
`Communicating` is an error. Starting while `Unpacking` first calls `wait()` so
the previous operation releases the receive buffers.

`pending()` reports this logical state, not hardware or MPI completion. It does
not poll MPI or advance the state. A physically completed unpack remains
logically pending until `wait()` or the next `start()` processes that state.

`wait()` completes this exchange, not arbitrary work submitted to `exec`.
In particular, a cycle with no incoming payload may finish without an execution
fence. Use an explicit execution-space fence when external access requires
completion of other kernels or copies.

MPI nonblocking calls permit communication/computation overlap but do not
guarantee effective background progress. See the official
[MPI nonblocking communication semantics](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node71.htm).

## Transfer modes

`Transfer::HostStaged` is the default portable path. With plain `HostSpace`
field storage, MPI uses the packed host buffers directly. Other memory spaces
use device-side communication buffers plus a host staging pair. The staging
space is selected from the actual `Exec` type:

| Execution space | Host staging space |
| --- | --- |
| `Kokkos::Cuda` | `Kokkos::CudaHostPinnedSpace` |
| `Kokkos::HIP` | `Kokkos::HIPHostPinnedSpace` |
| Other execution spaces | `Kokkos::HostSpace` |

`Transfer::DeviceDirect` passes the packed buffers in `Exec::memory_space`
directly to MPI. It requires MPI support for that memory space; the exchanger
does not detect support or fall back automatically. On a host execution space,
ordinary MPI supports the host buffers. Direct mode does not itself guarantee
GPUDirect RDMA or exclude internal staging by MPI.

```cpp
// Use only after establishing MPI support for Exec::memory_space.
halo::Exchange<Real, Exec> direct_halo(
    mesh_storage.view(), mesh_storage.halo_plan(), comm, 702, exec,
    halo::Transfer::DeviceDirect);
direct_halo.add_field(density);
direct_halo.prepare();
```

## Validation and lifetime

The constructor checks selected halo data, counts, tag bounds, local plan
offsets, and peer ranks. Registration checks component size, accessibility,
mutability, and scalar type, with duplicate-base-pointer rejection for nonempty
components. `prepare()` checks map index ranges and duplicate receive
destinations. Map validation runs on the host; non-HostSpace maps are copied
there once, with synchronization before inspection.

These are local checks. They do not compare protocol schemas or map order
between ranks, prove arbitrary component non-aliasing, or require every ghost
cell to appear in the receive map. Contract violations are reported through the
KRONA MPI error layer.

Mesh map allocations and field allocations must remain alive and retain their
addresses throughout exchanger use. Registration stores raw descriptors even
when `add_component()` receives a managed View. Adding unrelated fields to a
manager is safe; replacing or reallocating registered storage is not.
The host plan's ranks and offsets are copied into internal peer metadata by
the constructor; the map allocations themselves remain borrowed.

Destroy the exchanger before its communicator, registered field storage, mesh
maps, and runtimes. Its destructor drains an active MPI/unpack cycle and may
block. Complete any separate users of the fields before destroying their owners.
With `MPI_THREAD_FUNNELED`, call the host API from the thread that initialized
MPI. A single exchanger has no internal thread-safety mechanism.

## Current limits and potential improvements

- **Progress API:** `test()` or `progress()` could poll MPI and queue unpacking
  once communication completes, allowing the host to continue other work.
  There is currently no nonblocking completion query or background progress
  mechanism in this module.
- **Completion granularity:** `finish_async()` waits for both sends and receives
  before scattering. Separating receive readiness from send completion, or
  processing completed peers individually, could expose additional overlap.
  Receive/send buffer lifetimes must remain protected.
- **Message counts:** peer payloads use MPI-3 `int` counts and must not exceed
  `INT_MAX` scalar values. Oversized payloads are rejected; automatic chunking
  is not implemented. Persistent MPI requests and packing tile choices could
  also be evaluated through profiling for repeatedly used plans.
- **Changing meshes and allocations:** the prepared plan and component
  descriptors are fixed. Reordering, adaptation, or field reallocation requires
  rebuilding the exchanger. Mesh motion that preserves numbering and allocation
  addresses does not change the exchange plan.
