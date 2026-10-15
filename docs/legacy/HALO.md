# Cell halo exchange

`krona::halo::Exchange<T, Exec>` exchanges selected mutable cell-field components using
the existing mesh halo maps. The exchanger knows scalar types, strides, cell indices,
peers and tags. It has no knowledge of physical variables, gradients, limiters or equations.

## Setup

Create the mesh storage with `MeshData::Halo` selected. Resolve field views once and
register them in a stable order. All fields include both owned and ghost cells.

```cpp
#include <krona/halo/exchange.hpp>

// mesh_storage and fields already exist; MPI and Kokkos are initialized.
const auto exec = fields.execution_space();
const auto density = fields.get_field_view<krona::StorageReal>("density");
const auto velocity = fields.get_field_view<krona::StorageReal, 3>("velocity");

krona::halo::Exchange<> state_halo(
    mesh_storage.view(), mesh_storage.halo_plan(), MPI_COMM_WORLD, 700, exec);
state_halo.add_field(density);
state_halo.add_field(velocity);
state_halo.prepare();
```

`add_field()` appends every component, ordered from zero to `N-1`.
`add_component(field.component(c))` registers a single component or another mutable
rank-one Kokkos View. Strided components are supported. Normal LayoutLeft components
use a specialized unit-stride packing path.

One group uses one scalar type: float, double, or a fixed-width signed/unsigned
32/64-bit integer. The default is `StorageReal`. Different precision types use
different groups. A gradient field is registered exactly like a vector or tensor field.
An occasional field can use a temporary exchanger; this allocates during its setup,
so keep a prepared exchanger when that exchange repeats frequently.

## Ordering and lifetime contracts

- Peer ranks are ranks in the supplied intracommunicator. Passing a different
  communicator requires correspondingly remapped peer ranks.
- Each peer must use the same scalar type, logical component order and MPI tag.
  Local entity counts and peer-list sizes may differ. Registration is not a negotiation.
- The sender's map order must match the receiver's map order for that peer.
  This contract comes from mesh partitioning/migration.
- Distinct groups that can be active together use distinct tags on the same communicator.
  Reserve those tags against other application traffic. No communicator is duplicated.
- Send indices reference owned cells; receive indices reference ghost cells.
  Receive destinations are unique. Registered component arrays must not overlap each other.
- Mesh map allocations and field allocations outlive the exchanger. Its component
  descriptors are non-owning; adding unrelated fields to the manager is allowed.
- The communicator, MPI and Kokkos remain active until the exchanger is destroyed.
  With MPI_THREAD_FUNNELED, call its host API only from the MPI initialization thread.
  A single exchanger is not thread-safe.

Construction and `prepare()` are local operations. They validate counts, ranks, tags,
map bounds and duplicate ghost destinations. These checks do not exchange schema data.
MPI errors and contract violations use `krona::mpi::check` / `krona::mpi::fatal`.
Application memory allocation failures during setup follow the caller's exception policy.

## Repeated exchange

```cpp
state_halo.start();

// Queue work that does not need updated ghosts, using the same exec instance.
// compute_interior(exec, ...);

state_halo.finish_async();

// Ghost upload/scatter precedes subsequent kernels on this same instance.
// compute_boundary(exec, ...);
```

`start()` posts receives first, queues one packing kernel, and makes the send buffer
ready before posting nonblocking sends. This includes an execution-space fence when
there is a send payload. Thus start is not a fully asynchronous device-stream enqueue.
It returns with MPI requests outstanding.

`finish_async()` blocks the host until all sends and receives complete, then queues
the receive copy, if needed, and one unpack kernel. It does not fence after unpacking.
Its name refers to the remaining Kokkos work, not to an asynchronous MPI wait.
Do not read ghosts concurrently from another execution-space instance. Use `wait()`
before host access or external consumers, or establish the required cross-instance
dependency explicitly. Producers on other instances must also complete before `start()`.

`wait()` completes an active exchange and fences pending unpack work. `exchange()`
is the blocking `start()` + `wait()` convenience path. Starting the next exchange
drains pending previous unpack work before reusing receive buffers. The destructor
also drains pending work; it must not be used as a substitute for normal step ordering.

Snapshots in send buffers allow modification of owned field values after `start()`.
Ghost modifications must respect queued unpack ordering. Multiple groups can be active
together when their destinations are disjoint and their tags differ.

MPI communication progress during computation depends on the MPI library and transport.
Nonblocking calls permit overlap but do not guarantee continuous background progress.

## Message and memory layout

Each peer receives one contiguous typed MPI message per group:
`[component 0 values][component 1 values]...[component K-1 values]`.
Each component plane follows that peer's map order. The full buffer concatenates peer
segments, each with its own cell count. Original field allocations are never transposed.
MPI uses the matching scalar datatype; counts are scalar entries, not bytes.

Packing gathers irregular owned-cell indices; unpacking scatters into ghosts.
Eliminating this gather requires a different mesh/storage contract or noncontiguous MPI
datatypes. The contiguous path keeps message count small and works for CPU and GPU buffers.

Serial/OpenMP use only two host communication buffers. No additional staging buffers
are allocated. CUDA/HIP HostStaged uses two device buffers and two persistent pinned
host buffers. Only the packed halo is copied, not complete fields or mesh arrays.
DeviceDirect allocates only the two device communication buffers.
Metadata includes component pointer/stride descriptors, peer segments, and one peer
index per send/receive map entry. Buffers, descriptors and MPI request storage are
allocated during prepare; the hot path performs no application-side heap allocations.
MPI and the backend may still allocate internally.

## GPU transport

HostStaged is the default and works without GPU-aware MPI:
device gather -> host copy -> MPI -> device copy -> ghost scatter.
CUDA/HIP staging uses their pinned host memory spaces.

To request direct device-buffer transfer:

```cpp
krona::halo::Exchange<> direct_halo(
    mesh_storage.view(), mesh_storage.halo_plan(), MPI_COMM_WORLD, 701, exec,
    krona::halo::Transfer::DeviceDirect);
```

The caller must verify that MPI supports those device pointers at runtime. CUDA-aware
support does not imply HIP-aware support. Direct mode does not automatically imply
GPUDirect RDMA or better performance, and the send readiness fence is still required.
On HostSpace, both modes use the same host-buffer path.

The implementation uses MPI-3-compatible int counts. A peer message larger than
INT_MAX scalar entries is rejected during prepare; split that group if needed.
Mesh maps and field registrations are immutable after prepare. Rebuild the exchanger
when adaptation, migration, allocation changes or a new halo plan changes these contracts.
