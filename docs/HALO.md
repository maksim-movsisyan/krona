# Halo Exchange

`krona::halo::Exchange<T, Exec>` updates registered **ghost-cell values** from their owning ranks. It uses existing cell send/receive maps, packs all registered scalar components into one message per peer and reuses its buffers across exchanges.

Header: `krona/halo/exchange.hpp`. The exchanger does not discover neighbours, build a halo, apply boundary conditions or look up field names.

## Register once, prepare once, exchange repeatedly

This example loads a prepared mesh, registers a scalar and an AoS vector, closes registration with `prepare()` and exchanges both fields at each step.

```cpp
#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/fields/fields_manager.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/execution/utils.hpp>
#include <krona/mpi/runtime.hpp>
#include <krona/mpi/error.hpp>
#include <Kokkos_Core.hpp>
#include <cstddef>

int main(int argc, char** argv)
{
    krona::mpi::Runtime runtime{argc, argv};
    return krona::mpi::guarded([&] {
        Kokkos::ScopeGuard kokkos{argc, argv};
        namespace fields = krona::fields;
        namespace device = krona::mesh::device;
        using Real = krona::StorageReal;
        using Exec = krona::execution::ExecutionSpace;
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        Exec exec;
        const auto comm = krona::mpi::CommView::world();

        const auto host_mesh = krona::io::solver_mesh::import_mesh_hdf5("mesh.h5", comm);
        device::MeshStorage<Exec> mesh{
            host_mesh, {.data = device::MeshData::Halo}, exec};
        const auto owned = static_cast<std::size_t>(host_mesh.n_own);
        const auto cells = static_cast<std::size_t>(host_mesh.n_cells);
        const int rank = comm.rank();
        fields::FieldsManager<Exec> state{{.own_cells = owned, .cells = cells}, exec};
        const auto density = state.add_field<Real>(
            "density", fields::FieldLocation::Cell,
            fields::FieldInitialization::Uninitialized);
        const auto velocity = state.add_field<Real, 3, Kokkos::LayoutRight>(
            "velocity", fields::FieldLocation::Cell,
            fields::FieldInitialization::Uninitialized);

        // Reserve this tag for this exchange group on comm.
        krona::halo::Exchange<Real, Exec> halo{
            mesh.view(), mesh.halo_plan(), comm, 910, exec};
        halo.add_field(density);  // One scalar component.
        halo.add_field(velocity); // x, y, z, in that order.
        halo.prepare();          // Registration is now permanently closed.

        for (int step = 0; step < 3; ++step) {
            krona::execution::complete(exec, [&] {
                Kokkos::parallel_for("set_local_state", Policy(exec, 0, cells),
                    KOKKOS_LAMBDA(std::size_t cell) {
                        const Real value = cell < owned
                            ? static_cast<Real>(rank + step + 1) : Real{-1};
                        density(cell) = value;
                        for (int c = 0; c < 3; ++c) velocity(cell, c) = value;
                    });
                halo.exchange(); // MPI and incoming ghost updates complete here.
            });
            // Registered ghosts now contain their donor ranks' values.
        }
        return 0;
    });
}
```

No fence is needed between the producer kernel and `start()` / `exchange()` when both use the same execution-space instance. The outer `execution::complete` also finishes local user work, including on isolated ranks, and drains submitted work if submission throws.

Fields and mesh-map storage are declared before the exchanger so that they outlive it. See [FIELDS](FIELDS.md) for field allocation and [MESH-DEVICE](MESH-DEVICE.md) for selecting halo maps.

## Constructor and input maps

The constructor accepts `(mesh_view, host_plan, comm, tag, exec, transfer)`. The default transfer mode is `Transfer::HostStaged`.

| Input | Required contents |
| --- | --- |
| `mesh_view` | A `mesh::device::MeshView<Exec::memory_space>` with `MeshData::Halo`, `sizes.n_own`, `sizes.n_cells` and the two halo maps. Other mesh array groups are unnecessary. |
| `mesh_view.halo.send_owned_local` | Local owned-cell indices to gather; every entry is in `[0, n_own)`. |
| `mesh_view.halo.recv_ghost_local` | Local ghost indices to overwrite; every entry is in `[n_own, n_cells)`, with no repeated destinations. |
| `host_plan` | `mesh::host::HaloPlanView`: unique valid peer ranks, plus `send_offsets` and `recv_offsets` partitioning the corresponding maps by peer. |
| `comm`, `tag` | A live, non-null communicator with matching rank numbering, and a tag in its valid MPI range. Reserve the tag for this group. |
| `exec` | Instance used for metadata copies, packing, staging and unpacking. Kokkos and MPI must be active. Use the MPI main thread under the usual `MPI_THREAD_FUNNELED` runtime. |

Offsets count **cells**, not scalar values or bytes. Each offset array has `peer_count + 1` nondecreasing entries, begins at zero and ends at its map length. For each transfer between two ranks, the sender’s cell list and the receiver’s ghost-cell list must describe the same global cells in the same order. The two lists may start at different offsets in their respective maps, and their local cell indices may differ.

Pass `view()` and `halo_plan()` from the same `MeshStorage` object. The mesh view supplies the send/receive cell maps; the halo plan supplies the peer ranks and offsets into those maps. The exchanger copies the peer ranks and segment boundaries but borrows the map arrays, which must remain valid and unchanged throughout its lifetime.

prepare() is synchronous: it validates the maps, allocates reusable buffers, and completes all setup copies before returning. It performs no MPI communication and does not require other ranks to call it at the same time.

Ranks without exchange peers are supported. Every `Exchange` must register at least one component before `prepare()`, even when no data will be transferred. A zero-length component is valid only when the mesh has no local cells (`n_cells == 0`).

## Supported fields and registration

All components in one exchanger have the same mutable scalar type `T`. It defaults to `StorageReal`. `T` must have a `krona::mpi::datatype<T>()` mapping and support assignment in Kokkos kernels for the chosen execution space. Typical choices are `float`, `double` and signed/unsigned integer types, including fixed-width aliases. Other mapped types, such as `bool` or `std::complex`, also depend on MPI mapping availability and suitability for the selected Kokkos backend. There is no automatic serialization or mixed-type conversion.

| Registration | Supported data |
| --- | --- |
| `add_field(view)` | Mutable `fields::FieldView<T, N, Space, Layout>`; registers components `0 ... N-1`. Scalars, vectors, full tensors and other positive fixed component counts are supported. |
| `add_component(view)` | Mutable rank-one Kokkos view of exactly `T`, including strided component subviews or externally owned arrays. |

The execution space must be able to access each component's memory. SoA, AoS and accessible strided components can be mixed within one group. Packing handles their strides; the MPI message uses a packed SoA arrangement regardless of source layout.

Every registered field/component must contain at least `n_cells` entries and use the mesh's local **cell indexing**:

- `FieldLocation::Cell` is the normal choice.
- `CellExtnd` also works: appended boundary slots beyond `n_cells` remain untouched.
- An owned-only field is too short when ghosts exist. A custom array works if its length and indexing satisfy the same contract.
- Location tags are not inspected. Registering a face or node field does not create a face/node exchange; this exchanger still applies cell maps.

To exchange selected components instead of the whole vector, use a separate exchanger with its own tag:

```cpp
krona::halo::Exchange<Real, Exec> selected{
    mesh.view(), mesh.halo_plan(), comm, 911, exec};
selected.add_component(velocity.component(0)); // x
selected.add_component(velocity.component(2)); // z
selected.prepare();
selected.exchange(); // Ghost y values are not modified by this group.
```

The variable names are those from the main example. Do not register the same nonempty component twice. More generally, registered components must not write to the same scalar locations; arbitrary overlapping aliases are not fully detected. Registration stores pointers and strides without copying field values or taking ownership.

`component_count()` returns the total registered scalar components: scalar + vector = 4, a full tensor = 9. It is not the number of field objects. After `prepare()`, neither `add_field` nor `add_component` is allowed. Create a new exchanger to change the schema, pointers, maps, tag or transfer mode.

## Matching neighbours

There is no registration handshake between ranks. For each communicating pair, the application must provide:

- The same logical component order and scalar type. Field names are not transmitted or compared.
- Matching cell-slot order and payload counts in the peer maps.
- Matching communicator, tag and exchange round.

`prepare()` does not establish these remote agreements. Local success does not prove that neighbours registered compatible fields. For different scalar types, create separate exchangers with distinct tags. Independent groups may overlap if tags are distinct and their incoming writes do not conflict; otherwise serialize them.

Only peers with matching messages need to participate in a round. The exchanger uses point-to-point MPI and does not impose a barrier across the whole communicator. One-way payloads, zero-sized peer segments and ranks without peers are supported.

## Exchange steps

1. `start()` posts nonempty receives, gathers the current owned values into a private send buffer, completes outgoing staging if needed, then posts sends.
2. `finish_async()` waits for **both sends and receives** through `MPI_Waitall`, then queues incoming staging and scatter to the registered ghost slots.
3. `wait()` completes outstanding MPI and queued unpacking. `exchange()` is the blocking convenience path: `start(); wait();`.

Only registered components and mapped receive slots are overwritten. Owned cells, unregistered components and appended boundary slots are preserved. Values are assigned, not accumulated.

Outgoing values are a snapshot packed during `start()`. Once it returns, owned values may be updated while MPI uses the private send buffer. Do not access incoming ghost values concurrently with their pending scatter, except through correctly ordered work on the exchanger's instance.

## Asynchronous operations and completion

**`start()` is not a fully nonblocking launch, and `finish_async()` still blocks for MPI.** The asynchronous part after `finish_async()` is the queued receive upload/unpack.

| Method | What is complete when it returns? |
| --- | --- |
| Constructor, `add_field`, `add_component` | Local setup/registration only. No field exchange or producer completion is performed. |
| `prepare()` | Blocking setup: map checks, reusable buffers and uploaded metadata are ready. No remote schema check or field exchange. |
| `start()` | Receives and sends have been posted. With outgoing data, packing and optional device-to-host staging have completed before MPI uses the send buffer. MPI communication may still be pending. |
| `finish_async()` | MPI sends/receives are complete. Incoming host-to-device staging and ghost scatter may still be queued on `exec`. |
| `wait()` | This exchange's outstanding MPI and unpack work is complete. If already idle, it is a no-op. |
| `exchange()` | Blocking completion of this exchange's communication and ghost updates. |
| `pending()` | Reports bookkeeping state only. It does not poll MPI, drive progress or test device completion. |
| Destructor | Calls `wait()` when the exchange is pending; may block. Dependencies and runtimes must still be alive. |

For overlap, replace the blocking exchange with this sequence:

```cpp
halo.start();
// Queue owned-only or other independent work here; incoming ghosts are not ready.
halo.finish_async(); // Blocks for MPI, then queues ghost updates.
// Queue ghost-consuming kernels on halo.execution_space() here.
halo.wait();
```

A consumer queued **after** `finish_async()` on the same instance observes updated ghosts in order, without a host-side wait before its launch. Producers on another instance must finish before `start()` reads their values. Consumers on another instance need `halo.wait()` before using the received ghosts.

`wait()` / `exchange()` are not general fences for arbitrary user kernels. In particular, when no incoming unpack is pending, `wait()` may perform no execution-space fence. Complete any remaining user kernels with their own instance before releasing their arrays.

After `start()`, another `start()` is invalid until MPI completion has been processed. `finish_async()` is valid only for an active MPI exchange; calling it twice is an error. After `finish_async()`, `pending()` can remain true even if device work has already finished: `wait()` clears the unpacking state. A new `start()` from that state automatically waits before reusing buffers. Actual overlap and MPI progress depend on the MPI implementation and backend; these methods do not promise background progress.

## CPU/GPU transfer modes

| `Transfer` | MPI buffers and requirements |
| --- | --- |
| `HostStaged` (default) | Portable host-buffer MPI path. For CUDA/HIP execution, uses reusable pinned host staging buffers around the execution-space pack/unpack buffers. For other non-HostSpace execution spaces, uses ordinary host staging. |
| `DeviceDirect` | Gives MPI the packed buffers in `Exec::memory_space`. MPI must support that memory space; capability is not detected automatically. Packing is still performed and fenced before sends. |

With exactly `Kokkos::HostSpace`, both modes use the host pack/unpack buffers directly, without an extra staging copy. DeviceDirect does not send the original field allocations directly: it changes which **packed communication buffers** MPI sees.

For each peer, `cell_count * component_count()` must fit an MPI-3 `int` count, for both send and receive payloads. Oversized groups are rejected; split the group when component width causes the limit. If a single component already exceeds the peer limit, the current exchanger has no batching facility.

## Lifetime

Keep registered field allocations, mesh-map arrays and the borrowed communicator valid for the exchanger's lifetime. Do not resize, replace or renumber them after registration/preparation. Adding unrelated fields to a `FieldsManager` preserves existing allocations and views, but does not add them to an already prepared exchanger.

The exchanger cannot be copied or moved. Its ordinary exchange path reuses prepared buffers. Its destructor drains its own pending communication/unpack, but does not establish completion of unrelated kernels or repair unmatched peer messages. Prefer explicit `wait()` at the intended completion point, and destroy the exchanger before fields, map owners, MPI and Kokkos runtimes.
