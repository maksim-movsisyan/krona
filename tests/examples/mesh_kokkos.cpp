#include <krona/mesh/compute/mesh_storage.hpp>

namespace krona::examples {

// Call after MPI/Kokkos initialization and mesh loading, before finalization.
// This is the local volume; use an MPI sum to obtain the global volume.
AccumulationReal local_owned_volume(const mesh::MeshPart& host_mesh)
{
    using Exec = execution::ExecutionSpace;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    const Exec exec{};
    mesh::compute::MeshStorage<Exec> storage(host_mesh, {}, exec);
    const auto geometry = storage.view().geometry;

    AccumulationReal volume = 0;
    Kokkos::parallel_reduce("owned cell volume", Policy(exec, 0, host_mesh.n_own),
        KOKKOS_LAMBDA(const LocalIndex c, AccumulationReal& sum) {
            sum += static_cast<AccumulationReal>(geometry.cell_volume(c));
        }, volume);

    // Required before releasing borrowed/device storage used by queued work.
    exec.fence("owned cell volume complete");
    return volume;
}

} // namespace krona::examples
