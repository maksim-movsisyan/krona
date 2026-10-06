#include <krona/mesh/compute/mesh_storage.hpp>

namespace krona::examples {
namespace {

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

    exec.fence("owned cell volume complete");
    return volume;
}

} // namespace
} // namespace krona::examples

int main(int argc, char** argv) {
    static_cast<void>(argc);
    static_cast<void>(argv);
    static_cast<void>(&krona::examples::local_owned_volume);  // чтобы не было unused-function
    return 1;  // ← не хватало `;`
}