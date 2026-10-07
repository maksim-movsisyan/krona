#include <krona/mesh/compute/aux_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>

namespace krona::examples {

// Demonstration only: caches are already prepared by the existing host algorithms.
// In a solver keep these storages alive across steps instead of creating them per call.
AccumulationReal local_incident_face_distance_sum(
    const mesh::MeshPart& host_mesh,
    const mesh::MeshAuxConnectivity& host_connectivity,
    const mesh::MeshAuxGeometry& host_geometry)
{
    using Exec = execution::ExecutionSpace;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    const Exec exec{};

    // No base geometric arrays are needed by this particular calculation.
    mesh::compute::MeshStorage<Exec> storage(host_mesh, {.data = mesh::compute::MeshData::None}, exec);
    mesh::compute::AuxConnectivityStorage<Exec> connectivity_storage(
        host_mesh, host_connectivity, mesh::AuxConnType::CellFaces, exec);
    mesh::compute::AuxGeometryStorage<Exec> geometry_storage(
        host_mesh, host_geometry, mesh::AuxGeomType::FaceCellDistance, exec);

    const auto connectivity = connectivity_storage.view();
    const auto geometry = geometry_storage.view();
    const auto n_own = storage.view().topology.n_own;
    AccumulationReal sum = 0;
    Kokkos::parallel_reduce("incident face distances", Policy(exec, 0, n_own),
        KOKKOS_LAMBDA(const LocalIndex cell, AccumulationReal& result) {
            const auto faces = connectivity.cell_faces_of(cell);
            for (std::size_t i = 0; i < faces.size(); ++i)
                result += static_cast<AccumulationReal>(geometry.face_cell_dist(faces[i]));
        }, sum);
    exec.fence("incident face distances complete");
    return sum;
}

} // namespace krona::examples
