#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mesh/device/validation.hpp>
#include <krona/numerics/gradient/types.hpp>

#include <cmath>
#include <limits>

namespace krona::numerics::gradient::detail {

/// @brief Checks host descriptors and returns the required extended input length.
/// @tparam Space Kokkos memory space of the mesh and auxiliary arrays.
template <class Space>
std::size_t validate_descriptors(mesh::device::MeshView<Space> m,
                                 mesh::device::AuxConnectivityView<Space> conn,
                                 mesh::device::AuxGeometryView<Space> geometry, Options options) {
    execution::require_kokkos();
    const auto& s = m.sizes;
    const auto needed = gradient::requirements(options);

    mpi::require(std::isfinite(options.pivot_tolerance) && options.pivot_tolerance > 0 &&
                 options.pivot_tolerance < 1,
                 "Gradient: pivot tolerance must be finite and between zero and one");

    // Grouped faces define the boundary suffix used to index the extended input.
    mpi::require(s.faces_grouped && s.n_own >= 0 && s.n_cells >= s.n_own && s.n_internal_faces >= 0 &&
                     s.n_internal_faces <= s.n_inner_faces && s.n_inner_faces <= s.n_faces,
                 "Gradient: requires valid counts and grouped faces");
    mpi::require(m.has(needed.mesh), "Gradient: requires face connectivity and cell/face geometry");
    const auto cells = static_cast<std::size_t>(s.n_cells), faces = static_cast<std::size_t>(s.n_faces);
    const auto boundary = static_cast<std::size_t>(s.n_faces - s.n_inner_faces);
    mpi::require(cells <= std::numeric_limits<std::size_t>::max() - boundary,
                 "Gradient: extended size overflows size_t");

    mpi::require(m.connectivity.face_owner.extent(0) == faces && m.connectivity.face_neigh.extent(0) == faces,
                 "Gradient: face connectivity size mismatch");

    // Centroids cover all cells, including ghosts. Green-Gauss only divides by owned volumes.
    mpi::require(m.geometry.cell_centroid_x.extent(0) >= cells &&
                     m.geometry.cell_centroid_y.extent(0) >= cells &&
                     m.geometry.cell_centroid_z.extent(0) >= cells &&
                     m.geometry.cell_volume.extent(0) >= static_cast<std::size_t>(s.n_own),
                 "Gradient: cell geometry size mismatch");
    mpi::require(
        m.geometry.face_centroid_x.extent(0) == faces && m.geometry.face_centroid_y.extent(0) == faces &&
            m.geometry.face_centroid_z.extent(0) == faces && m.geometry.face_area.extent(0) == faces &&
            m.geometry.face_normal_x.extent(0) == faces && m.geometry.face_normal_y.extent(0) == faces &&
            m.geometry.face_normal_z.extent(0) == faces,
        "Gradient: face geometry size mismatch");

    // The gather schemes need one face list per owned cell. The scatter scheme iterates faces directly.
    if (needed.connectivity != mesh::AuxConnType::None) {
        mpi::require(conn.has_cell_faces() && conn.n_cells_own == s.n_own && conn.n_faces == s.n_faces &&
                         conn.cell_faces_offsets.extent(0) == static_cast<std::size_t>(s.n_own) + 1,
                     "Gradient: attach ready owned-cell-to-face CSR");
        mpi::require(s.n_own != 0 || conn.cell_faces.extent(0) == 0,
                     "Gradient: empty partition has nonempty CSR");
    }

    // Check that attached auxiliary face arrays match the number of faces in the base mesh.
    if (geometry.has_face_cell_dist_vector())
        mpi::require(geometry.n_faces == s.n_faces && geometry.face_cell_dist_x.extent(0) == faces &&
                         geometry.face_cell_dist_y.extent(0) == faces &&
                         geometry.face_cell_dist_z.extent(0) == faces,
                     "Gradient: auxiliary displacement size mismatch");
    if (needed.geometry != mesh::AuxGeomType::None)
        mpi::require(geometry.has_face_interp_weights() && geometry.n_faces == s.n_faces &&
                         geometry.face_interp_weight.extent(0) == faces,
                     "Gradient: attach ready owner interpolation weights");

    return cells + boundary;
}

/// @brief Runs the shared mesh checks on the groups selected for this scheme.
/// @tparam Data Which base mesh arrays to check.
/// @tparam ConnectivityMask Which auxiliary adjacency tables to check.
/// @tparam GeometryMask Which cached metrics to check.
/// @tparam Exec Kokkos execution space used for the checks.
/// @tparam Space Kokkos memory space of the arrays.
template <mesh::device::MeshData Data, mesh::AuxConnType ConnectivityMask, mesh::AuxGeomType GeometryMask,
          class Exec, class Space>
void validate_selected(const Exec& exec, mesh::device::MeshView<Space> mesh_view,
                       mesh::device::AuxConnectivityView<Space> connectivity,
                       mesh::device::AuxGeometryView<Space> geometry) {
    namespace validation = mesh::device::validation;
    // Cross-check adjacency against face owners and neighbours whenever a cell-to-face table is used.
    const auto result = validation::validate<Data, ConnectivityMask, GeometryMask, true>(
        exec, mesh_view, connectivity, geometry);
    mpi::require(result.shapes == 0, "Gradient: invalid selected mesh/auxiliary descriptors");
    mpi::require(result.faces == 0, "Gradient: invalid face topology or metrics");
    mpi::require(result.cell_faces == 0 && result.cell_incidence == 0 && result.face_incidence == 0,
                 "Gradient: invalid cell-to-face CSR or incidence");
}

/// @brief Checks mesh contents before any gradient kernel uses them.
/// @tparam Exec Kokkos execution space used for the checks.
/// @tparam Space Kokkos memory space of the arrays.
template <class Exec, class Space>
void validate_mesh(const Exec& exec, mesh::device::MeshView<Space> mesh_view,
                   mesh::device::AuxConnectivityView<Space> connectivity,
                   mesh::device::AuxGeometryView<Space> geometry, Options options) {
    using MeshData = mesh::device::MeshData;
    using mesh::AuxConnType;
    using mesh::AuxGeomType;
    namespace validation = mesh::device::validation;

    // Least squares needs valid face links and cell rows. Preparation checks each displacement.
    if (options.scheme == Scheme::LeastSquaresFace) {
        if (geometry.has_face_cell_dist_vector())
            validate_selected<MeshData::FaceCells, AuxConnType::CellFaces,
                              AuxGeomType::FaceCellDistanceVector>(exec, mesh_view, connectivity, geometry);
        else
            validate_selected<MeshData::FaceCells, AuxConnType::CellFaces, AuxGeomType::None>(
                exec, mesh_view, connectivity, geometry);
        return;
    }

    // Both Green-Gauss schemes use face geometry and may use cached interpolation weights.
    const auto green_gauss = [&]<AuxConnType ConnectivityMask>() {
        constexpr auto data = MeshData::FaceCells | MeshData::FaceGeometry;
        if (options.interpolation == Interpolation::AuxiliaryWeights)
            validate_selected<data, ConnectivityMask, AuxGeomType::FaceInterpolationWeights>(
                exec, mesh_view, connectivity, geometry);
        else
            validate_selected<data, ConnectivityMask, AuxGeomType::None>(exec, mesh_view, connectivity,
                                                                         geometry);
    };
    if (options.scheme == Scheme::GreenGaussCell)
        green_gauss.template operator()<AuxConnType::CellFaces>();
    else
        green_gauss.template operator()<AuxConnType::None>();

    // Only owned cells receive a gradient, so apply the volume check to that prefix.
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    std::size_t bad = 0;
    Kokkos::parallel_reduce(
        "gradient/validate_owned_geometry", Policy(exec, 0, mesh_view.sizes.n_own),
        validation::ValidateCells<MeshData::CellGeometry, AuxGeomType::None, Space>{mesh_view, {}}, bad);
    mpi::require(bad == 0, "Gradient: owned centres must be finite and volumes positive");
}

} // namespace krona::numerics::gradient::detail
