#pragma once

#include <krona/transport/scalar/convection/plan.hpp>
#include <krona/transport/scalar/convection/reconstruction.hpp>
#include <krona/transport/scalar/detail/views.hpp>

#include <cmath>
#include <type_traits>

namespace krona::transport::scalar::convection::detail {

/// @brief Fills in the default end and checks the interval against the inner face numbering.
/// @param faces Half-open interval supplied by the caller.
/// @param n_inner Number of local faces between two cells, including MPI interfaces.
inline FaceRange checked_inner_range(FaceRange faces, LocalIndex n_inner) {
    if (faces.end == -1)
        faces.end = n_inner;
    mpi::require(faces.begin >= 0 && faces.end >= faces.begin && faces.end <= n_inner,
                 "Scalar convection: invalid inner face range");
    return faces;
}

/// @brief Custom mass flux providers manage their own input checks.
/// @tparam Exec Execution space that will read the provider.
/// @tparam Flux Callable providing the signed mass rate.
template<class Exec, class Flux>
void validate_flux(const Flux&, std::size_t) {}

/// @brief Checks that a stored flux supplies an entry for each local face.
/// @tparam Exec Execution space that will read the flux array.
/// @tparam View Type of the mass flux array.
/// @param flux Provider wrapping the array.
/// @param faces Number of faces in the complete local mesh.
template<class Exec, class View>
void validate_flux(const FaceMassFlux<View>& flux, std::size_t faces) {
    scalar::detail::validate_view<Exec>(flux.values, faces);
}

/// @brief Checks that the boundary manager borrows the geometry from this compute mesh.
/// @tparam Mesh Type of the plan's device mesh view.
/// @tparam Boundary Scalar manager that will supply the patch conditions.
/// @param mesh Device mesh from the convection plan.
/// @param boundary Manager whose face numbering and view storage must match.
template<class Mesh, class Boundary>
void validate_boundary(const Mesh& mesh, const Boundary& boundary) {
    const auto b = boundary.geometry_view();
    const auto& g = mesh.geometry;
    mpi::require(b.n_cells == mesh.sizes.n_cells && b.n_inner_faces == mesh.sizes.n_inner_faces &&
        b.face_owner.data() == mesh.connectivity.face_owner.data() &&
        b.face_x.data() == g.face_centroid_x.data() && b.face_y.data() == g.face_centroid_y.data() &&
        b.face_z.data() == g.face_centroid_z.data() &&
        b.normal_x.data() == g.face_normal_x.data() && b.normal_y.data() == g.face_normal_y.data() &&
        b.normal_z.data() == g.face_normal_z.data() && b.face_area.data() == g.face_area.data(),
        "Scalar convection: boundary manager must use the same compute mesh");

    // A first-order convection plan can omit the cell centres that the boundary manager holds.
    if (mesh.has(mesh::device::MeshData::CellGeometry))
        mpi::require(b.cell_x.data() == g.cell_centroid_x.data() && b.cell_y.data() == g.cell_centroid_y.data() &&
            b.cell_z.data() == g.cell_centroid_z.data(), "Scalar convection: boundary cell geometry mismatch");
}

/// @brief Rejects an invalid evaluation time before handing it to the providers.
/// @tparam Real Arithmetic type used by the operator.
/// @param time Time at which the face terms are evaluated.
template<class Real>
void validate_time(Real time) {
    mpi::require(std::isfinite(time), "Scalar convection: time must be finite");
}

/// @brief Checks the fields borrowed by the built-in providers.
/// @tparam Exec Execution space that will read the provider inputs.
/// @tparam Real Floating-point type used to evaluate face fluxes.
/// @tparam Flux Callable providing the signed mass rate.
/// @tparam Reconstruction Policy providing face trace increments.
/// @param plan Convection plan defining the required input sizes.
/// @param flux Mass rate provider for this launch.
/// @param reconstruction Reconstruction policy for this launch.
/// @param with_boundary Whether to include the boundary formula in the reconstruction checks.
template<class Exec, class Real, class Flux, class Reconstruction>
void validate_providers(const ConvectionPlan<Exec, Real>& plan, const Flux& flux, const Reconstruction& reconstruction,
                        bool with_boundary) {
    static_assert(!std::is_arithmetic_v<Reconstruction>, "Pass time after reconstruction and inflow policies");
    const auto sizes = plan.mesh_view().sizes;
    validate_flux<Exec>(flux, static_cast<std::size_t>(sizes.n_faces));
    validate_reconstruction<Exec>(reconstruction, static_cast<std::size_t>(sizes.n_cells), with_boundary);
}

/// @brief Checks that the system has storage and entry lists for this local mesh.
/// @tparam Exec Execution space that will write the system.
/// @tparam Target System slice containing the matrix, RHS and entry map.
/// @param target Writable system to increment.
/// @param sizes Local mesh sizes defining the required map and RHS extents.
template<class Exec, class Target>
void validate_system(const Target& target, const mesh::device::MeshSizes& sizes) {
    const auto rows = static_cast<std::size_t>(sizes.n_own);
    scalar::detail::validate_output<Exec>(target.values, target.values.extent(0));
    scalar::detail::validate_output<Exec>(target.rhs, rows);
    mpi::require(target.rhs.extent(0) == rows && target.entries.diagonal.extent(0) == rows &&
        target.entries.owner_entries.extent(0) == static_cast<std::size_t>(sizes.n_inner_faces) &&
        target.entries.neighbour_entries.extent(0) == static_cast<std::size_t>(sizes.n_internal_faces),
        "Scalar convection: incompatible system slice/entry map");
}

/// @brief Checks field and residual extents and rejects using the same view for both.
/// @tparam Exec Execution space that will access the views.
/// @tparam Field Rank-one view of the transported scalar.
/// @tparam Output Writable rank-one view of the owned residual.
/// @param field Scalar input.
/// @param output Residual to increment.
/// @param input_size Number of cell values that must fit in the field.
/// @param rows Number of owned entries that must fit in the output.
template<class Exec, class Field, class Output>
void validate_residual(const Field& field, const Output& output, std::size_t input_size, std::size_t rows) {
    scalar::detail::validate_view<Exec>(field, input_size);
    scalar::detail::validate_output<Exec>(output, rows);
    mpi::require(rows == 0 || static_cast<const void*>(field.data()) != static_cast<const void*>(output.data()),
                 "Scalar convection: in-place residual is unsupported");
}

} // namespace krona::transport::scalar::convection::detail
