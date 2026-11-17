#pragma once

#include <krona/mesh/aux_types.hpp>
#include <krona/mesh/device/mesh_view.hpp>
#include <krona/mpi/error.hpp>

namespace krona::numerics::gradient {

/// @brief Selects how the operator computes one gradient per owned cell.
enum class Scheme {
    GreenGaussCell,  ///< Each cell gathers contributions from its faces.
    GreenGaussFace,  ///< Each face adds to the selected owned cells it touches.
    LeastSquaresFace ///< Each cell fits its gradient to samples across its faces.
};

/// @brief Chooses how Green-Gauss interpolates the value on an interior face.
enum class Interpolation {
    Arithmetic,      ///< Uses the mean of the owner and neighbour values.
    AuxiliaryWeights ///< Uses the owner weight stored in FaceInterpolationWeights.
};

/// @brief Numerical settings for building and applying a gradient operator.
struct Options {
    Scheme scheme = Scheme::LeastSquaresFace;
    Interpolation interpolation = Interpolation::Arithmetic;
    /// Relative Cholesky pivot threshold for a 3D least-squares stencil.
    double pivot_tolerance = 1.0e-12;
};

/// @brief Masks of the arrays that must be attached before constructing the operator.
struct Requirements {
    krona::mesh::device::MeshData mesh;
    krona::mesh::AuxConnType connectivity;
    krona::mesh::AuxGeomType geometry;
};

/// @brief Returns the mesh, connectivity and geometry groups needed to build the operator.
/// @param options Scheme and interpolation choice used to determine the groups.
/// The least-squares displacement cache is optional and can be attached separately.
[[nodiscard]] constexpr Requirements requirements(Options options = {}) noexcept {
    using MeshData = mesh::device::MeshData;
    using mesh::AuxConnType;
    using mesh::AuxGeomType;

    if (options.scheme != Scheme::GreenGaussCell &&
        options.scheme != Scheme::GreenGaussFace &&
        options.scheme != Scheme::LeastSquaresFace)
        mpi::fatal("Gradient: unknown scheme");

    if (options.interpolation != Interpolation::Arithmetic &&
        options.interpolation != Interpolation::AuxiliaryWeights)
        mpi::fatal("Gradient: unknown interpolation");

    // Current MeshData groups include the unused LS volumes, areas and normals.
    // Displacement vectors are an optional cache; centroids provide the fallback.
    return {MeshData::FaceCells | MeshData::Geometry,
            options.scheme == Scheme::GreenGaussFace ? AuxConnType::None : AuxConnType::CellFaces,
            options.scheme != Scheme::LeastSquaresFace &&
                    options.interpolation == Interpolation::AuxiliaryWeights ? AuxGeomType::FaceInterpolationWeights : AuxGeomType::None};
}

/// @brief Selects the owned cells whose gradients are written, from begin up to end.
/// With end == -1, selects all owned cells starting at begin.
struct CellRange {
    LocalIndex begin = 0, end = -1;
};

} // namespace krona::numerics::gradient
