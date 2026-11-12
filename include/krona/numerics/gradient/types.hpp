#pragma once

#include <krona/mesh/device/aux_view.hpp>
#include <krona/mpi/error.hpp>

namespace krona::numerics::gradient {

enum class Scheme { GreenGaussCell, GreenGaussFace, LeastSquaresFace };
enum class Interpolation { Arithmetic, AuxiliaryWeights };

struct Options {
    Scheme scheme = Scheme::LeastSquaresFace;
    Interpolation interpolation = Interpolation::Arithmetic;
    double pivot_tolerance = 1.0e-12;
};

struct Requirements {
    krona::mesh::device::MeshData mesh;
    krona::mesh::AuxConnType connectivity;
    krona::mesh::AuxGeomType geometry;
};

/// @brief Returns mandatory storage groups before creating the gradient plan.
[[nodiscard]] constexpr Requirements requirements(Options options = {}) noexcept {
    using MeshData = mesh::device::MeshData;
    using mesh::AuxConnType;
    using mesh::AuxGeomType;
    if (options.scheme != Scheme::GreenGaussCell && options.scheme != Scheme::GreenGaussFace &&
        options.scheme != Scheme::LeastSquaresFace)
        mpi::fatal("Gradient: unknown scheme");
    if (options.interpolation != Interpolation::Arithmetic && options.interpolation != Interpolation::AuxiliaryWeights)
        mpi::fatal("Gradient: unknown interpolation");
    // Current MeshData groups include the unused LS volumes, areas and normals.
    // Displacement vectors are an optional cache; centroids provide the fallback.
    return {MeshData::FaceCells | MeshData::Geometry,
            options.scheme == Scheme::GreenGaussFace ? AuxConnType::None : AuxConnType::CellFaces,
            options.scheme != Scheme::LeastSquaresFace && options.interpolation == Interpolation::AuxiliaryWeights
                ? AuxGeomType::FaceInterpolationWeights : AuxGeomType::None};
}

/// @brief Half-open owned-cell interval; end=-1 selects all remaining owned cells.
struct CellRange { LocalIndex begin = 0, end = -1; };

} // namespace krona::numerics::gradient
