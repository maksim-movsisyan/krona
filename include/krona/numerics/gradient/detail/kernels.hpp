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
[[nodiscard]] constexpr Requirements requirements(Options options = {}) noexcept
{
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

namespace detail {

template<class T>
struct Component {
    T* data = nullptr;
    std::size_t stride = 1;
    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t i) const { return data[i * stride]; }
};

template<class T, std::size_t N>
struct Batch {
    Kokkos::Array<Component<const T>, N> input;
    Kokkos::Array<Component<T>, N> x, y, z;
};

template<class Space>
struct Metrics {
    template<class T> using Array = mesh::device::detail::MeshArray<T, Space, false>;
    // Capture only gradient inputs; node tables, halo plans and unrelated metrics stay outside kernels.
    struct MeshDescriptor {
        mesh::device::MeshSizes sizes;
        mesh::device::MeshData active_mask;
        struct { Array<LocalIndex> face_owner, face_neigh; } connectivity;
        struct {
            Array<GeometryReal> cell_centroid_x, cell_centroid_y, cell_centroid_z, cell_volume;
            Array<GeometryReal> face_centroid_x, face_centroid_y, face_centroid_z;
            Array<GeometryReal> face_normal_x, face_normal_y, face_normal_z, face_area;
        } geometry;
        bool has(mesh::device::MeshData flags) const
        {
            const std::uint64_t bits = static_cast<std::uint64_t>(flags);
            return (static_cast<std::uint64_t>(active_mask) & bits) == bits;
        }
    } mesh;
    struct ConnectivityDescriptor {
        LocalIndex n_cells_own, n_faces;
        bool ready;
        Array<std::size_t> cell_faces_offsets;
        Array<LocalIndex> cell_faces;
        bool has_cell_faces() const { return ready; }
    } connectivity;
    struct GeometryDescriptor {
        LocalIndex n_faces;
        bool vectors, weights;
        Array<GeometryReal> face_cell_dist_x, face_cell_dist_y, face_cell_dist_z, face_interp_weight;
        KOKKOS_INLINE_FUNCTION bool has_face_cell_dist_vector() const { return vectors; }
        bool has_face_interp_weights() const { return weights; }
    } geometry;
    Options options;

    Metrics(mesh::device::MeshView<Space> m, mesh::device::AuxConnectivityView<Space> c,
            mesh::device::AuxGeometryView<Space> g, Options o)
        : mesh{m.sizes, m.active_mask, {m.connectivity.face_owner, m.connectivity.face_neigh},
               {m.geometry.cell_centroid_x, m.geometry.cell_centroid_y, m.geometry.cell_centroid_z, m.geometry.cell_volume,
                m.geometry.face_centroid_x, m.geometry.face_centroid_y, m.geometry.face_centroid_z,
                m.geometry.face_normal_x, m.geometry.face_normal_y, m.geometry.face_normal_z, m.geometry.face_area}},
          connectivity{c.n_cells_own, c.n_faces, c.has_cell_faces(), c.cell_faces_offsets, c.cell_faces},
          geometry{g.n_faces, g.has_face_cell_dist_vector(), g.has_face_interp_weights(),
                   g.face_cell_dist_x, g.face_cell_dist_y, g.face_cell_dist_z, g.face_interp_weight}, options(o) {}

    KOKKOS_INLINE_FUNCTION std::size_t sample(LocalIndex cell, LocalIndex face) const
    {
        const std::size_t f = static_cast<std::size_t>(face);
        if (face >= mesh.sizes.n_inner_faces)
            return static_cast<std::size_t>(mesh.sizes.n_cells) + static_cast<std::size_t>(face - mesh.sizes.n_inner_faces);
        return static_cast<std::size_t>(mesh.connectivity.face_owner(f) == cell
            ? mesh.connectivity.face_neigh(f) : mesh.connectivity.face_owner(f));
    }

    template<class Real>
    KOKKOS_INLINE_FUNCTION void displacement(LocalIndex cell, LocalIndex face, Real d[3]) const
    {
        const std::size_t i = static_cast<std::size_t>(cell), f = static_cast<std::size_t>(face);
        const bool boundary = face >= mesh.sizes.n_inner_faces;
        if (geometry.has_face_cell_dist_vector()) {
            const Real sign = mesh.connectivity.face_owner(f) == cell ? Real{1} : Real{-1};
            d[0] = sign * static_cast<Real>(geometry.face_cell_dist_x(f));
            d[1] = sign * static_cast<Real>(geometry.face_cell_dist_y(f));
            d[2] = sign * static_cast<Real>(geometry.face_cell_dist_z(f));
        } else if (boundary) {
            d[0] = (static_cast<Real>(mesh.geometry.face_centroid_x(f)) - static_cast<Real>(mesh.geometry.cell_centroid_x(i)));
            d[1] = (static_cast<Real>(mesh.geometry.face_centroid_y(f)) - static_cast<Real>(mesh.geometry.cell_centroid_y(i)));
            d[2] = (static_cast<Real>(mesh.geometry.face_centroid_z(f)) - static_cast<Real>(mesh.geometry.cell_centroid_z(i)));
        } else {
            const auto j = sample(cell, face);
            d[0] = static_cast<Real>(mesh.geometry.cell_centroid_x(j)) - static_cast<Real>(mesh.geometry.cell_centroid_x(i));
            d[1] = static_cast<Real>(mesh.geometry.cell_centroid_y(j)) - static_cast<Real>(mesh.geometry.cell_centroid_y(i));
            d[2] = static_cast<Real>(mesh.geometry.cell_centroid_z(j)) - static_cast<Real>(mesh.geometry.cell_centroid_z(i));
        }
    }

    template<class Real>
    KOKKOS_INLINE_FUNCTION Real owner_weight(std::size_t face) const
    {
        return options.interpolation == Interpolation::AuxiliaryWeights
            ? static_cast<Real>(geometry.face_interp_weight(face)) : Real{0.5};
    }
};

// Relative Cholesky pivot checks reject singular 3-D stencils instead of returning zero gradients.
template<class Real>
KOKKOS_INLINE_FUNCTION bool inverse_spd(const Real a[9], Real inverse[9], Real tolerance)
{
    Real l[9]{};
    const Real threshold = tolerance * (a[0] + a[4] + a[8]);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col <= row; ++col) {
            Real v = a[3 * row + col];
            for (int k = 0; k < col; ++k) v -= l[3 * row + k] * l[3 * col + k];
            if (row == col) {
                if (!Kokkos::isfinite(v) || v <= threshold) return false;
                l[3 * row + col] = Kokkos::sqrt(v);
            } else l[3 * row + col] = v / l[3 * col + col];
        }
    }
    for (int col = 0; col < 3; ++col) {
        Real y[3]{}, x[3]{};
        for (int row = 0; row < 3; ++row) {
            Real v = row == col ? Real{1} : Real{0};
            for (int k = 0; k < row; ++k) v -= l[3 * row + k] * y[k];
            y[row] = v / l[3 * row + row];
        }
        for (int row = 2; row >= 0; --row) {
            Real v = y[row];
            for (int k = row + 1; k < 3; ++k) v -= l[3 * k + row] * x[k];
            x[row] = v / l[3 * row + row];
            inverse[3 * row + col] = x[row];
        }
    }
    return true;
}

template<class Space, class Coefficients, class Real>
struct PrepareLeastSquares {
    Metrics<Space> metrics;
    Coefficients coefficients;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell, std::size_t& bad) const
    {
        const std::size_t i = static_cast<std::size_t>(cell);
        const auto& conn = metrics.connectivity;
        const auto begin = conn.cell_faces_offsets(i), end = conn.cell_faces_offsets(i + 1);
        Real scale = 0;
        for (auto k = begin; k < end; ++k) {
            Real d[3]; metrics.displacement(cell, conn.cell_faces(k), d);
            for (int axis = 0; axis < 3; ++axis) {
                if (!Kokkos::isfinite(d[axis])) { ++bad; return; }
                const Real a = Kokkos::abs(d[axis]);
                if (a > scale) scale = a;
            }
        }
        if (!(scale > 0)) { ++bad; return; }
        Real a[9]{};
        for (auto k = begin; k < end; ++k) {
            Real d[3]; metrics.displacement(cell, conn.cell_faces(k), d);
            for (auto& value : d) value /= scale;
            const Real r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            if (!(r2 > 0)) { ++bad; return; }
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) a[3*row+col] += d[row]*d[col]/r2;
        }
        Real inverse[9];
        if (!inverse_spd(a, inverse, static_cast<Real>(metrics.options.pivot_tolerance))) { ++bad; return; }
        // A second pass avoids fixed-size neighbour buffers and their degree limits.
        for (auto k = begin; k < end; ++k) {
            Real d[3]; metrics.displacement(cell, conn.cell_faces(k), d);
            for (auto& value : d) value /= scale;
            const Real r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
            for (int axis = 0; axis < 3; ++axis) {
                const Real value = (inverse[3*axis]*d[0] + inverse[3*axis+1]*d[1] + inverse[3*axis+2]*d[2]) / r2 / scale;
                using C = typename Coefficients::non_const_value_type;
                const C stored = static_cast<C>(value);
                if (!Kokkos::isfinite(stored)) { ++bad; return; }
                coefficients(k, axis) = stored;
            }
        }
    }
};

template<bool LeastSquares, class Space, class Coefficients, class T, std::size_t N, class Real>
struct CellKernel {
    Metrics<Space> metrics;
    Coefficients coefficients;
    Batch<T, N> fields;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const std::size_t i = static_cast<std::size_t>(cell);
        Real cell_vol_inv = Real{1};
        if constexpr (!LeastSquares) {
            cell_vol_inv = Real{1} / static_cast<Real>(metrics.mesh.geometry.cell_volume(i));
        }
        Real gx[N]{}, gy[N]{}, gz[N]{}, self[N];
        for (std::size_t c = 0; c < N; ++c) self[c] = static_cast<Real>(fields.input[c](i));
        const auto& conn = metrics.connectivity;
        for (auto k = conn.cell_faces_offsets(i); k < conn.cell_faces_offsets(i + 1); ++k) {
            const auto face = conn.cell_faces(k);
            const std::size_t f = static_cast<std::size_t>(face);
            const auto sample = metrics.sample(cell, face);
            Real x, y, z;
            if constexpr (LeastSquares) {
                x = static_cast<Real>(coefficients(k, 0));
                y = static_cast<Real>(coefficients(k, 1));
                z = static_cast<Real>(coefficients(k, 2));
            } else {
                const bool owner = metrics.mesh.connectivity.face_owner(f) == cell;
                const Real alpha = face >= metrics.mesh.sizes.n_inner_faces
                    ? Real{1} : (owner ? Real{1} - metrics.template owner_weight<Real>(f) : metrics.template owner_weight<Real>(f));
                const Real factor = (owner ? alpha : -alpha) * static_cast<Real>(metrics.mesh.geometry.face_area(f)) * cell_vol_inv;
                x = factor * static_cast<Real>(metrics.mesh.geometry.face_normal_x(f));
                y = factor * static_cast<Real>(metrics.mesh.geometry.face_normal_y(f));
                z = factor * static_cast<Real>(metrics.mesh.geometry.face_normal_z(f));
            }
            for (std::size_t c = 0; c < N; ++c) {
                const Real delta = static_cast<Real>(fields.input[c](sample)) - self[c];
                gx[c] += x*delta; gy[c] += y*delta; gz[c] += z*delta;
            }
        }
        for (std::size_t c = 0; c < N; ++c) {
            fields.x[c](i) = static_cast<T>(gx[c]);
            fields.y[c](i) = static_cast<T>(gy[c]);
            fields.z[c](i) = static_cast<T>(gz[c]);
        }
    }
};

template<class T, std::size_t N>
struct ClearKernel {
    Batch<T, N> fields;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const std::size_t i = static_cast<std::size_t>(cell);
        for (std::size_t c = 0; c < N; ++c) fields.x[c](i) = fields.y[c](i) = fields.z[c](i) = T{};
    }
};

template<class Space, class T, std::size_t N, class Real>
struct FaceKernel {
    Metrics<Space> metrics;
    Batch<T, N> fields;
    CellRange range;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const std::size_t f = static_cast<std::size_t>(face);
        const auto owner = metrics.mesh.connectivity.face_owner(f);
        const auto neigh = metrics.mesh.connectivity.face_neigh(f);
        const bool boundary = face >= metrics.mesh.sizes.n_inner_faces;

        const bool write_owner = owner >= range.begin && owner < range.end;
        const bool write_neigh = !boundary && neigh >= range.begin && neigh < range.end;

        if (!write_owner && !write_neigh) return;

        const std::size_t i = static_cast<std::size_t>(owner);
        const auto j = metrics.sample(owner, face);

        const Real w = boundary ? Real{0} : metrics.template owner_weight<Real>(f);
        const Real alpha = boundary ? Real{1} : Real{1} - w;

        // Read volumes only for owned cells whose gradients are updated.
        const Real owner_factor = write_owner ? alpha / static_cast<Real>(metrics.mesh.geometry.cell_volume(i)) : Real{0};
        const Real neigh_factor = write_neigh ? w / static_cast<Real>(metrics.mesh.geometry.cell_volume(j)) : Real{0};

        const Real area = static_cast<Real>(metrics.mesh.geometry.face_area(f));
        const Real sx = area * static_cast<Real>(metrics.mesh.geometry.face_normal_x(f));
        const Real sy = area * static_cast<Real>(metrics.mesh.geometry.face_normal_y(f));
        const Real sz = area * static_cast<Real>(metrics.mesh.geometry.face_normal_z(f));

        for (std::size_t c = 0; c < N; ++c) {
            const Real delta = static_cast<Real>(fields.input[c](j)) - static_cast<Real>(fields.input[c](i));

            if (write_owner) {
                const Real v = owner_factor * delta;
                Kokkos::atomic_add(&fields.x[c](i), static_cast<T>(sx * v));
                Kokkos::atomic_add(&fields.y[c](i), static_cast<T>(sy * v));
                Kokkos::atomic_add(&fields.z[c](i), static_cast<T>(sz * v));
            }

            if (write_neigh) {
                const Real v = neigh_factor * delta;
                Kokkos::atomic_add(&fields.x[c](j), static_cast<T>(sx * v));
                Kokkos::atomic_add(&fields.y[c](j), static_cast<T>(sy * v));
                Kokkos::atomic_add(&fields.z[c](j), static_cast<T>(sz * v));
            }
        }
    }
};

} // namespace detail

} // namespace krona::numerics::gradient