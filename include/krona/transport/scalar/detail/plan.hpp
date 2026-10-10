#pragma once

#include <krona/boundary/scalar/conditions.hpp>
#include <krona/linalg/mesh_adapter.hpp>
#include <krona/mesh/compute/mesh_view.hpp>
#include <krona/transport/scalar/source.hpp>

#include <Kokkos_NumericTraits.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <source_location>
#include <string>
#include <utility>

namespace krona::transport::scalar::detail {
\
inline void require(bool condition, const char* message,
                    std::source_location where = std::source_location::current()) noexcept
{ if (!condition) [[unlikely]] mpi::fatal(message, where); }

template<class Function>
decltype(auto) guarded(Function&& function)
{
    try { return std::forward<Function>(function)(); }
    catch (const std::exception& error) { mpi::fatal(error.what()); }
    catch (...) { mpi::fatal("Scalar diffusion: unknown exception"); }
}

template<class Space, class Real>
struct RowPlan {
    Kokkos::View<const std::size_t*, Space> offsets, entries, diagonal;
    Kokkos::View<const LocalIndex*, Space> faces;
    Kokkos::View<const Real*, Space> area_over_distance;
};

template<class Space, class Real, class Metric>
struct ValidateGeometry {
    mesh::compute::MeshView<Space> mesh;
    Metric metric;
    bool require_orthogonal;
    Real tolerance;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, int& bad) const
    {
        const auto f = static_cast<std::size_t>(face);
        const auto i = static_cast<std::size_t>(mesh.connectivity.face_owner(f));
        const auto& g = mesh.geometry;
        Real dx, dy, dz;
        if (face < mesh.sizes.n_inner_faces) {
            const auto j = static_cast<std::size_t>(mesh.connectivity.face_neigh(f));
            dx = static_cast<Real>(g.cell_centroid_x(j)) - static_cast<Real>(g.cell_centroid_x(i));
            dy = static_cast<Real>(g.cell_centroid_y(j)) - static_cast<Real>(g.cell_centroid_y(i));
            dz = static_cast<Real>(g.cell_centroid_z(j)) - static_cast<Real>(g.cell_centroid_z(i));
        } else {
            dx = static_cast<Real>(g.face_centroid_x(f)) - static_cast<Real>(g.cell_centroid_x(i));
            dy = static_cast<Real>(g.face_centroid_y(f)) - static_cast<Real>(g.cell_centroid_y(i));
            dz = static_cast<Real>(g.face_centroid_z(f)) - static_cast<Real>(g.cell_centroid_z(i));
        }
        const Real nx = static_cast<Real>(g.face_normal_x(f));
        const Real ny = static_cast<Real>(g.face_normal_y(f));
        const Real nz = static_cast<Real>(g.face_normal_z(f));
        const Real area = static_cast<Real>(g.face_area(f));
        const Real h = dx*nx + dy*ny + dz*nz;
        const Real normal2 = nx*nx + ny*ny + nz*nz;
        const Real normal_tolerance = Real{128} *
            (static_cast<Real>(Kokkos::epsilon_v<GeometryReal>) + Kokkos::epsilon_v<Real>);
        bool valid = Kokkos::isfinite(dx) && Kokkos::isfinite(dy) && Kokkos::isfinite(dz) &&
            Kokkos::isfinite(area) && area > Real{0} && Kokkos::isfinite(h) && h > Real{0} &&
            Kokkos::isfinite(normal2) && Kokkos::abs(normal2-Real{1}) <= normal_tolerance;
        if (valid && require_orthogonal) {
            const Real tx = dx-h*nx, ty = dy-h*ny, tz = dz-h*nz;
            valid = Kokkos::sqrt(tx*tx+ty*ty+tz*tz) <= tolerance*h;
        }
        if (!valid) { ++bad; return; }
        if (face < mesh.sizes.n_inner_faces) {
            const Real factor = area/h;
            if (!Kokkos::isfinite(factor) || !(factor > Real{0})) { ++bad; return; }
            metric(f) = factor;
        }
    }
};

template<class Space, class Real>
struct ValidateCells {
    mesh::compute::MeshView<Space> mesh;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell, int& bad) const
    {
        const auto i = static_cast<std::size_t>(cell);
        const auto& g = mesh.geometry;
        if (!Kokkos::isfinite(static_cast<Real>(g.cell_centroid_x(i))) ||
            !Kokkos::isfinite(static_cast<Real>(g.cell_centroid_y(i))) ||
            !Kokkos::isfinite(static_cast<Real>(g.cell_centroid_z(i)))) ++bad;
        if (cell < mesh.sizes.n_own) {
            const Real volume = static_cast<Real>(g.cell_volume(i));
            if (!Kokkos::isfinite(volume) || !(volume > Real{0})) ++bad;
        }
    }
};

template<class Exec, class T>
Kokkos::View<const T*, typename Exec::memory_space>
upload(const std::vector<T>& input, const char* name, const Exec& exec)
{
    Kokkos::View<T*, typename Exec::memory_space> device(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string{name}), input.size());
    auto host = Kokkos::create_mirror_view(device);
    for (std::size_t i = 0; i < input.size(); ++i) host(i) = input[i];
    Kokkos::deep_copy(exec, device, host);
    // Retain the temporary upload source until its copy completes.
    exec.fence("Scalar diffusion: plan upload complete");
    return device;
}

template<class Exec, class Real>
struct Prepared {
    linalg::CsrPattern<Exec> pattern;
    RowPlan<typename Exec::memory_space, Real> rows;
};

/// @brief One-time mesh-to-algebra translation; repeated assembly uses native entry IDs directly.
template<class Exec, class Real>
Prepared<Exec, Real> prepare(const mesh::MeshPart& host,
    mesh::compute::MeshView<typename Exec::memory_space> mesh,
    MPI_Comm comm, const Exec& exec, DiffusionOptions options)
{
    return guarded([&] {
        require(Kokkos::is_initialized() && !Kokkos::is_finalized(), "Scalar diffusion requires active Kokkos");
        require(std::isfinite(options.orthogonality_tolerance) && options.orthogonality_tolerance >= 0 &&
                options.orthogonality_tolerance < 1, "Scalar diffusion: invalid orthogonality tolerance");
        require(host.n_own >= 0 && host.n_cells >= host.n_own && host.faces_grouped &&
            mesh.sizes.faces_grouped && host.n_internal_faces >= 0 &&
            host.n_inner_faces >= host.n_internal_faces && host.n_faces >= host.n_inner_faces &&
            mesh.sizes.n_own == host.n_own && mesh.sizes.n_cells == host.n_cells &&
            mesh.sizes.n_faces == host.n_faces && mesh.sizes.n_internal_faces == host.n_internal_faces &&
            mesh.sizes.n_inner_faces == host.n_inner_faces, "Scalar diffusion: grouped host/view mesh mismatch");
        require(mesh.has(mesh::compute::MeshData::FaceCells | mesh::compute::MeshData::Geometry),
                "Scalar diffusion requires face cells and cell/face geometry");
        const std::size_t n = static_cast<std::size_t>(host.n_own);
        const std::size_t cells = static_cast<std::size_t>(host.n_cells);
        const std::size_t nf = static_cast<std::size_t>(host.n_faces);
        require(mesh.connectivity.face_owner.extent(0) == nf && mesh.connectivity.face_neigh.extent(0) == nf &&
            mesh.geometry.cell_centroid_x.extent(0) == cells && mesh.geometry.cell_centroid_y.extent(0) == cells &&
            mesh.geometry.cell_centroid_z.extent(0) == cells && mesh.geometry.cell_volume.extent(0) >= n &&
            mesh.geometry.face_centroid_x.extent(0) == nf && mesh.geometry.face_centroid_y.extent(0) == nf &&
            mesh.geometry.face_centroid_z.extent(0) == nf && mesh.geometry.face_normal_x.extent(0) == nf &&
            mesh.geometry.face_normal_y.extent(0) == nf && mesh.geometry.face_normal_z.extent(0) == nf &&
            mesh.geometry.face_area.extent(0) == nf, "Scalar diffusion: invalid mesh array extents");

        const auto numbering = linalg::build_cell_numbering(host, comm);
        const auto input = linalg::make_face_csr(host, numbering);
        for (std::size_t f = 0; f < nf; ++f) {
            const auto neighbour = host.face_neigh[f];
            const bool valid = f < static_cast<std::size_t>(host.n_internal_faces) ?
                neighbour >= 0 && neighbour < host.n_own :
                (f < static_cast<std::size_t>(host.n_inner_faces) ? neighbour >= host.n_own : neighbour == -1);
            require(valid, "Scalar diffusion: face endpoints disagree with grouped intervals");
        }
        linalg::Distribution<Exec> distribution(numbering.owned_gids(), comm);
        linalg::CsrPattern<Exec> pattern(distribution, input, exec);
        const auto permutation = pattern.host_input_to_entry();
        const auto find_entry = [&](std::size_t row, linalg::GlobalOrdinal gid) {
            const auto begin = input.column_gids.begin()+static_cast<std::ptrdiff_t>(input.row_offsets[row]);
            const auto end = input.column_gids.begin()+static_cast<std::ptrdiff_t>(input.row_offsets[row+1]);
            const auto found = std::lower_bound(begin, end, gid);
            require(found != end && *found == gid, "Scalar diffusion: missing matrix entry");
            return permutation[static_cast<std::size_t>(found-input.column_gids.begin())];
        };
        std::vector<std::size_t> offsets(n+1, 0), diagonal(n);
        for (LocalIndex f = 0; f < host.n_inner_faces; ++f) {
            const std::size_t u = static_cast<std::size_t>(f);
            ++offsets[static_cast<std::size_t>(host.face_owner[u])+1];
            if (host.face_neigh[u] < host.n_own) ++offsets[static_cast<std::size_t>(host.face_neigh[u])+1];
        }
        for (std::size_t i = 0; i < n; ++i) {
            require(offsets[i+1] <= std::numeric_limits<std::size_t>::max()-offsets[i],
                    "Scalar diffusion: incidence size overflow");
            offsets[i+1] += offsets[i];
            diagonal[i] = find_entry(i, numbering.cell_to_global[i]);
        }
        std::vector<LocalIndex> faces(offsets.back());
        std::vector<std::size_t> entries(offsets.back()), cursor = offsets;
        for (LocalIndex f = 0; f < host.n_inner_faces; ++f) {
            const auto u = static_cast<std::size_t>(f);
            const auto i = static_cast<std::size_t>(host.face_owner[u]);
            const auto j = static_cast<std::size_t>(host.face_neigh[u]);
            const auto a = cursor[i]++;
            faces[a] = f; entries[a] = find_entry(i, numbering.cell_to_global[j]);
            if (j < n) {
                const auto b = cursor[j]++;
                faces[b] = f; entries[b] = find_entry(j, numbering.cell_to_global[i]);
            }
        }
        using Space = typename Exec::memory_space;
        RowPlan<Space, Real> plan;
        plan.offsets = upload(offsets, "diffusion/row_faces_offsets", exec);
        plan.faces = upload(faces, "diffusion/row_faces", exec);
        plan.entries = upload(entries, "diffusion/face_entries", exec);
        plan.diagonal = upload(diagonal, "diffusion/diagonal_entries", exec);
        Kokkos::View<Real*, Space> metric(Kokkos::view_alloc(Kokkos::WithoutInitializing,
            "diffusion/area_over_distance"), static_cast<std::size_t>(host.n_inner_faces));
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
        int bad_faces = 0, bad_cells = 0;
        Kokkos::parallel_reduce("diffusion/validate_faces", Policy(exec, 0, host.n_faces),
            ValidateGeometry<Space, Real, decltype(metric)>{mesh, metric, options.require_orthogonal,
                static_cast<Real>(options.orthogonality_tolerance)}, bad_faces);
        Kokkos::parallel_reduce("diffusion/validate_cells", Policy(exec, 0, host.n_cells),
            ValidateCells<Space, Real>{mesh}, bad_cells);
        exec.fence("Scalar diffusion: geometry preparation complete");
        require(bad_faces == 0, "Scalar diffusion: invalid or nonorthogonal face geometry");
        require(bad_cells == 0, "Scalar diffusion: invalid cell geometry");
        plan.area_over_distance = metric;
        return Prepared<Exec, Real>{pattern, plan};
    });
}

template<class Exec, class View>
void validate_view(View view, std::size_t count)
{
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);
    require(view.extent(0) >= count && (count == 0 || (view.data() != nullptr && view.stride(0) > 0)),
            "Scalar diffusion: field/coefficient/source extent or storage is invalid");
}

template<class Exec, class Source> void validate_source(const Source&, std::size_t) {}
template<class Exec, class View>
void validate_source(CellSource<View> source, std::size_t count) { validate_view<Exec>(source.values, count); }
template<class Exec, class Diffusivity> void validate_diffusion(const Diffusivity&, std::size_t) {}
template<class Exec, class View>
void validate_diffusion(boundary::scalar::FaceDiffusivity<View> diffusion, std::size_t count)
{ validate_view<Exec>(diffusion.values, count); }

} // namespace krona::transport::scalar::detail
