#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/error.hpp>
#include <krona/fields/fields_view.hpp>
#include <krona/mesh/compute/detail/array_transfer.hpp>
#include <krona/mesh/compute/validation.hpp>
#include <krona/numerics/gradient/detail/kernels.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace krona::numerics::gradient {

/// @brief Immutable static-mesh gradient plan; only owned output cells are written.
/// Construction validates and prepares locally, then fences exec. Apply calls are asynchronous.
/// Borrowed mesh/auxiliary buffers and this plan must outlive all submitted work.
template<class Exec = execution::ExecutionSpace, class Coefficient = GeometryReal, class Accumulator = AccumulationReal>
class GradientOperator {
    static_assert(std::is_floating_point_v<Coefficient> && std::is_floating_point_v<Accumulator>);
    using memory_space = typename Exec::memory_space;
    using mesh_view_type = mesh::compute::MeshView<memory_space>;
    using connectivity_view_type = mesh::compute::AuxConnectivityView<memory_space>;
    using geometry_view_type = mesh::compute::AuxGeometryView<memory_space>;
    using coefficients_type = Kokkos::View<Coefficient*[3], Kokkos::LayoutLeft, Kokkos::Device<Exec, memory_space>>;
    using policy_type = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;

public:
    /// @brief Uses already attached mesh and optional caches; never builds topology or performs MPI.
    GradientOperator(mesh_view_type mesh_view, connectivity_view_type connectivity, 
                     geometry_view_type geometry, Options options, Exec exec = {})
        : metrics_{mesh_view, connectivity, geometry, options}, exec_(exec)
    {
        validate_descriptors();
        execution::complete(exec, [&](){
            validate_mesh(mesh_view, connectivity, geometry);
            std::size_t bad = 0;
            const bool least_squares = options.scheme == Scheme::LeastSquaresFace;
            if (least_squares) {
                const auto count = connectivity.cell_faces.extent(0);
                mpi::require(count <= std::numeric_limits<std::size_t>::max() / (3 * sizeof(Coefficient)),
                             "Gradient: coefficient allocation overflows size_t");
                coefficients_ = coefficients_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "gradient/ls_coefficients"), count);
                Kokkos::parallel_reduce("gradient/prepare_least_squares", policy_type(exec_, 0, mesh_view.sizes.n_own),
                    detail::PrepareLeastSquares<memory_space, coefficients_type, double>{metrics_, coefficients_}, bad);
                mpi::require(bad == 0, "Gradient: singular/ill-conditioned 3-D LS stencil or invalid sample displacement");
            }
        });
    }

    GradientOperator(const GradientOperator&) = delete;
    GradientOperator& operator=(const GradientOperator&) = delete;
    GradientOperator(GradientOperator&&) = delete;
    GradientOperator& operator=(GradientOperator&&) = delete;

    /// @brief Computes all input components; output component 3*c+axis is d(input_c)/d(axis).
    template<class T, int N, class InputSpace, class InputLayout, class U, class OutputSpace, class OutputLayout>
    void apply(fields::FieldView<T, N, InputSpace, InputLayout> input,
               fields::FieldView<U, 3*N, OutputSpace, OutputLayout> output, CellRange range = {}) const
    {
        using Value = std::remove_const_t<T>;
        static_assert(std::is_same_v<Value, U> && std::is_floating_point_v<Value>);
        static_assert(Kokkos::SpaceAccessibility<Exec, InputSpace>::accessible &&
                      Kokkos::SpaceAccessibility<Exec, OutputSpace>::accessible);
        detail::Batch<Value, static_cast<std::size_t>(N)> batch;
        for (int c = 0; c < N; ++c) {
            const std::size_t index = static_cast<std::size_t>(c);
            batch.input[index] = input_component<Value>(input.component(c));
            batch.x[index] = output_component<Value>(output.component(3*c));
            batch.y[index] = output_component<Value>(output.component(3*c+1));
            batch.z[index] = output_component<Value>(output.component(3*c+2));
        }
        submit(batch, resolve(range));
    }

    /// @brief Computes one scalar component into three rank-one Views, including strided subviews.
    template<class Input, class OutputX, class OutputY, class OutputZ>
    void apply_component(Input input, OutputX x, OutputY y, OutputZ z, CellRange range = {}) const
    {
        apply_set(std::array{input}, std::array{x}, std::array{y}, std::array{z}, range);
    }

    /// @brief Fuses a compile-time batch of unrelated scalar components without device pointer tables.
    template<class Input, class OutputX, class OutputY, class OutputZ, std::size_t N>
    void apply_set(const std::array<Input, N>& input, const std::array<OutputX, N>& x,
                   const std::array<OutputY, N>& y, const std::array<OutputZ, N>& z, CellRange range = {}) const
    {
        static_assert(N > 0);
        using T = std::remove_const_t<typename Input::value_type>;
        static_assert(std::is_floating_point_v<T>);
        detail::Batch<T, N> batch;
        for (std::size_t c = 0; c < N; ++c) {
            batch.input[c] = input_component<T>(input[c]);
            batch.x[c] = output_component<T>(x[c]);
            batch.y[c] = output_component<T>(y[c]);
            batch.z[c] = output_component<T>(z[c]);
        }
        submit(batch, resolve(range));
    }
    
    [[nodiscard]] Scheme scheme() const noexcept { return metrics_.options.scheme; }
    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    [[nodiscard]] std::size_t required_input_size() const noexcept { return input_size_; }
    [[nodiscard]] std::size_t coefficient_bytes() const noexcept { return coefficients_.span() * sizeof(Coefficient); }
    [[nodiscard]] const char* name() const noexcept
    {
        switch (scheme()) {
            case Scheme::GreenGaussCell: return "GREEN_GAUSS_CELL";
            case Scheme::GreenGaussFace: return "GREEN_GAUSS_FACE";
            case Scheme::LeastSquaresFace: return "LEAST_SQUARES_FACE";
        }
        return "INVALID";
    }

private:
    detail::Metrics<memory_space> metrics_;
    Exec exec_;
    coefficients_type coefficients_;
    std::size_t input_size_ = 0;

    template<mesh::compute::MeshData Data, mesh::AuxConnType ConnectivityMask, mesh::AuxGeomType GeometryMask>
    void validate_selected(mesh_view_type mesh_view, connectivity_view_type connectivity, geometry_view_type geometry) const
    {
        namespace validation = mesh::compute::validation;
        // Compare checks CSR incidence here; CellGeometry is checked separately on owned cells.
        const auto result = validation::validate<Data, ConnectivityMask, GeometryMask, true>(
            exec_, mesh_view, connectivity, geometry);
        mpi::require(result.shapes == 0, "Gradient: invalid selected mesh/auxiliary descriptors");
        mpi::require(result.faces == 0, "Gradient: invalid face topology or metrics");
        mpi::require(result.cell_faces == 0 && result.cell_incidence == 0 && result.face_incidence == 0,
                     "Gradient: invalid cell-to-face CSR or incidence");
    }

    void validate_mesh(mesh_view_type mesh_view, connectivity_view_type connectivity, geometry_view_type geometry) const
    {
        using MeshData = mesh::compute::MeshData;
        using mesh::AuxConnType;
        using mesh::AuxGeomType;
        namespace validation = mesh::compute::validation;
        if (scheme() == Scheme::LeastSquaresFace) {
            if (geometry.has_face_cell_dist_vector())
                validate_selected<MeshData::FaceCells, AuxConnType::CellFaces, AuxGeomType::FaceCellDistanceVector>(mesh_view, connectivity, geometry);
            else
                validate_selected<MeshData::FaceCells, AuxConnType::CellFaces, AuxGeomType::None>(mesh_view, connectivity, geometry);
            return;
        }
        const auto green_gauss = [&]<AuxConnType ConnectivityMask>() {
            constexpr auto data = MeshData::FaceCells | MeshData::FaceGeometry;
            if (metrics_.options.interpolation == Interpolation::AuxiliaryWeights)
                validate_selected<data, ConnectivityMask, AuxGeomType::FaceInterpolationWeights>(mesh_view, connectivity, geometry);
            else
                validate_selected<data, ConnectivityMask, AuxGeomType::None>(mesh_view, connectivity, geometry);
        };
        if (scheme() == Scheme::GreenGaussCell) green_gauss.template operator()<AuxConnType::CellFaces>();
        else green_gauss.template operator()<AuxConnType::None>();

        // Ghost volumes are not used; preserve the owned-only output contract.
        std::size_t bad = 0;
        Kokkos::parallel_reduce("gradient/validate_owned_geometry", policy_type(exec_, 0, mesh_view.sizes.n_own),
            validation::ValidateCells<MeshData::CellGeometry, AuxGeomType::None, memory_space>{mesh_view, {}}, bad);
        mpi::require(bad == 0, "Gradient: owned centres must be finite and volumes positive");
    }

    template<class View>
    static void accessible_component()
    {
        static_assert(Kokkos::is_view<View>::value && View::rank == 1);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);
    }

    template<class T, class View>
    detail::Component<const T> input_component(View view) const
    {
        accessible_component<View>();
        static_assert(std::is_same_v<std::remove_const_t<typename View::value_type>, T>);
        mpi::require(view.extent(0) >= input_size_, "Gradient: input must include owned, MPI ghost and boundary values");
        mpi::require(view.extent(0) == 0 || (view.data() != nullptr && view.stride(0) > 0), "Gradient: invalid input storage");
        return {view.data(), view.stride(0)};
    }

    template<class T, class View>
    detail::Component<T> output_component(View view) const
    {
        accessible_component<View>();
        static_assert(std::is_same_v<typename View::value_type, T>, "Gradient output must be mutable and match input precision");
        mpi::require(view.extent(0) >= static_cast<std::size_t>(metrics_.mesh.sizes.n_own), "Gradient: output must contain all owned cells");
        mpi::require(view.extent(0) == 0 || (view.data() != nullptr && view.stride(0) > 0), "Gradient: invalid output storage");
        return {view.data(), view.stride(0)};
    }

    CellRange resolve(CellRange range) const
    {
        if (range.end == -1) range.end = metrics_.mesh.sizes.n_own;
        mpi::require(range.begin >= 0 && range.begin <= range.end && range.end <= metrics_.mesh.sizes.n_own,
                     "Gradient: range must be an owned-cell interval");
        return range;
    }

    template<class T, std::size_t N>
    void submit(detail::Batch<T, N> batch, CellRange range) const
    {
        if (range.begin == range.end) return;
        const policy_type policy(exec_, range.begin, range.end);
        switch (scheme()) {
            case Scheme::GreenGaussCell:
                Kokkos::parallel_for("gradient/green_gauss_cell", policy,
                    detail::CellKernel<false, memory_space, coefficients_type, T, N, Accumulator>{metrics_, coefficients_, batch});
                break;
            case Scheme::LeastSquaresFace:
                Kokkos::parallel_for("gradient/least_squares_face", policy,
                    detail::CellKernel<true, memory_space, coefficients_type, T, N, Accumulator>{metrics_, coefficients_, batch});
                break;
            case Scheme::GreenGaussFace:
                Kokkos::parallel_for("gradient/clear_owned", policy, detail::ClearKernel<T, N>{batch});
                Kokkos::parallel_for("gradient/green_gauss_face", policy_type(exec_, 0, metrics_.mesh.sizes.n_faces),
                    detail::FaceKernel<memory_space, T, N, Accumulator>{metrics_, batch, range});
                break;
        }
    }

    void validate_descriptors()
    {
        execution::require_kokkos();
        const auto& m = metrics_.mesh;
        const auto& s = m.sizes;
        const auto& options = metrics_.options;
        const auto needed = gradient::requirements(options);
        mpi::require(std::isfinite(options.pivot_tolerance) && options.pivot_tolerance > 0 && options.pivot_tolerance < 1,
                "Gradient: pivot tolerance must be finite and between zero and one");
        mpi::require(s.faces_grouped && s.n_own >= 0 && s.n_cells >= s.n_own && s.n_internal_faces >= 0 &&
                s.n_internal_faces <= s.n_inner_faces && s.n_inner_faces <= s.n_faces,
                "Gradient: requires valid counts and grouped faces");
        mpi::require(m.has(needed.mesh),
                "Gradient: requires face connectivity and cell/face geometry");
        const auto cells = static_cast<std::size_t>(s.n_cells), faces = static_cast<std::size_t>(s.n_faces);
        const auto boundary = static_cast<std::size_t>(s.n_faces - s.n_inner_faces);
        mpi::require(cells <= std::numeric_limits<std::size_t>::max() - boundary, "Gradient: extended size overflows size_t");
        input_size_ = cells + boundary;
        mpi::require(m.connectivity.face_owner.extent(0) == faces && m.connectivity.face_neigh.extent(0) == faces,
                "Gradient: face connectivity size mismatch");
        mpi::require(m.geometry.cell_centroid_x.extent(0) >= cells && m.geometry.cell_centroid_y.extent(0) >= cells &&
                m.geometry.cell_centroid_z.extent(0) >= cells && m.geometry.cell_volume.extent(0) >= static_cast<std::size_t>(s.n_own),
                "Gradient: cell geometry size mismatch");
        mpi::require(m.geometry.face_centroid_x.extent(0) == faces && m.geometry.face_centroid_y.extent(0) == faces &&
                m.geometry.face_centroid_z.extent(0) == faces && m.geometry.face_area.extent(0) == faces &&
                m.geometry.face_normal_x.extent(0) == faces && m.geometry.face_normal_y.extent(0) == faces &&
                m.geometry.face_normal_z.extent(0) == faces, "Gradient: face geometry size mismatch");
        if (needed.connectivity != mesh::AuxConnType::None) {
            const auto& conn = metrics_.connectivity;
            mpi::require(conn.has_cell_faces() && conn.n_cells_own == s.n_own && conn.n_faces == s.n_faces &&
                    conn.cell_faces_offsets.extent(0) == static_cast<std::size_t>(s.n_own) + 1,
                    "Gradient: attach ready owned-cell-to-face CSR");
            mpi::require(s.n_own != 0 || conn.cell_faces.extent(0) == 0, "Gradient: empty partition has nonempty CSR");
        }
        const auto& geometry = metrics_.geometry;
        if (geometry.has_face_cell_dist_vector())
            mpi::require(geometry.n_faces == s.n_faces && geometry.face_cell_dist_x.extent(0) == faces &&
                    geometry.face_cell_dist_y.extent(0) == faces && geometry.face_cell_dist_z.extent(0) == faces,
                    "Gradient: auxiliary displacement size mismatch");
        if (needed.geometry != mesh::AuxGeomType::None)
            mpi::require(geometry.has_face_interp_weights() && geometry.n_faces == s.n_faces && geometry.face_interp_weight.extent(0) == faces,
                    "Gradient: attach ready owner interpolation weights");
    }
};

} // namespace krona::numerics::gradient
