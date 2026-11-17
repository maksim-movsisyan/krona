#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/error.hpp>
#include <krona/fields/fields_view.hpp>
#include <krona/numerics/gradient/detail/kernels.hpp>
#include <krona/numerics/gradient/detail/validation.hpp>

#include <array>
#include <limits>
#include <type_traits>

namespace krona::numerics::gradient {

/// @brief Prepares a scheme once on a fixed mesh and applies it to fields on owned cells.
/// @tparam Exec Kokkos execution space used to prepare and apply the scheme.
/// @tparam Coefficient Storage type of the precomputed least-squares weights.
/// @tparam Accumulator Arithmetic type for application. The face scheme converts each
/// contribution to the output type before adding it to the field.
/// Construction waits for local setup. Application queues kernels on execution_space().
/// Keep the operator, mesh arrays, auxiliary arrays and field values alive until the kernels finish.
template <class Exec = execution::ExecutionSpace, class Coefficient = GeometryReal,
          class Accumulator = AccumulationReal>
class GradientOperator {
    static_assert(std::is_floating_point_v<Coefficient> && std::is_floating_point_v<Accumulator>);
    using memory_space = typename Exec::memory_space;
    using mesh_view_type = mesh::device::MeshView<memory_space>;
    using connectivity_view_type = mesh::device::AuxConnectivityView<memory_space>;
    using geometry_view_type = mesh::device::AuxGeometryView<memory_space>;
    using coefficients_type =
        Kokkos::View<Coefficient* [3], Kokkos::LayoutLeft, Kokkos::Device<Exec, memory_space>>;
    using policy_type = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;

  public:
    /// @brief Checks the attached mesh and prepares any weights the selected scheme needs.
    /// @param mesh_view Mesh with grouped faces and the arrays returned by requirements(options).
    /// @param connectivity Auxiliary connectivity. The cell-based schemes need CellFaces.
    /// @param geometry Auxiliary metrics, such as owner weights or cached least-squares displacements.
    /// @param options Scheme, interpolation choice and least-squares pivot threshold.
    /// @param exec Instance used for setup and all later gradient kernels.
    GradientOperator(mesh_view_type mesh_view, connectivity_view_type connectivity,
                     geometry_view_type geometry, Options options, Exec exec = {})
        : stencil_{mesh_view, connectivity, geometry, options}, exec_(exec) {
        mpi::guarded([&] {
            input_size_ = detail::validate_descriptors(mesh_view, connectivity, geometry, options);
            execution::complete(exec, [&]() {
                detail::validate_mesh(exec_, mesh_view, connectivity, geometry, options);
                std::size_t bad = 0;
                const bool least_squares = options.scheme == Scheme::LeastSquaresFace;
                if (least_squares) {
                    // A cell may need a different weight from its neighbour for the same face.
                    // Store three weights per entry of the owned-cell-to-face table.
                    const auto count = connectivity.cell_faces.extent(0);
                    mpi::require(count <= std::numeric_limits<std::size_t>::max() / (3 * sizeof(Coefficient)),
                                 "Gradient: coefficient allocation overflows size_t");
                    coefficients_ = coefficients_type(
                        Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "gradient/ls_coefficients"),
                        count);
                    Kokkos::parallel_reduce(
                        "gradient/prepare_least_squares", policy_type(exec_, 0, mesh_view.sizes.n_own),
                        detail::PrepareLeastSquares<memory_space, coefficients_type, double>{stencil_,
                                                                                             coefficients_},
                        bad);
                    mpi::require(bad == 0,
                        "Gradient: singular/ill-conditioned 3-D LS stencil or invalid sample displacement");
                }
            });
        });
    }

    GradientOperator(const GradientOperator&) = delete;
    GradientOperator& operator=(const GradientOperator&) = delete;
    GradientOperator(GradientOperator&&) = delete;
    GradientOperator& operator=(GradientOperator&&) = delete;

    /// @brief Queues a gradient calculation for every input component.
    /// @tparam T Floating-point input scalar type, optionally const.
    /// @tparam N Number of input components.
    /// @tparam InputSpace Kokkos memory space holding the input field.
    /// @tparam InputLayout Kokkos layout of the input field.
    /// @tparam U Writable output scalar type matching T with const removed.
    /// @tparam OutputSpace Kokkos memory space holding the output field.
    /// @tparam OutputLayout Kokkos layout of the output field.
    /// @param input Values for owned cells, MPI ghosts and appended boundary-face slots.
    /// @param output Space for all owned cells. Component 3*c + axis is the derivative
    /// of input component c along x, y or z.
    /// @param range Owned-cell interval to write. The default selects all owned cells.
    /// Keep the output components disjoint from each other and from the input.
    template <class T, int N, class InputSpace, class InputLayout, class U, class OutputSpace,
              class OutputLayout>
    void apply_async(fields::FieldView<T, N, InputSpace, InputLayout> input,
                     fields::FieldView<U, 3 * N, OutputSpace, OutputLayout> output,
                     CellRange range = {}) const {
        using Value = std::remove_const_t<T>;
        static_assert(std::is_same_v<Value, U> && std::is_floating_point_v<Value>);
        static_assert(Kokkos::SpaceAccessibility<Exec, InputSpace>::accessible &&
                      Kokkos::SpaceAccessibility<Exec, OutputSpace>::accessible);
        detail::Batch<Value, static_cast<std::size_t>(N), memory_space> batch;
        for (int c = 0; c < N; ++c) {
            const std::size_t index = static_cast<std::size_t>(c);
            batch.input[index] = input_component<Value>(input.component(c));
            batch.x[index] = output_component<Value>(output.component(3 * c));
            batch.y[index] = output_component<Value>(output.component(3 * c + 1));
            batch.z[index] = output_component<Value>(output.component(3 * c + 2));
        }
        submit(batch, resolve(range));
    }

    /// @brief Queues one scalar gradient with separate views for its x, y and z derivatives.
    /// @tparam Input Rank-one Kokkos view of the scalar samples.
    /// @tparam OutputX Rank-one Kokkos view type for the writable x derivative.
    /// @tparam OutputY Rank-one Kokkos view type for the writable y derivative.
    /// @tparam OutputZ Rank-one Kokkos view type for the writable z derivative.
    /// @param input Samples in the same extended layout as apply_async().
    /// @param x,y,z Derivative views for all owned cells, matching the input's scalar type.
    /// @param range Owned-cell interval to write.
    template <class Input, class OutputX, class OutputY, class OutputZ>
    void apply_component_async(Input input, OutputX x, OutputY y, OutputZ z, CellRange range = {}) const {
        apply_set_async(std::array{input}, std::array{x}, std::array{y}, std::array{z}, range);
    }

    /// @brief Computes several scalar gradients together while visiting each stencil once.
    /// @tparam Input Rank-one Kokkos view type shared by the input components.
    /// @tparam OutputX Rank-one Kokkos view type shared by the x outputs.
    /// @tparam OutputY Rank-one Kokkos view type shared by the y outputs.
    /// @tparam OutputZ Rank-one Kokkos view type shared by the z outputs.
    /// @tparam N Number of scalar components in the batch.
    /// @param input Extended input views, one per scalar component.
    /// @param x,y,z Output arrays. Entry c in each array holds that axis of the gradient of input[c].
    /// @param range Owned-cell interval to write. All component views follow the sizes in apply_async().
    template <class Input, class OutputX, class OutputY, class OutputZ, std::size_t N>
    void apply_set_async(const std::array<Input, N>& input, const std::array<OutputX, N>& x,
                         const std::array<OutputY, N>& y, const std::array<OutputZ, N>& z,
                         CellRange range = {}) const {
        static_assert(N > 0);
        using T = std::remove_const_t<typename Input::value_type>;
        static_assert(std::is_floating_point_v<T>);
        detail::Batch<T, N, memory_space> batch;
        for (std::size_t c = 0; c < N; ++c) {
            batch.input[c] = input_component<T>(input[c]);
            batch.x[c] = output_component<T>(x[c]);
            batch.y[c] = output_component<T>(y[c]);
            batch.z[c] = output_component<T>(z[c]);
        }
        submit(batch, resolve(range));
    }

    [[nodiscard]] Scheme scheme() const noexcept { return stencil_.options.scheme; }
    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    /// @brief Returns the number of input slots, including MPI ghosts and boundary faces.
    [[nodiscard]] std::size_t required_input_size() const noexcept { return input_size_; }
    [[nodiscard]] std::size_t coefficient_bytes() const noexcept {
        return coefficients_.span() * sizeof(Coefficient);
    }
    [[nodiscard]] const char* name() const noexcept {
        switch (scheme()) {
        case Scheme::GreenGaussCell:
            return "GREEN_GAUSS_CELL";
        case Scheme::GreenGaussFace:
            return "GREEN_GAUSS_FACE";
        case Scheme::LeastSquaresFace:
            return "LEAST_SQUARES_FACE";
        }
        return "INVALID";
    }

  private:
    detail::StencilData<memory_space> stencil_;
    Exec exec_;
    coefficients_type coefficients_;
    std::size_t input_size_ = 0;

    template <class View> static void accessible_component() {
        static_assert(Kokkos::is_view<View>::value && View::rank == 1);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);
    }

    template <class T, class View>
    fields::ComponentDescriptor<const T, memory_space> input_component(View view) const {
        accessible_component<View>();
        static_assert(std::is_same_v<std::remove_const_t<typename View::value_type>, T>);
        mpi::require(view.extent(0) >= input_size_,
                     "Gradient: input must include owned, MPI ghost and boundary values");
        const auto descriptor = fields::descriptor_from_view(view);
        return {descriptor.data, descriptor.stride};
    }

    template <class T, class View>
    fields::ComponentDescriptor<T, memory_space> output_component(View view) const {
        accessible_component<View>();
        static_assert(std::is_same_v<typename View::value_type, T>,
                      "Gradient output must be mutable and match input precision");
        mpi::require(view.extent(0) >= static_cast<std::size_t>(stencil_.mesh.sizes.n_own),
                     "Gradient: output must contain all owned cells");
        const auto descriptor = fields::descriptor_from_view(view);
        return {descriptor.data, descriptor.stride};
    }

    CellRange resolve(CellRange range) const {
        if (range.end == -1)
            range.end = stencil_.mesh.sizes.n_own;
        mpi::require(range.begin >= 0 && range.begin <= range.end && range.end <= stencil_.mesh.sizes.n_own,
                     "Gradient: range must be an owned-cell interval");
        return range;
    }

    template <class T, std::size_t N>
    void submit(detail::Batch<T, N, memory_space> batch, CellRange range) const {
        if (range.begin == range.end)
            return;
        mpi::guarded([&] {
            const policy_type policy(exec_, range.begin, range.end);
            switch (scheme()) {
            case Scheme::GreenGaussCell:
                Kokkos::parallel_for(
                    "gradient/green_gauss_cell", policy,
                    detail::GreenGaussCellKernel<memory_space, T, N, Accumulator>{stencil_, batch});
                break;
            case Scheme::LeastSquaresFace:
                Kokkos::parallel_for(
                    "gradient/least_squares_face", policy,
                    detail::LeastSquaresFaceKernel<memory_space, coefficients_type, T, N, Accumulator>{
                        stencil_, coefficients_, batch});
                break;
            case Scheme::GreenGaussFace:
                // The face kernel adds to the output, so start the requested cells at zero.
                // Both launches use the same instance and execute in order.
                Kokkos::parallel_for("gradient/clear_owned", policy,
                                     detail::ClearKernel<T, N, memory_space>{batch});
                Kokkos::parallel_for(
                    "gradient/green_gauss_face", policy_type(exec_, 0, stencil_.mesh.sizes.n_faces),
                    detail::GreenGaussFaceKernel<memory_space, T, N, Accumulator>{stencil_, batch, range});
                break;
            }
        });
    }
};

} // namespace krona::numerics::gradient