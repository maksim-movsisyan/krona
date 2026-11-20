#pragma once

#include <krona/boundary/scalar/manager.hpp>
#include <krona/transport/scalar/diffusion/detail/assembly.hpp>
#include <krona/transport/scalar/diffusion/detail/validation.hpp>
#include <krona/transport/scalar/system_view.hpp>

#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

namespace krona::transport::scalar::diffusion {

/// @brief Assembles isotropic diffusion or evaluates its residual using a borrowed mesh stencil.
/// The residual follows r = b - A*phi, so each face contributes minus the outward flux.
/// Coefficients are sampled at faces and held fixed in the assembled matrix.
/// @tparam Exec Kokkos execution space used by the plan and its kernels.
/// @tparam Real Floating-point type used to evaluate fluxes before writing the output.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class DiffusionOperator {
    using Space = typename Exec::memory_space;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    using Flags = typename AssemblyStatus<Exec>::flags_type;
public:
    using execution_space = Exec;
    using memory_space = Space;
    using plan_type = DiffusionPlan<Exec, Real>;

    /// @brief Retains a copy of the validated plan's borrowed descriptors.
    explicit DiffusionOperator(plan_type plan) : plan_(std::move(plan)) {}

    /// @brief Adds A_diff and b_boundary into an initialized SystemView. No zeroing, fence, MPI, or commit.
    template<class Target, class Boundary, class Coefficient = UniformCoefficient<Real>>
    void add_system_async(Target target, const Boundary& boundary, AssemblyStatus<Exec>& status,
        IsotropicDiffusion<Coefficient> law = {}, Real time = 0) const
    { add_system_async(target, boundary, status, law, numerics::gradient::NoCorrection{}, time); }

    /// @brief Adds diffusion coefficients and boundary/correction contributions to an existing system.
    /// @tparam Target Writable system slice with a matching native entry map.
    /// @tparam Boundary Scalar boundary manager for the same mesh.
    /// @tparam Coefficient Face coefficient provider stored by the diffusion law.
    /// @tparam Correction Policy projecting cell gradients, or NoCorrection.
    /// @param target Matrix entries and RHS to increment.
    /// @param boundary Conditions for the physical faces.
    /// @param status Open diagnostic batch on the plan's execution instance.
    /// @param law Isotropic relation between the normal gradient and flux.
    /// @param correction Provider of the explicit corrections. Its gradients must include current ghosts.
    /// @param time Evaluation time passed to the face providers.
    template<class Target, class Boundary, class Coefficient, class Correction> requires (!std::is_arithmetic_v<Correction>)
    void add_system_async(Target target, const Boundary& boundary, AssemblyStatus<Exec>& status,
        IsotropicDiffusion<Coefficient> law, Correction correction, Real time = 0) const {
        mpi::guarded([&] {
            validate_inputs(boundary, law, correction, time);
            validate_system(target);
            const auto flags = status.device_flags(execution_space_instance());
            submit_inner_system(target, flags, all_inner_faces(), law, correction, time);
            submit_boundary_system(target, boundary, flags, law, correction, time);
        });
    }

    /// @brief Adds the matrix and RHS terms from a selected range of inner faces.
    /// @param faces Half-open local face interval. The default covers all inner faces.
    /// @param target Writable system slice with the full entry map.
    /// @param status Open batch using the plan's execution instance.
    /// @param law Coefficient provider for the selected faces.
    /// @param correction Optional gradient projection. All data read by the selected faces must be ready.
    /// @param time Evaluation time passed to the providers.
    template<class Target, class Coefficient = UniformCoefficient<Real>, class Correction = numerics::gradient::NoCorrection>
    void add_inner_system_async(Target target, AssemblyStatus<Exec>& status, FaceRange faces = {},
                                IsotropicDiffusion<Coefficient> law = {}, Correction correction = {}, Real time = 0) const {
        mpi::guarded([&] {
            faces = inner_range(faces);
            validate_providers(law, correction, time);
            validate_system(target);
            const auto flags = status.device_flags(execution_space_instance());
            if (faces.begin == faces.end) return;
            submit_inner_system(target, flags, faces, law, correction, time);
        });
    }

    /// @brief Adds the diagonal and RHS terms from the physical boundary patches.
    /// Submit once when composing a system from several inner ranges.
    template<class Target, class Boundary, class Coefficient = UniformCoefficient<Real>,
        class Correction = numerics::gradient::NoCorrection>
    void add_boundary_system_async(Target target, const Boundary& boundary, AssemblyStatus<Exec>& status,
        IsotropicDiffusion<Coefficient> law = {}, Correction correction = {}, Real time = 0) const {
        mpi::guarded([&] {
            validate_inputs(boundary, law, correction, time);
            validate_system(target);
            const auto flags = status.device_flags(execution_space_instance());
            submit_boundary_system(target, boundary, flags, law, correction, time);
        });
    }

    /// @brief Adds b_boundary-A_diff*phi to owned output slots; no extended boundary samples are read.
    /// Input real-cell ghosts must be current. Input/output must not overlap, including strided aliases.
    template<class Field, class Output, class Boundary, class Coefficient = UniformCoefficient<Real>>
    void add_residual_async(Field field, Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
        IsotropicDiffusion<Coefficient> law = {}, Real time = 0) const
    { add_residual_async(field, output, boundary, status, law, numerics::gradient::NoCorrection{}, time); }

    /// @brief Adds the corrected face fluxes to the owned residual without assembling a matrix.
    /// Keep the output disjoint from the input field and the correction's gradient inputs.
    /// @tparam Field Rank-one scalar field view covering owned and real ghost cells.
    /// @tparam Output Writable rank-one residual view covering owned cells.
    /// @tparam Boundary Scalar boundary manager for the same mesh.
    /// @tparam Coefficient Face coefficient provider stored by the diffusion law.
    /// @tparam Correction Policy projecting cell gradients, or NoCorrection.
    template<class Field, class Output, class Boundary, class Coefficient, class Correction>
        requires (!std::is_arithmetic_v<Correction>)
    void add_residual_async(Field field, Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
        IsotropicDiffusion<Coefficient> law, Correction correction, Real time = 0) const {
        mpi::guarded([&] {
            validate_inputs(boundary, law, correction, time);
            validate_residual(field, output, plan_.mesh_view().sizes.n_cells);
            const auto flags = status.device_flags(execution_space_instance());
            submit_inner_residual(field, output, flags, all_inner_faces(), law, correction, time);
            submit_boundary_residual(field, output, boundary, flags, law, correction, time);
        });
    }

    /// @brief Adds the negative outward flux from a selected range of inner faces.
    /// @param field Local cell view with space for real ghosts. Values read in the selected range must be ready.
    /// @param output Owned residual entries to increment. Keep them disjoint from the inputs.
    /// @param status Open batch using the plan's execution instance.
    /// @param faces Half-open local face interval. The default covers all inner faces.
    /// @param law Coefficient provider for the selected faces.
    /// @param correction Optional gradient projection. Provide current gradients for all cells it reads.
    /// @param time Evaluation time passed to the providers.
    template<class Field, class Output, class Coefficient = UniformCoefficient<Real>,
        class Correction = numerics::gradient::NoCorrection>
    void add_inner_residual_async(Field field, Output output, AssemblyStatus<Exec>& status, FaceRange faces = {},
        IsotropicDiffusion<Coefficient> law = {}, Correction correction = {}, Real time = 0) const {
        mpi::guarded([&] {
            faces = inner_range(faces);
            validate_providers(law, correction, time);
            validate_residual(field, output, plan_.mesh_view().sizes.n_cells);
            const auto flags = status.device_flags(execution_space_instance());
            if (faces.begin == faces.end) return;
            submit_inner_residual(field, output, flags, faces, law, correction, time);
        });
    }

    /// @brief Adds the negative outward flux from the physical boundary patches.
    /// The scalar field only needs to cover owned cells. Keep the output disjoint from the inputs.
    template<class Field, class Output, class Boundary, class Coefficient = UniformCoefficient<Real>,
        class Correction = numerics::gradient::NoCorrection>
    void add_boundary_residual_async(Field field, Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
        IsotropicDiffusion<Coefficient> law = {}, Correction correction = {}, Real time = 0) const {
        mpi::guarded([&] {
            validate_inputs(boundary, law, correction, time);
            validate_residual(field, output, plan_.mesh_view().sizes.n_own);
            const auto flags = status.device_flags(execution_space_instance());
            submit_boundary_residual(field, output, boundary, flags, law, correction, time);
        });
    }

    /// @brief Adds the full difference between corrected and base RHS to existing owned entries.
    /// Start from the frozen base RHS for each outer iteration. Keep the material law, conditions and time consistent
    /// with that base system. The boundary matrix slope must be independent of the correction.
    template<class Output, class Boundary, class Coefficient, class Correction>
    void add_correction_async(Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
        IsotropicDiffusion<Coefficient> law, Correction correction, Real time = 0) const {
        mpi::guarded([&] {
            validate_inputs(boundary, law, correction, time);
            scalar::detail::validate_output<Exec>(output, owned_rows());
            const auto flags = status.device_flags(execution_space_instance());
            submit_inner_correction(output, flags, all_inner_faces(), law, correction, time);
            submit_boundary_correction(output, boundary, flags, law, correction, time);
        });
    }

    /// @brief Adds the explicit RHS correction for a selected range of inner faces.
    /// @param output Owned RHS or residual entries to increment.
    /// @param status Open batch using the plan's execution instance.
    /// @param faces Half-open local face interval. Use {} to select all inner faces.
    /// @param law Coefficient provider matching the frozen base system.
    /// @param correction Gradient projection using current data for the selected faces.
    /// @param time Evaluation time passed to the providers.
    template<class Output, class Coefficient, class Correction>
    void add_inner_correction_async(Output output, AssemblyStatus<Exec>& status, FaceRange faces,
        IsotropicDiffusion<Coefficient> law, Correction correction, Real time = 0) const {
        mpi::guarded([&] {
            faces = inner_range(faces);
            validate_providers(law, correction, time);
            scalar::detail::validate_output<Exec>(output, owned_rows());
            const auto flags = status.device_flags(execution_space_instance());
            if (faces.begin == faces.end) return;
            submit_inner_correction(output, flags, faces, law, correction, time);
        });
    }

    /// @brief Adds the change in the physical boundary RHS due to the explicit correction.
    /// Submit once per RHS update. Keep the boundary matrix slope fixed.
    template<class Output, class Boundary, class Coefficient, class Correction>
    void add_boundary_correction_async(Output output, const Boundary& boundary, AssemblyStatus<Exec>& status,
        IsotropicDiffusion<Coefficient> law, Correction correction, Real time = 0) const {
        mpi::guarded([&] {
            validate_inputs(boundary, law, correction, time);
            scalar::detail::validate_output<Exec>(output, owned_rows());
            const auto flags = status.device_flags(execution_space_instance());
            submit_boundary_correction(output, boundary, flags, law, correction, time);
        });
    }

    /// @brief Optional blocking collective diagnostic of the supplied field; source is volume density.
    template<class Field, class Boundary, class Coefficient = UniformCoefficient<Real>, class Source = UniformSource<Real>>
    [[nodiscard]] BalanceResult balance(Field field, const Boundary& boundary,
        IsotropicDiffusion<Coefficient> law = {}, Source source = {}, Real time = 0) const
    { return balance(field, boundary, law, source, numerics::gradient::NoCorrection{}, time); }

    template<class Field, class Boundary, class Coefficient, class Source, class Correction>
        requires (!std::is_arithmetic_v<Correction>)
    [[nodiscard]] BalanceResult balance(Field field, const Boundary& boundary,
            IsotropicDiffusion<Coefficient> law, Source source, Correction correction, Real time = 0) const {
        return mpi::guarded([&]() -> BalanceResult {
            validate_inputs(boundary, law, correction, time);
            const auto mesh = plan_.mesh_view();
            const auto& exec = execution_space_instance();
            scalar::detail::validate_view<Exec>(field, static_cast<std::size_t>(mesh.sizes.n_own));
            scalar::detail::validate_source<Exec>(source, static_cast<std::size_t>(mesh.sizes.n_own));
            AccumulationReal source_sum = 0, boundary_sum = 0;
            Kokkos::parallel_reduce("diffusion/integrate_source", Policy(exec, 0, mesh.sizes.n_own),
                detail::IntegrateSource<Space, Source>{mesh, source}, source_sum);
            boundary.for_each_patch([&](auto range, const auto& condition) {
                using C = std::remove_cvref_t<decltype(condition)>;
                AccumulationReal sum = 0;
                Kokkos::parallel_reduce("diffusion/integrate_boundary", Policy(exec, range.begin, range.end),
                    detail::IntegrateBoundary<Space, Real, C, decltype(law), Field, Correction>{boundary.geometry_view(), condition, law, field, correction, time}, sum);
                boundary_sum += sum;
            });
            const std::array<double, 2> local{static_cast<double>(source_sum), static_cast<double>(boundary_sum)};
            std::array<double, 2> global{};
            mpi::check(MPI_Allreduce(local.data(), global.data(), 2, MPI_DOUBLE, MPI_SUM, plan_.communicator().native()),
                       "MPI_Allreduce(diffusion balance)");
            mpi::require(std::isfinite(global[0]) && std::isfinite(global[1]), "Scalar diffusion: nonfinite balance");
            return {global[0], global[1], global[1]-global[0]};
        });
    }
    [[nodiscard]] const plan_type& plan() const noexcept { return plan_; }
    [[nodiscard]] const Exec& execution_space_instance() const noexcept { return plan_.execution_space_instance(); }
private:
    plan_type plan_;
    auto stencil_data() const {
        return detail::StencilData<Space>{plan_.mesh_view(), plan_.connectivity_view(), plan_.geometry_view()};
    }
    std::size_t owned_rows() const {
        return static_cast<std::size_t>(plan_.mesh_view().sizes.n_own);
    }

    FaceRange all_inner_faces() const {
        return {0, plan_.mesh_view().sizes.n_inner_faces};
    }

    FaceRange inner_range(FaceRange range) const {
        const auto count = plan_.mesh_view().sizes.n_inner_faces;
        if (range.end == -1) range.end = count;
        mpi::require(range.begin >= 0 && range.end >= range.begin && range.end <= count,
                     "Scalar diffusion: invalid inner face range");
        return range;
    }

    template<class Target>
    void validate_system(const Target& target) const {
        const auto sizes = plan_.mesh_view().sizes;
        const std::size_t rows = static_cast<std::size_t>(sizes.n_own);
        scalar::detail::validate_output<Exec>(target.values, target.values.extent(0));
        scalar::detail::validate_output<Exec>(target.rhs, rows);
        mpi::require(target.rhs.extent(0) == rows && target.entries.diagonal.extent(0) == rows &&
            target.entries.owner_entries.extent(0) == static_cast<std::size_t>(sizes.n_inner_faces) &&
            target.entries.neighbour_entries.extent(0) == static_cast<std::size_t>(sizes.n_internal_faces),
            "Scalar diffusion: incompatible system slice/entry map");
    }

    template<class Field, class Output>
    void validate_residual(const Field& field, const Output& output, LocalIndex input_size) const {
        scalar::detail::validate_view<Exec>(field, static_cast<std::size_t>(input_size));
        const auto rows = owned_rows();
        scalar::detail::validate_output<Exec>(output, rows);
        mpi::require(rows == 0 || static_cast<const void*>(field.data()) != static_cast<const void*>(output.data()),
                     "Scalar diffusion: in-place residual is unsupported");
    }

    template<class Law, class Correction>
    void validate_providers(const Law& law, const Correction& correction, Real time) const {
        mpi::require(std::isfinite(time), "Scalar diffusion: time must be finite");
        detail::validate_coefficient(plan_, law.coefficient);
        detail::validate_correction(plan_, correction);
    }

    template<class Boundary, class Law, class Correction>
    void validate_inputs(const Boundary& boundary, const Law& law, const Correction& correction, Real time) const {
        detail::validate_boundary(plan_, boundary);
        validate_providers(law, correction, time);
    }

    template<class Target, class Law, class Correction>
    void submit_inner_system(const Target& target, const Flags& flags, FaceRange faces, const Law& law,
                             const Correction& correction, Real time) const {
        const auto& exec = execution_space_instance();
        Kokkos::parallel_for("diffusion/add_rows", Policy(exec, 0, plan_.mesh_view().sizes.n_own),
            detail::AddRows<Space, Real, Target, Law, Correction, Flags>{
                stencil_data(), target, law, correction, flags, time, faces});
    }

    template<class Target, class Boundary, class Law, class Correction>
    void submit_boundary_system(const Target& target, const Boundary& boundary, const Flags& flags,
                                const Law& law, const Correction& correction, Real time) const {
        const auto& exec = execution_space_instance();
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("diffusion/add_boundary", Policy(exec, range.begin, range.end),
                detail::AddBoundary<Space, Real, Target, C, Law, Correction, Flags>{
                    boundary.geometry_view(), target, condition, law, correction, flags, time});
        });
    }

    template<class Field, class Output, class Law, class Correction>
    void submit_inner_residual(const Field& field, const Output& output, const Flags& flags, FaceRange faces,
                               const Law& law, const Correction& correction, Real time) const {
        const auto& exec = execution_space_instance();
        Kokkos::parallel_for("diffusion/residual_rows", Policy(exec, 0, plan_.mesh_view().sizes.n_own),
            detail::ResidualRows<Space, Real, Field, Output, Law, Correction, Flags>{
                stencil_data(), field, output, law, correction, flags, time, faces});
    }

    template<class Field, class Output, class Boundary, class Law, class Correction>
    void submit_boundary_residual(const Field& field, const Output& output, const Boundary& boundary,
                                  const Flags& flags, const Law& law, const Correction& correction, Real time) const {
        const auto& exec = execution_space_instance();
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("diffusion/residual_boundary", Policy(exec, range.begin, range.end),
                detail::ResidualBoundary<Space, Real, Field, Output, C, Law, Correction, Flags>{
                    boundary.geometry_view(), field, output, condition, law, correction, flags, time});
        });
    }

    template<class Output, class Law, class Correction>
    void submit_inner_correction(const Output& output, const Flags& flags, FaceRange faces, const Law& law,
                                 const Correction& correction, Real time) const {
        if constexpr (detail::corrected<Correction>) {
            const auto& exec = execution_space_instance();
            Kokkos::parallel_for("diffusion/correction_rows", Policy(exec, 0, plan_.mesh_view().sizes.n_own),
                detail::CorrectionRows<Space, Real, Output, Law, Correction, Flags>{
                    stencil_data(), output, law, correction, flags, time, faces});
        }
    }

    template<class Output, class Boundary, class Law, class Correction>
    void submit_boundary_correction(const Output& output, const Boundary& boundary, const Flags& flags,
                                    const Law& law, const Correction& correction, Real time) const {
        if constexpr (detail::corrected<Correction>) {
            const auto& exec = execution_space_instance();
            boundary.for_each_patch([&](auto range, const auto& condition) {
                using C = std::remove_cvref_t<decltype(condition)>;
                Kokkos::parallel_for("diffusion/correction_boundary", Policy(exec, range.begin, range.end),
                    detail::CorrectionBoundary<Space, Real, Output, C, Law, Correction, Flags>{
                        boundary.geometry_view(), output, condition, law, correction, flags, time});
            });
        }
    }
};

} // namespace krona::transport::scalar::diffusion
