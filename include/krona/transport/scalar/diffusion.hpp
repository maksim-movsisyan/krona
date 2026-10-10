#pragma once

#include <krona/boundary/scalar/manager.hpp>
#include <krona/linalg/csr_matrix.hpp>
#include <krona/transport/scalar/detail/assembly.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace krona::transport::scalar {

/// @brief Integrated FV operator for -div(D*grad(phi)) = s, with one owned scalar unknown per cell.
/// Construction and commit are collective; assemble_async submits only local Kokkos work.
template<class Scalar = SolverReal, class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class DiffusionOperator {
    static_assert(std::is_floating_point_v<Real>);
    using Space = typename Exec::memory_space;
    using Mesh = mesh::compute::MeshView<Space>;
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
    using Flags = Kokkos::View<int*, Space>;

public:
    DiffusionOperator(const mesh::MeshPart& host, Mesh mesh_view, MPI_Comm comm, Exec exec,
                      DiffusionOptions options = {})
        : exec_(exec), mesh_(mesh_view), comm_(comm),
          prepared_(detail::prepare<Exec, Real>(host, mesh_view, comm, exec, options)),
          matrix_(prepared_.pattern), rhs_(prepared_.pattern.distribution()),
          flags_("diffusion/status", detail::StatusCount), host_flags_(Kokkos::create_mirror(flags_)) {}

    DiffusionOperator(const DiffusionOperator&) = delete;
    DiffusionOperator& operator=(const DiffusionOperator&) = delete;
    DiffusionOperator(DiffusionOperator&&) = delete;
    DiffusionOperator& operator=(DiffusionOperator&&) = delete;
    ~DiffusionOperator() { if (pending_) exec_.fence("Scalar diffusion: drain pending assembly"); }

    /// @brief Overwrites coefficients and RHS in place; input providers must survive until commit().
    /// Source returns density by local cell ID; diffusivity returns D by absolute local face ID.
    template<class Boundary, class Source = UniformSource<Real>,
             class Diffusivity = boundary::scalar::UniformDiffusivity<Real>>
    void assemble_async(const Boundary& boundary, Source source = {}, Diffusivity diffusion = {}, Real time = 0)
    {
        detail::require(!pending_, "Scalar diffusion: assembly is pending; call commit()");
        validate_boundary(boundary);
        detail::require(std::isfinite(time), "Scalar diffusion: time must be finite");
        detail::validate_source<Exec>(source, static_cast<std::size_t>(mesh_.sizes.n_own));
        detail::validate_diffusion<Exec>(diffusion, static_cast<std::size_t>(mesh_.sizes.n_faces));
        detail::guarded([&] {
            auto values = matrix_.edit_values(exec_);
            auto rhs = rhs_.edit(exec_);
            pending_ = true; ready_ = false;
            Kokkos::deep_copy(exec_, values, Scalar{0});
            Kokkos::deep_copy(exec_, rhs, Scalar{0});
            Kokkos::deep_copy(exec_, flags_, 0);
            Kokkos::parallel_for("diffusion/assemble_rows", Policy(exec_, 0, mesh_.sizes.n_own),
                detail::AssembleRows<Space, Real, decltype(values), decltype(rhs), Source, Diffusivity, Flags>{
                    mesh_, prepared_.rows, values, rhs, source, diffusion, flags_});
            // Typed patch dispatch is on the host; every kernel uses this operator's execution instance.
            boundary.for_each_patch([&](auto range, const auto& condition) {
                using C = std::remove_cvref_t<decltype(condition)>;
                auto diagonal = prepared_.rows.diagonal;
                Kokkos::parallel_for("diffusion/assemble_boundary", Policy(exec_, range.begin, range.end),
                    detail::AssembleBoundary<Space, Real, C, Diffusivity, decltype(values), decltype(rhs), decltype(diagonal), Flags>{
                        boundary.geometry_view(), condition, diffusion, values, rhs, diagonal, flags_, time});
            });
            using EntryPolicy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
            Kokkos::parallel_for("diffusion/validate_system",
                EntryPolicy(exec_, 0, std::max(values.extent(0), rhs.extent(0))),
                detail::ValidateSystem<decltype(values), decltype(rhs), decltype(prepared_.rows.diagonal), Flags>{
                    values, rhs, prepared_.rows.diagonal, flags_});
        });
    }

    /// @brief Completes local assembly, checks data collectively and rejects an unanchored global problem.
    void commit()
    {
        detail::require(pending_, "Scalar diffusion: no pending assembly to commit");
        Kokkos::deep_copy(exec_, host_flags_, flags_);
        matrix_.commit(); rhs_.commit();
        pending_ = false;
        std::array<int, detail::StatusCount> global{};
        mpi::check(MPI_Allreduce(host_flags_.data(), global.data(), detail::StatusCount, MPI_INT, MPI_MAX, comm_),
                   "MPI_Allreduce(diffusion status)");
        detail::require(global[detail::BadSource] == 0, "Scalar diffusion: source density must be finite");
        detail::require(global[detail::BadDiffusion] == 0, "Scalar diffusion: face diffusivity must be finite and positive");
        detail::require(global[detail::BadBoundary] == 0, "Scalar diffusion: boundary flux relation must be finite with a nonnegative owner coefficient");
        detail::require(global[detail::BadSystem] == 0, "Scalar diffusion: coefficients are not representable or a row has no positive diagonal");
        detail::require(prepared_.pattern.distribution().global_size() == 0 || global[detail::HasAnchor] != 0,
                        "Scalar diffusion: pure Neumann problem needs an explicit reference; not supported yet");
        ready_ = true;
    }

    /// @brief Blocking collective diagnostic; use the same source, BC parameters and D as assembly.
    /// Reports the published field's balance, including storage-precision rounding.
    template<class Component, class Boundary, class Source = UniformSource<Real>,
             class Diffusivity = boundary::scalar::UniformDiffusivity<Real>>
    [[nodiscard]] BalanceResult balance(Component field, const Boundary& boundary, Source source = {},
                                        Diffusivity diffusion = {}, Real time = 0) const
    {
        require_ready(); validate_boundary(boundary);
        detail::require(std::isfinite(time), "Scalar diffusion: time must be finite");
        detail::validate_view<Exec>(field, static_cast<std::size_t>(mesh_.sizes.n_own));
        detail::validate_source<Exec>(source, static_cast<std::size_t>(mesh_.sizes.n_own));
        detail::validate_diffusion<Exec>(diffusion, static_cast<std::size_t>(mesh_.sizes.n_faces));
        AccumulationReal source_sum = 0, boundary_sum = 0;
        Kokkos::parallel_reduce("diffusion/integrate_source", Policy(exec_, 0, mesh_.sizes.n_own),
            detail::IntegrateSource<Space, Source>{mesh_, source}, source_sum);
        boundary.for_each_patch([&](auto range, const auto& condition) {
            using C = std::remove_cvref_t<decltype(condition)>;
            AccumulationReal patch_sum = 0;
            Kokkos::parallel_reduce("diffusion/integrate_boundary", Policy(exec_, range.begin, range.end),
                detail::IntegrateBoundary<Space, Real, C, Diffusivity, Component>{
                    boundary.geometry_view(), condition, diffusion, field, time}, patch_sum);
            boundary_sum += patch_sum;
        });
        exec_.fence("Scalar diffusion: balance integration complete");
        const std::array<double, 2> local{static_cast<double>(source_sum), static_cast<double>(boundary_sum)};
        std::array<double, 2> global{};
        mpi::check(MPI_Allreduce(local.data(), global.data(), 2, MPI_DOUBLE, MPI_SUM, comm_),
                   "MPI_Allreduce(diffusion balance)");
        detail::require(std::isfinite(global[0]) && std::isfinite(global[1]), "Scalar diffusion: nonfinite balance");
        return {global[0], global[1], global[1]-global[0]};
    }

    [[nodiscard]] const linalg::CsrMatrix<Scalar, Exec>& matrix() const noexcept { return matrix_; }
    [[nodiscard]] const linalg::Vector<Scalar, Exec>& rhs() const noexcept { return rhs_; }
    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    [[nodiscard]] bool pending() const noexcept { return pending_; }
    void require_ready() const { detail::require(ready_ && !pending_, "Scalar diffusion: assemble and commit before use"); }

private:
    Exec exec_;
    Mesh mesh_;
    MPI_Comm comm_;
    detail::Prepared<Exec, Real> prepared_;
    linalg::CsrMatrix<Scalar, Exec> matrix_;
    linalg::Vector<Scalar, Exec> rhs_;
    Flags flags_;
    Kokkos::View<int*, Kokkos::HostSpace> host_flags_;
    bool pending_ = false, ready_ = false;

    template<class Boundary>
    void validate_boundary(const Boundary& boundary) const
    {
        const auto b = boundary.geometry_view();
        const auto& g = mesh_.geometry;
        detail::require(b.n_cells == mesh_.sizes.n_cells && b.n_inner_faces == mesh_.sizes.n_inner_faces &&
            b.face_owner.data() == mesh_.connectivity.face_owner.data() &&
            b.cell_x.data() == g.cell_centroid_x.data() && b.cell_y.data() == g.cell_centroid_y.data() &&
            b.cell_z.data() == g.cell_centroid_z.data() && b.face_x.data() == g.face_centroid_x.data() &&
            b.face_y.data() == g.face_centroid_y.data() && b.face_z.data() == g.face_centroid_z.data() &&
            b.normal_x.data() == g.face_normal_x.data() && b.normal_y.data() == g.face_normal_y.data() &&
            b.normal_z.data() == g.face_normal_z.data() && b.face_area.data() == g.face_area.data(),
            "Scalar diffusion: boundary manager must use the same compute mesh");
    }
};

} // namespace krona::transport::scalar
