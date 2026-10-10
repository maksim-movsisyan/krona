#pragma once

#include <krona/halo/exchange.hpp>
#include <krona/linalg/solver.hpp>
#include <krona/transport/scalar/diffusion.hpp>

#include <limits>
#include <type_traits>
#include <utility>

namespace krona::transport::scalar {

struct SteadyDiffusionOptions {
    DiffusionOptions diffusion;
    linalg::SolverOptions linear;
    halo::Transfer transfer = halo::Transfer::HostStaged;
};

namespace detail {

/// @brief Private communication context isolates solver halo tags from application exchanges.
class SolverComm {
public:
    explicit SolverComm(MPI_Comm source)
    {
        require(source != MPI_COMM_NULL, "Scalar diffusion: null solver communicator");
        mpi::check(MPI_Comm_dup(source, &comm_), "MPI_Comm_dup(scalar diffusion)");
    }
    ~SolverComm() { mpi::check(MPI_Comm_free(&comm_), "MPI_Comm_free(scalar diffusion)"); }
    SolverComm(const SolverComm&) = delete;
    SolverComm& operator=(const SolverComm&) = delete;
    [[nodiscard]] MPI_Comm get() const noexcept { return comm_; }
private:
    MPI_Comm comm_ = MPI_COMM_NULL;
};

} // namespace detail

/// @brief Reusable blocking solve, followed by field publication, halo exchange and face-value filling.
/// The field is borrowed; its address, mesh arrays and MPI/Kokkos runtimes must remain valid.
template<class Storage = StorageReal, class Scalar = SolverReal,
         class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class SteadyDiffusion {
    using Space = typename Exec::memory_space;
    using Mesh = mesh::compute::MeshView<Space>;
    using Component = Kokkos::View<Storage*, Kokkos::LayoutStride, Space,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

public:
    /// @brief Borrows one writable rank-one component, including a strided vector-field subview.
    template<class View>
    SteadyDiffusion(const mesh::MeshPart& host, Mesh mesh_view, View component,
                    MPI_Comm comm, Exec exec, SteadyDiffusionOptions options = {})
        : comm_(comm), exec_(exec), field_(borrow(component)),
          operation_(host, mesh_view, comm_.get(), exec_, options.diffusion),
          solution_(operation_.matrix().pattern().distribution()),
          linear_(operation_.matrix(), std::move(options.linear)),
          halo_(mesh_view, {host.nb_ranks, host.send_offsets, host.recv_offsets}, comm_.get(), 0, exec_, options.transfer)
    {
        const auto cells = static_cast<std::size_t>(host.n_cells);
        const auto boundary_faces = static_cast<std::size_t>(host.n_faces-host.n_inner_faces);
        detail::require(boundary_faces <= std::numeric_limits<std::size_t>::max()-cells,
                        "Scalar diffusion: extended field size overflow");
        detail::validate_view<Exec>(field_, cells+boundary_faces);
        // The existing halo API accepts exactly n_cells; borrow that prefix without copying data.
        halo_.add_component(Kokkos::subview(field_, std::pair<std::size_t, std::size_t>{0, cells}));
        halo_.prepare();
    }
    SteadyDiffusion(const SteadyDiffusion&) = delete;
    SteadyDiffusion& operator=(const SteadyDiffusion&) = delete;
    SteadyDiffusion(SteadyDiffusion&&) = delete;
    SteadyDiffusion& operator=(SteadyDiffusion&&) = delete;

    /// @brief Collective solve using the current owned field as the initial guess.
    /// On nonconvergence the borrowed field is unchanged; the latest algebraic iterate remains accessible.
    template<class Boundary, class Source = UniformSource<Real>,
             class Diffusivity = boundary::scalar::UniformDiffusivity<Real>>
    [[nodiscard]] linalg::SolveResult solve(const Boundary& boundary, Source source = {},
                                          Diffusivity diffusion = {}, Real time = 0)
    {
        operation_.assemble_async(boundary, source, diffusion, time);
        solution_.copy_from_component(field_, exec_);
        operation_.commit(); solution_.commit();
        const auto result = linear_.solve(operation_.rhs(), solution_);
        if (!result.converged) return result;
        solution_.copy_to_component(field_, exec_);
        halo_.exchange();
        // Halo completion makes the field ready even if the boundary manager uses another instance.
        boundary.apply_component_async(field_, diffusion, boundary::scalar::NoCorrection{}, time);
        boundary.execution_space().fence("Scalar diffusion: published boundary values complete");
        return result;
    }

    /// @brief Optional collective diagnostic; no balance reduction is hidden in solve().
    template<class Boundary, class Source = UniformSource<Real>,
             class Diffusivity = boundary::scalar::UniformDiffusivity<Real>>
    [[nodiscard]] BalanceResult balance(const Boundary& boundary, Source source = {},
                                        Diffusivity diffusion = {}, Real time = 0) const
    { return operation_.balance(field_, boundary, source, diffusion, time); }

    [[nodiscard]] const DiffusionOperator<Scalar, Exec, Real>& operation() const noexcept { return operation_; }
    [[nodiscard]] const linalg::Vector<Scalar, Exec>& solution() const noexcept { return solution_; }
    [[nodiscard]] Component component() const noexcept { return field_; }

private:
    // Declared first, hence destroyed last: all Tpetra and halo users release their context first.
    detail::SolverComm comm_;
    Exec exec_;
    Component field_;
    DiffusionOperator<Scalar, Exec, Real> operation_;
    linalg::Vector<Scalar, Exec> solution_;
    linalg::Solver<Scalar, Exec> linear_;
    halo::Exchange<Storage, Exec> halo_;

    template<class View>
    static Component borrow(View view)
    {
        static_assert(Kokkos::is_view<View>::value && View::rank == 1);
        static_assert(std::is_same_v<typename View::value_type, Storage>, "Supply a writable component of the selected storage scalar");
        static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);
        return Component{view};
    }
};

} // namespace krona::transport::scalar
