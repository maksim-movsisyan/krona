#pragma once

#include <krona/mesh/device/mesh_view.hpp>
#include <krona/transport/scalar/source/providers.hpp>
#include <krona/transport/scalar/source/detail/kernels.hpp>

namespace krona::transport::scalar {

/// @brief Adds the volume integral of a scalar source to an RHS or residual.
/// The operator borrows the mesh volumes. Each submission updates only owned cells.
/// @tparam Exec Kokkos execution space that runs the source kernel.
/// @tparam Real Floating-point type used to compute the integrated source.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class SourceOperator {
    using Mesh = mesh::device::MeshView<typename Exec::memory_space>;
    using Volumes = decltype(Mesh{}.geometry.cell_volume);

public:
    /// @brief Borrows the cell volumes from a mesh with attached cell geometry.
    /// @param mesh Mesh with one volume per owned cell.
    /// @param exec Execution instance shared with the assembly status.
    explicit SourceOperator(Mesh mesh, Exec exec = {})
        : volumes_(mesh.geometry.cell_volume), owned_(static_cast<std::size_t>(mesh.sizes.n_own)), exec_(exec) {
        execution::require_kokkos();
        mpi::require(mesh.sizes.n_own >= 0 && mesh.has(mesh::device::MeshData::CellGeometry),
                     "Scalar source: attach owned cell volumes");
        detail::validate_view<Exec>(volumes_, owned_);
    }

    /// @brief Queues output[cell] += source(cell) * volume[cell] on the operator's execution instance.
    /// Keep the mesh, output and provider data available until the batch completes.
    /// @tparam Output Writable rank-one floating-point view indexed by local cell ID.
    /// @tparam Source Device-callable provider of source density. Defaults to a constant zero source.
    /// @param output RHS or residual with space for every owned cell. Existing entries are incremented.
    /// @param status Open diagnostic batch on the same execution instance.
    /// @param source Source density to evaluate in each owned cell.
    template<class Output, class Source = UniformSource<Real>>
    void add_async(Output output, AssemblyStatus<Exec>& status, Source source = {}) const {
        detail::validate_output<Exec>(output, owned_);
        detail::validate_source<Exec>(source, owned_);
        const auto flags = status.device_flags(exec_);
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        mpi::guarded([&] {
            Kokkos::parallel_for("transport/add_source", Policy(exec_, 0, owned_),
                detail::AddSource<Real, Volumes, Output, Source, decltype(flags)>{volumes_, output, source, flags});
        });
    }

private:
    Volumes volumes_;
    std::size_t owned_;
    Exec exec_;
};

} // namespace krona::transport::scalar
