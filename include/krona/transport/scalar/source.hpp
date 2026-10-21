#pragma once

#include <krona/mesh/compute/mesh_view.hpp>
#include <krona/transport/scalar/detail/views.hpp>
#include <krona/transport/scalar/status.hpp>

namespace krona::transport::scalar {

/// @brief Source density before volume integration; positive values add to b or to r=b-A*phi.
template<class Real = ComputeReal>
struct UniformSource {
    Real value = 0;
    KOKKOS_INLINE_FUNCTION Real operator()(std::size_t) const { return value; }
};
template<class View>
struct CellSource {
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    View values;
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell) const { return values(cell); }
};

namespace detail {
template<class Exec, class Source> void validate_source(const Source&, std::size_t) {}
template<class Exec, class View>
void validate_source(CellSource<View> source, std::size_t count) { validate_view<Exec>(source.values, count); }

template<class Real, class Volumes, class Output, class Source, class Flags>
struct AddSource {
    Volumes volumes; Output output; Source source; Flags flags;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    {
        const Real value = static_cast<Real>(source(i));
        if (!Kokkos::isfinite(value)) { flag(flags, BadSource); return; }
        const Real volume = static_cast<Real>(volumes(i));
        if (!Kokkos::isfinite(volume) || !(volume > Real{0})) { flag(flags, BadSystem); return; }
        add_checked(output, i, value*volume, flags);
    }
};
} // namespace detail

/// @brief Independent additive volume source. Borrows owned volumes, allocates no storage and submits no MPI.
template<class Exec = execution::ExecutionSpace, class Real = ComputeReal>
class SourceOperator {
    using Mesh = mesh::compute::MeshView<typename Exec::memory_space>;
    using Volumes = decltype(Mesh{}.geometry.cell_volume);
public:
    explicit SourceOperator(Mesh mesh, Exec exec = {})
        : volumes_(mesh.geometry.cell_volume), owned_(static_cast<std::size_t>(mesh.sizes.n_own)), exec_(exec)
    {
        execution::require_kokkos();
        mpi::require(mesh.sizes.n_own >= 0 && mesh.has(mesh::compute::MeshData::CellGeometry),
                     "Scalar source: attach owned cell volumes");
        detail::validate_view<Exec>(volumes_, owned_);
    }
    template<class Output, class Source = UniformSource<Real>>
    void add_async(Output output, AssemblyStatus<Exec>& status, Source source = {}) const
    {
        detail::validate_output<Exec>(output, owned_);
        detail::validate_source<Exec>(source, owned_);
        const auto flags = status.device_flags(exec_);
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        Kokkos::parallel_for("transport/add_source", Policy(exec_, 0, owned_),
            detail::AddSource<Real, Volumes, Output, Source, decltype(flags)>{volumes_, output, source, flags});
    }
private:
    Volumes volumes_;
    std::size_t owned_;
    Exec exec_;
};

} // namespace krona::transport::scalar
