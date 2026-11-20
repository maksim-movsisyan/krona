#pragma once

#include <krona/transport/scalar/status.hpp>

namespace krona::transport::scalar::detail {

/// @brief Integrates one cell's source and adds it to that cell's output entry.
/// @tparam Real Floating-point type used to multiply source density by cell volume.
/// @tparam Volumes View of cell volumes indexed by local cell ID.
/// @tparam Output Writable view of RHS or residual entries.
/// @tparam Source Device-callable source taking a local cell index.
/// @tparam Flags Writable device view of assembly flags.
template<class Real, class Volumes, class Output, class Source, class Flags>
struct AddSource {
    Volumes volumes;
    Output output;
    Source source;
    Flags flags;

    /// @brief Adds the source contribution for the owned cell handled by this thread.
    /// @param i Local index of the owned cell.
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
        const Real value = static_cast<Real>(source(i));
        if (!Kokkos::isfinite(value)) {
            flag(flags, BadSource);
            return;
        }
        const Real volume = static_cast<Real>(volumes(i));
        if (!Kokkos::isfinite(volume) || !(volume > Real{0})) {
            flag(flags, BadSystem);
            return;
        }
        // Convert the per-volume source to a cell total. Each thread owns one output entry.
        add_checked(output, i, value * volume, flags);
    }
};

} // namespace krona::transport::scalar::detail
