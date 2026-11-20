#pragma once

#include <krona/transport/scalar/diffusion/detail/face_flux.hpp>
#include <krona/transport/scalar/diffusion/types.hpp>
#include <krona/boundary/geometry_view.hpp>

namespace krona::transport::scalar::diffusion::detail {

/// @brief Adds -A_interior*phi and its explicit correction without a graph or face work buffer.
/// @tparam Space Kokkos memory space of the input and output views.
/// @tparam Real Floating-point type used for the face fluxes.
/// @tparam Field Rank-one field view indexed by local cell, including needed ghost values.
/// @tparam Output Writable rank-one output view indexed by owned cell.
/// @tparam Law Diffusion law containing the face coefficient provider.
/// @tparam Correction Policy supplying the gradient correction or NoCorrection.
/// @tparam Flags Writable device view of assembly diagnostics.
template<class Space, class Real, class Field, class Output, class Law, class Correction, class Flags>
struct ResidualRows {
    StencilData<Space> data; Field field; Output output; Law law; Correction correction; Flags flags; Real time;
    FaceRange range;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const {
        const std::size_t i = static_cast<std::size_t>(cell);
        const Real own = static_cast<Real>(field(i));
        if (!Kokkos::isfinite(own)) { flag(flags, BadSystem); return; }
        Real residual = 0;
        for (auto k = data.offsets(i); k < data.offsets(i+1); ++k) {
            const auto face = data.faces(k);
            // Filter before loading the neighbour, since its ghost value may still be in flight.
            if (face < range.begin || face >= range.end) continue;
            const std::size_t f = static_cast<std::size_t>(face);
            const bool owner = data.face_owner(f) == cell;
            const auto other = owner ? data.face_neigh(f) : data.face_owner(f);
            const Real value = static_cast<Real>(field(static_cast<std::size_t>(other)));
            const auto flux = interior_flux<Real>(data, face, law, correction, flags, time);
            residual += flux.conductance*(value-own);
            if constexpr (corrected<Correction>) residual += owner ? -flux.constant : flux.constant;
        }
        add_checked(output, i, residual, flags);
    }
};

/// @brief Adds minus the outward boundary flux to the owner residual.
/// @tparam Space Kokkos memory space of the input and output views.
/// @tparam Real Floating-point type used for the face fluxes.
/// @tparam Field Rank-one field view indexed by local cell, including needed ghost values.
/// @tparam Output Writable rank-one output view indexed by owned cell.
/// @tparam Condition Scalar boundary condition applied to this patch.
/// @tparam Law Diffusion law containing the face coefficient provider.
/// @tparam Correction Policy supplying the gradient correction or NoCorrection.
/// @tparam Flags Writable device view of assembly diagnostics.
template<class Space, class Real, class Field, class Output, class Condition, class Law, class Correction, class Flags>
struct ResidualBoundary {
    boundary::GeometryView<Space> geometry; Field field; Output output;
    Condition condition; Law law; Correction correction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const {
        const auto c = geometry.template context<Real>(face, time);
        const auto flux = boundary_flux(c, condition, law, correction, flags);
        const std::size_t i = static_cast<std::size_t>(c.owner);
        const Real value = flux.owner == Real{0} ? flux.constant :
            flux.owner*static_cast<Real>(field(i))+flux.constant;
        atomic_add_checked(output, i, -value, flags);
    }
};

} // namespace krona::transport::scalar::diffusion::detail
