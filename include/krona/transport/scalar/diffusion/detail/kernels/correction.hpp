#pragma once

#include <krona/transport/scalar/diffusion/detail/face_flux.hpp>
#include <krona/boundary/geometry_view.hpp>

namespace krona::transport::scalar::diffusion::detail {

/// @brief Adds only the explicit interior correction to an initialized owned RHS/residual prefix.
/// @tparam Space Kokkos memory space of the input and output views.
/// @tparam Real Floating-point type used for the face fluxes.
/// @tparam Output Writable rank-one output view indexed by owned cell.
/// @tparam Law Diffusion law containing the face coefficient provider.
/// @tparam Correction Policy supplying the gradient correction or NoCorrection.
/// @tparam Flags Writable device view of assembly diagnostics.
template<class Space, class Real, class Output, class Law, class Correction, class Flags>
struct CorrectionRows {
    StencilData<Space> data; Output output; Law law; Correction correction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const {
        Real value = 0;
        const std::size_t i = static_cast<std::size_t>(cell);
        for (auto k = data.offsets(i); k < data.offsets(i+1); ++k) {
            const auto face = data.faces(k);
            if (face >= data.n_inner_faces) continue;
            const auto flux = interior_flux<Real>(data, face, law, correction, flags, time);
            value += data.face_owner(static_cast<std::size_t>(face)) == cell ? -flux.constant : flux.constant;
        }
        add_checked(output, i, value, flags);
    }
};

/// @brief Adds the change in the boundary flux caused by the explicit correction.
/// @tparam Space Kokkos memory space of the input and output views.
/// @tparam Real Floating-point type used for the face fluxes.
/// @tparam Output Writable rank-one output view indexed by owned cell.
/// @tparam Condition Scalar boundary condition applied to this patch.
/// @tparam Law Diffusion law containing the face coefficient provider.
/// @tparam Correction Policy supplying the gradient correction or NoCorrection.
/// @tparam Flags Writable device view of assembly diagnostics.
template<class Space, class Real, class Output, class Condition, class Law, class Correction, class Flags>
struct CorrectionBoundary {
    boundary::GeometryView<Space> geometry; Output output; Condition condition; Law law;
    Correction correction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const {
        const auto c = geometry.template context<Real>(face, time);
        const auto flux = boundary_flux<true>(c, condition, law, correction, flags);
        atomic_add_checked(output, static_cast<std::size_t>(c.owner), -flux.constant, flags);
    }
};

} // namespace krona::transport::scalar::diffusion::detail
