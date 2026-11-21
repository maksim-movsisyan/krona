#pragma once

#include <krona/transport/scalar/diffusion/detail/face_flux.hpp>
#include <krona/transport/scalar/diffusion/types.hpp>
#include <krona/boundary/geometry_view.hpp>

namespace krona::transport::scalar::diffusion::detail {

/// @brief Adds inner-face matrix entries and optionally the explicit correction for one owned row.
/// @tparam AddRhs Include the explicit nonorthogonal correction in the RHS.
/// @tparam Space Kokkos memory space of the input and output views.
/// @tparam Real Floating-point type used for the face fluxes.
/// @tparam Target System slice with native entries and writable matrix/RHS views.
/// @tparam Law Diffusion law containing the face coefficient provider.
/// @tparam Correction Policy supplying the gradient correction or NoCorrection.
/// @tparam Flags Writable device view of assembly diagnostics.
template<bool AddRhs, class Space, class Real, class Target, class Law, class Correction, class Flags>
struct AddRows {
    StencilData<Space> data; Target target; Law law; Correction correction; Flags flags; Real time;
    FaceRange range;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const {
        const std::size_t i = static_cast<std::size_t>(cell);
        using Value = typename decltype(target.values)::non_const_value_type;
        Real diagonal = 0, explicit_rhs = 0;
        for (auto k = data.offsets(i); k < data.offsets(i+1); ++k) {
            const auto face = data.faces(k);
            if (face < range.begin || face >= range.end) continue;
            const auto flux = interior_flux<Real>(data, face, law, correction, flags, time);
            const Real a = flux.conductance;
            if (!(a > Real{0})) continue;
            if (!Kokkos::isfinite(static_cast<Value>(a)) || !(static_cast<Value>(a) > Value{0})) {
                flag(flags, BadSystem); continue;
            }
            const std::size_t f = static_cast<std::size_t>(face);
            const bool owner = data.face_owner(f) == cell;
            const auto entry = owner ? target.entries.owner_entries(f) : target.entries.neighbour_entries(f);
            // The shared conductance adds to the diagonal and subtracts from the opposite cell entry.
            add_checked(target.values, entry, -a, flags);
            diagonal += a;
            if constexpr (AddRhs && corrected<Correction>)
                explicit_rhs += owner ? -flux.constant : flux.constant;
        }
        add_checked(target.values, target.entries.diagonal(i), diagonal, flags);
        if constexpr (AddRhs && corrected<Correction>)
            add_checked(target.rhs, i, explicit_rhs, flags);
    }
};

/// @brief Adds a boundary face contribution to the owner diagonal and optionally to the RHS.
/// @tparam AddRhs Include the known boundary flux term in the RHS.
/// @tparam Space Kokkos memory space of the input and output views.
/// @tparam Real Floating-point type used for the face fluxes.
/// @tparam Target System slice with native entries and writable matrix/RHS views.
/// @tparam Condition Scalar boundary condition applied to this patch.
/// @tparam Law Diffusion law containing the face coefficient provider.
/// @tparam Correction Policy supplying the gradient correction or NoCorrection.
/// @tparam Flags Writable device view of assembly diagnostics.
template<bool AddRhs, class Space, class Real, class Target, class Condition, class Law, class Correction, class Flags>
struct AddBoundary {
    boundary::GeometryView<Space> geometry; Target target; Condition condition; Law law;
    Correction correction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const {
        const auto c = geometry.template context<Real>(face, time);
        const auto flux = boundary_flux(c, condition, law, correction, flags);
        using Value = typename decltype(target.values)::non_const_value_type;
        if (flux.owner > Real{0} && !(static_cast<Value>(flux.owner) > Value{0})) {
            flag(flags, BadSystem); return;
        }
        const std::size_t i = static_cast<std::size_t>(c.owner);
        atomic_add_checked(target.values, target.entries.diagonal(i), flux.owner, flags);
        if constexpr (AddRhs)
            atomic_add_checked(target.rhs, i, -flux.constant, flags);
    }
};

} // namespace krona::transport::scalar::diffusion::detail
