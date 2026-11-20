#pragma once

#include <krona/boundary/geometry_view.hpp>
#include <krona/boundary/scalar/conditions.hpp>
#include <krona/mesh/device/mesh_view.hpp>

namespace krona::transport::scalar::diffusion::detail {

/// @brief Sums source density times volume over the owned cells.
/// @tparam Space Kokkos memory space of the input and output views.
/// @tparam Source Device-callable source density taking a local cell index.
template<class Space, class Source>
struct IntegrateSource {
    mesh::device::MeshView<Space> mesh; Source source;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell, AccumulationReal& sum) const {
        const std::size_t i = static_cast<std::size_t>(cell);
        sum += static_cast<AccumulationReal>(source(i))*static_cast<AccumulationReal>(mesh.geometry.cell_volume(i));
    }
};

/// @brief Sums the integrated outward flux on a physical boundary patch.
/// @tparam Space Kokkos memory space of the input and output views.
/// @tparam Real Floating-point type used for the face fluxes.
/// @tparam Condition Scalar boundary condition applied to this patch.
/// @tparam Law Diffusion law containing the face coefficient provider.
/// @tparam Field Rank-one field view indexed by local cell, including needed ghost values.
/// @tparam Correction Policy supplying the gradient correction or NoCorrection.
template<class Space, class Real, class Condition, class Law, class Field, class Correction>
struct IntegrateBoundary {
    boundary::GeometryView<Space> geometry; Condition condition; Law law; Field field; Correction correction; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, AccumulationReal& sum) const {
        const auto c = geometry.template context<Real>(face, time);
        const auto r = boundary::scalar::make_relation(condition, c, static_cast<Real>(correction(c)), law);
        sum += static_cast<AccumulationReal>(c.area)*static_cast<AccumulationReal>(
            law.outward_flux(c, r.normal_gradient(static_cast<Real>(field(static_cast<std::size_t>(c.owner))))));
    }
};

} // namespace krona::transport::scalar::diffusion::detail
