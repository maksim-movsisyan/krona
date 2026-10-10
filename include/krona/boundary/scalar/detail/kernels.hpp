#pragma once

#include <krona/boundary/scalar/geometry_view.hpp>
#include <krona/boundary/detail/support.hpp>

#include <Kokkos_NumericTraits.hpp>

namespace krona::boundary::scalar::detail {

using krona::boundary::detail::guarded;
using krona::boundary::detail::require;

template<class Space, class Real>
struct ValidateGeometry {
    GeometryView<Space> geometry;
    LocalIndex n_own = 0;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, std::size_t& bad) const
    {
        const auto f = static_cast<std::size_t>(face);
        const auto owner = geometry.face_owner(f);
        if (owner < 0 || owner >= n_own) { ++bad; return; }
        const auto i = static_cast<std::size_t>(owner);
        const Real x = static_cast<Real>(geometry.face_x(f)), y = static_cast<Real>(geometry.face_y(f));
        const Real z = static_cast<Real>(geometry.face_z(f));
        const Real cx = static_cast<Real>(geometry.cell_x(i)), cy = static_cast<Real>(geometry.cell_y(i));
        const Real cz = static_cast<Real>(geometry.cell_z(i));
        const Real nx = static_cast<Real>(geometry.normal_x(f)), ny = static_cast<Real>(geometry.normal_y(f));
        const Real nz = static_cast<Real>(geometry.normal_z(f));
        const Real area = static_cast<Real>(geometry.face_area(f));
        const Real norm2 = nx*nx + ny*ny + nz*nz;
        constexpr Real tolerance = Real{128} * static_cast<Real>(Kokkos::epsilon_v<GeometryReal>) +
                                   Real{128} * Kokkos::epsilon_v<Real>;
        const Real h = (x-cx)*nx + (y-cy)*ny + (z-cz)*nz;
        if (!Kokkos::isfinite(x) || !Kokkos::isfinite(y) || !Kokkos::isfinite(z) ||
            !Kokkos::isfinite(cx) || !Kokkos::isfinite(cy) || !Kokkos::isfinite(cz) ||
            !Kokkos::isfinite(norm2) || Kokkos::abs(norm2-Real{1}) > tolerance ||
            !Kokkos::isfinite(h) || !(h > Real{0}) || !Kokkos::isfinite(area) || !(area > Real{0})) ++bad;
    }
};

template<class Space, class Real, class Field, class Condition, class Diffusivity, class Correction>
struct FillBoundary {
    GeometryView<Space> geometry;
    Field field;
    Condition condition;
    Diffusivity diffusion;
    Correction correction;
    Real time = 0;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto context = geometry.template context<Real>(face, diffusion, correction, time);
        const auto relation = condition(context);
        const Real owner_value = static_cast<Real>(field(static_cast<std::size_t>(context.owner)));
        field(geometry.extended_index(face)) = static_cast<typename Field::non_const_value_type>(
            relation.face_value(owner_value));
    }
};

template<class Space, class Real, class Sink, class Condition, class Diffusivity, class Correction>
struct EvaluateBoundary {
    GeometryView<Space> geometry;
    Sink sink;
    Condition condition;
    Diffusivity diffusion;
    Correction correction;
    Real time = 0;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto context = geometry.template context<Real>(face, diffusion, correction, time);
        sink(context, condition(context));
    }
};

} // namespace krona::boundary::scalar::detail
