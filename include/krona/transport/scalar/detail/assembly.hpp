#pragma once

#include <krona/boundary/scalar/geometry_view.hpp>
#include <krona/transport/scalar/detail/plan.hpp>

namespace krona::transport::scalar::detail {

enum Status : int { BadSource, BadDiffusion, BadBoundary, HasAnchor, BadSystem, StatusCount };

template<class Flags>
KOKKOS_INLINE_FUNCTION void flag(Flags flags, Status index)
{ Kokkos::atomic_exchange(&flags(static_cast<int>(index)), 1); }

/// @brief One thread owns one row; repeated faces to the same neighbour accumulate without atomics.
template<class Space, class Real, class Values, class Rhs, class Source, class Diffusivity, class Flags>
struct AssembleRows {
    mesh::compute::MeshView<Space> mesh;
    RowPlan<Space, Real> plan;
    Values values;
    Rhs rhs;
    Source source;
    Diffusivity diffusion;
    Flags flags;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const std::size_t i = static_cast<std::size_t>(cell);
        using Scalar = typename Values::non_const_value_type;
        const Real density = static_cast<Real>(source(i));
        if (!Kokkos::isfinite(density)) { flag(flags, BadSource); return; }
        rhs(i, 0) = static_cast<Scalar>(density*static_cast<Real>(mesh.geometry.cell_volume(i)));
        Real diagonal = 0;
        for (auto k = plan.offsets(i); k < plan.offsets(i+1); ++k) {
            const std::size_t f = static_cast<std::size_t>(plan.faces(k));
            const Real d = static_cast<Real>(diffusion(f));
            const Real a = d*plan.area_over_distance(f);
            if (!Kokkos::isfinite(d) || !(d > Real{0}) || !Kokkos::isfinite(a) || !(a > Real{0})) {
                flag(flags, BadDiffusion); continue;
            }
            if (!(static_cast<Scalar>(a) > Scalar{0})) { flag(flags, BadSystem); continue; }
            diagonal += a;
            values(plan.entries(k)) -= static_cast<Scalar>(a);
        }
        values(plan.diagonal(i)) = static_cast<Scalar>(diagonal);
    }
};

/// @brief Adds area*q_n to the integrated cell equation; only boundary faces need atomic additions.
template<class Space, class Real, class Condition, class Diffusivity, class Values, class Rhs, class Diagonal, class Flags>
struct AssembleBoundary {
    boundary::scalar::GeometryView<Space> geometry;
    Condition condition;
    Diffusivity diffusion;
    Values values;
    Rhs rhs;
    Diagonal diagonal;
    Flags flags;
    Real time;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const std::size_t f = static_cast<std::size_t>(face);
        const Real d = static_cast<Real>(diffusion(f));
        if (!Kokkos::isfinite(d) || !(d > Real{0})) { flag(flags, BadDiffusion); return; }
        const auto c = geometry.template context<Real>(face,
            boundary::scalar::UniformDiffusivity<Real>{d}, boundary::scalar::NoCorrection{}, time);
        const auto r = condition(c);
        if (!Kokkos::isfinite(r.flux_owner) || !(r.flux_owner >= Real{0}) ||
            !Kokkos::isfinite(r.flux_constant)) { flag(flags, BadBoundary); return; }
        const std::size_t i = static_cast<std::size_t>(c.owner);
        using Scalar = typename Values::non_const_value_type;
        const Scalar a = static_cast<Scalar>(c.area*r.flux_owner);
        const Scalar b = static_cast<Scalar>(-c.area*r.flux_constant);
        if (!Kokkos::isfinite(a) || !Kokkos::isfinite(b)) { flag(flags, BadSystem); return; }
        if (r.flux_owner > Real{0}) {
            if (!(a > Scalar{0})) { flag(flags, BadSystem); return; }
            flag(flags, HasAnchor);
        }
        Kokkos::atomic_add(&values(diagonal(i)), a);
        Kokkos::atomic_add(&rhs(i, 0), b);
    }
};

template<class Values, class Rhs, class Diagonal, class Flags>
struct ValidateSystem {
    Values values;
    Rhs rhs;
    Diagonal diagonal;
    Flags flags;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t k) const
    {
        if (k < values.extent(0) && !Kokkos::isfinite(values(k))) flag(flags, BadSystem);
        if (k < rhs.extent(0) && (!Kokkos::isfinite(rhs(k, 0)) || !(values(diagonal(k)) > 0)))
            flag(flags, BadSystem);
    }
};

template<class Space, class Source>
struct IntegrateSource {
    mesh::compute::MeshView<Space> mesh;
    Source source;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell, AccumulationReal& sum) const
    {
        const auto i = static_cast<std::size_t>(cell);
        sum += static_cast<AccumulationReal>(source(i))*static_cast<AccumulationReal>(mesh.geometry.cell_volume(i));
    }
};

template<class Space, class Real, class Condition, class Diffusivity, class Field>
struct IntegrateBoundary {
    boundary::scalar::GeometryView<Space> geometry;
    Condition condition;
    Diffusivity diffusion;
    Field field;
    Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, AccumulationReal& sum) const
    {
        const auto c = geometry.template context<Real>(face, diffusion, boundary::scalar::NoCorrection{}, time);
        const auto r = condition(c);
        const auto value = static_cast<Real>(field(static_cast<std::size_t>(c.owner)));
        sum += static_cast<AccumulationReal>(c.area)*static_cast<AccumulationReal>(r.outward_flux(value));
    }
};

} // namespace krona::transport::scalar::detail
