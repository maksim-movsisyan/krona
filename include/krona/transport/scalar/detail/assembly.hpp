#pragma once

#include <krona/boundary/scalar/manager.hpp>
#include <krona/transport/scalar/detail/providers.hpp>

namespace krona::transport::scalar::detail {

enum Status : int { BadSource, BadDiffusion, BadBoundary, HasAnchor, BadSystem, StatusCount };
template<class Flags>
KOKKOS_INLINE_FUNCTION void flag(Flags flags, Status index)
{ Kokkos::atomic_exchange(&flags(static_cast<int>(index)), 1); }

/// @brief Rows own their updates. Duplicate connecting faces accumulate without atomics.
template<class Space, class Real, class Values, class Rhs, class Source, class Law, class Flags>
struct AssembleRows {
    RowData<Space> data;
    linalg::FaceAssemblyMap<Space> plan;
    Values values;
    Rhs rhs;
    Source source;
    Law law;
    Flags flags;
    Real time;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const auto i = static_cast<std::size_t>(cell);
        using Scalar = typename Values::non_const_value_type;
        const Real density = static_cast<Real>(source(i));
        if (!Kokkos::isfinite(density)) { flag(flags, BadSource); return; }
        rhs(i, 0) = static_cast<Scalar>(density*static_cast<Real>(data.cell_volume(i)));
        Real diagonal = 0;
        for (auto k = data.offsets(i); k < data.offsets(i+1); ++k) {
            const auto face = data.faces(k);
            if (face >= data.n_inner_faces) continue;
            const auto f = static_cast<std::size_t>(face);
            const auto c = interior_context<Real>(data, face, time);
            const Real d = static_cast<Real>(law.coefficient(c));
            const Real a = d*static_cast<Real>(data.area_over_distance(f));
            if (!Kokkos::isfinite(d) || !(d > Real{0}) || !Kokkos::isfinite(a) || !(a > Real{0})) {
                flag(flags, BadDiffusion); continue;
            }
            if (!(static_cast<Scalar>(a) > Scalar{0})) { flag(flags, BadSystem); continue; }
            diagonal += a;
            const auto entry = c.owner == cell ? plan.owner_entries(f) : plan.neighbour_entries(f);
            values(entry) -= static_cast<Scalar>(a);
        }
        values(plan.diagonal(i)) = static_cast<Scalar>(diagonal);
    }
};

/// @brief Converts a geometric normal-gradient relation into an integrated physical flux.
template<class Space, class Real, class Condition, class Law, class Values, class Rhs, class Diagonal, class Flags>
struct AssembleBoundary {
    boundary::GeometryView<Space> geometry;
    Condition condition;
    Law law;
    Values values;
    Rhs rhs;
    Diagonal diagonal;
    Flags flags;
    Real time;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const Real d = static_cast<Real>(law.coefficient(c));
        if (!Kokkos::isfinite(d) || !(d > Real{0})) { flag(flags, BadDiffusion); return; }
        const auto sampled = IsotropicDiffusion{UniformCoefficient<Real>{d}};
        const auto r = boundary::scalar::make_relation(condition, c, Real{0}, sampled);
        const Real owner = sampled.outward_flux(c, r.gradient_owner);
        const Real constant = sampled.outward_flux(c, r.gradient_constant);
        if (!Kokkos::isfinite(owner) || !(owner >= Real{0}) || !Kokkos::isfinite(constant)) {
            flag(flags, BadBoundary); return;
        }
        using Scalar = typename Values::non_const_value_type;
        const Scalar a = static_cast<Scalar>(c.area*owner), b = static_cast<Scalar>(-c.area*constant);
        if (!Kokkos::isfinite(a) || !Kokkos::isfinite(b)) { flag(flags, BadSystem); return; }
        if (owner > Real{0}) {
            if (!(a > Scalar{0})) { flag(flags, BadSystem); return; }
            flag(flags, HasAnchor);
        }
        const auto i = static_cast<std::size_t>(c.owner);
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
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    {
        if (i < values.extent(0) && !Kokkos::isfinite(values(i))) flag(flags, BadSystem);
        if (i < rhs.extent(0) && (!Kokkos::isfinite(rhs(i, 0)) || !(values(diagonal(i)) > 0))) flag(flags, BadSystem);
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

template<class Space, class Real, class Condition, class Law, class Field>
struct IntegrateBoundary {
    boundary::GeometryView<Space> geometry;
    Condition condition;
    Law law;
    Field field;
    Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, AccumulationReal& sum) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const auto r = boundary::scalar::make_relation(condition, c, Real{0}, law);
        sum += static_cast<AccumulationReal>(c.area)*static_cast<AccumulationReal>(
            law.outward_flux(c, r.normal_gradient(static_cast<Real>(field(static_cast<std::size_t>(c.owner))))));
    }
};

} // namespace krona::transport::scalar::detail