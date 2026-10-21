#pragma once

#include <krona/boundary/scalar/manager.hpp>
#include <krona/transport/scalar/diffusion/detail/providers.hpp>
#include <krona/transport/scalar/status.hpp>
#include <krona/transport/scalar/system_view.hpp>

namespace krona::transport::scalar::diffusion::detail {

// Reuse the scalar assembly diagnostics for every diffusion contribution.
using scalar::detail::flag;
using scalar::detail::add_checked;
using scalar::detail::atomic_add_checked;
using scalar::detail::BadDiffusion;
using scalar::detail::BadBoundary;
using scalar::detail::BadSystem;
using scalar::detail::HasAnchor;

// Both algebraic assembly and matrix-free residuals use these same integrated flux formulas.
template<class Real, class Space, class Law, class Flags>
KOKKOS_INLINE_FUNCTION Real conductance(const RowData<Space>& data, LocalIndex face,
    const Law& law, Flags flags, Real time)
{
    const auto c = interior_context<Real>(data, face, time);
    const Real coefficient = static_cast<Real>(law.coefficient(c));
    const Real value = coefficient*static_cast<Real>(data.area_over_distance(static_cast<std::size_t>(face)));
    if (!Kokkos::isfinite(coefficient) || !(coefficient > Real{0}) ||
        !Kokkos::isfinite(value) || !(value > Real{0})) {
        flag(flags, BadDiffusion); return Real{0};
    }
    return value;
}

template<class Real> struct AffineFlux { Real owner = 0, constant = 0; };
template<class Real, class Condition, class Law, class Flags>
KOKKOS_INLINE_FUNCTION AffineFlux<Real> boundary_flux(const boundary::FaceContext<Real>& c,
    const Condition& condition, const Law& law, Flags flags)
{
    const Real coefficient = static_cast<Real>(law.coefficient(c));
    if (!Kokkos::isfinite(coefficient) || !(coefficient > Real{0})) {
        flag(flags, BadDiffusion); return {};
    }
    const auto sampled = IsotropicDiffusion{UniformCoefficient<Real>{coefficient}};
    const auto relation = boundary::scalar::make_relation(condition, c, Real{0}, sampled);
    const Real owner = sampled.outward_flux(c, relation.gradient_owner);
    const Real constant = sampled.outward_flux(c, relation.gradient_constant);
    if (!Kokkos::isfinite(owner) || owner < Real{0} || !Kokkos::isfinite(constant)) {
        flag(flags, BadBoundary); return {};
    }
    const Real a = c.area*owner, b = c.area*constant;
    if (!Kokkos::isfinite(a) || !Kokkos::isfinite(b)) { flag(flags, BadSystem); return {}; }
    if (owner > Real{0}) flag(flags, HasAnchor);
    return {a, b};
}

/// @brief One thread owns one row; all existing coefficients and RHS entries are preserved.
template<class Space, class Real, class Target, class Law, class Flags>
struct AddRows {
    RowData<Space> data; Target target; Law law; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const std::size_t i = static_cast<std::size_t>(cell);
        using Value = typename decltype(target.values)::non_const_value_type;
        Real diagonal = 0;
        for (auto k = data.offsets(i); k < data.offsets(i+1); ++k) {
            const auto face = data.faces(k);
            if (face >= data.n_inner_faces) continue;
            const Real a = conductance<Real>(data, face, law, flags, time);
            if (!(a > Real{0})) continue;
            if (!Kokkos::isfinite(static_cast<Value>(a)) || !(static_cast<Value>(a) > Value{0})) {
                flag(flags, BadSystem); continue;
            }
            const std::size_t f = static_cast<std::size_t>(face);
            const auto entry = data.face_owner(f) == cell ? target.entries.owner_entries(f) : target.entries.neighbour_entries(f);
            add_checked(target.values, entry, -a, flags);
            diagonal += a;
        }
        add_checked(target.values, target.entries.diagonal(i), diagonal, flags);
    }
};

template<class Space, class Real, class Target, class Condition, class Law, class Flags>
struct AddBoundary {
    boundary::GeometryView<Space> geometry; Target target; Condition condition; Law law; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const auto flux = boundary_flux(c, condition, law, flags);
        using Value = typename decltype(target.values)::non_const_value_type;
        if (flux.owner > Real{0} && !(static_cast<Value>(flux.owner) > Value{0})) {
            flag(flags, BadSystem); return;
        }
        const std::size_t i = static_cast<std::size_t>(c.owner);
        atomic_add_checked(target.values, target.entries.diagonal(i), flux.owner, flags);
        atomic_add_checked(target.rhs, i, -flux.constant, flags);
    }
};

/// @brief Adds -A_interior*phi directly from values, without a graph or an intermediate face array.
template<class Space, class Real, class Field, class Output, class Law, class Flags>
struct ResidualRows {
    RowData<Space> data; Field field; Output output; Law law; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const std::size_t i = static_cast<std::size_t>(cell);
        const Real own = static_cast<Real>(field(i));
        if (!Kokkos::isfinite(own)) { flag(flags, BadSystem); return; }
        Real residual = 0;
        for (auto k = data.offsets(i); k < data.offsets(i+1); ++k) {
            const auto face = data.faces(k);
            if (face >= data.n_inner_faces) continue;
            const std::size_t f = static_cast<std::size_t>(face);
            const auto other = data.face_owner(f) == cell ? data.face_neigh(f) : data.face_owner(f);
            const Real value = static_cast<Real>(field(static_cast<std::size_t>(other)));
            const Real a = conductance<Real>(data, face, law, flags, time);
            residual += a*(value-own);
        }
        add_checked(output, i, residual, flags);
    }
};

template<class Space, class Real, class Field, class Output, class Condition, class Law, class Flags>
struct ResidualBoundary {
    boundary::GeometryView<Space> geometry; Field field; Output output;
    Condition condition; Law law; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const auto flux = boundary_flux(c, condition, law, flags);
        const std::size_t i = static_cast<std::size_t>(c.owner);
        const Real value = flux.owner == Real{0} ? flux.constant :
            flux.owner*static_cast<Real>(field(i))+flux.constant;
        atomic_add_checked(output, i, -value, flags);
    }
};

template<class Values, class Rhs, class Diagonal, class Flags>
struct ValidateSystem {
    Values values; Rhs rhs; Diagonal diagonal; Flags flags;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    {
        if (i < values.extent(0) && !Kokkos::isfinite(values(i))) flag(flags, BadSystem);
        if (i < rhs.extent(0) && (!Kokkos::isfinite(rhs(i)) || !(values(diagonal(i)) > 0))) flag(flags, BadSystem);
    }
};

template<class Space, class Source>
struct IntegrateSource {
    mesh::compute::MeshView<Space> mesh; Source source;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell, AccumulationReal& sum) const
    {
        const std::size_t i = static_cast<std::size_t>(cell);
        sum += static_cast<AccumulationReal>(source(i))*static_cast<AccumulationReal>(mesh.geometry.cell_volume(i));
    }
};

template<class Space, class Real, class Condition, class Law, class Field>
struct IntegrateBoundary {
    boundary::GeometryView<Space> geometry; Condition condition; Law law; Field field; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, AccumulationReal& sum) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const auto r = boundary::scalar::make_relation(condition, c, Real{0}, law);
        sum += static_cast<AccumulationReal>(c.area)*static_cast<AccumulationReal>(
            law.outward_flux(c, r.normal_gradient(static_cast<Real>(field(static_cast<std::size_t>(c.owner))))));
    }
};

} // namespace krona::transport::scalar::diffusion::detail
