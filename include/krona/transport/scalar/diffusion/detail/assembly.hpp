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
using scalar::detail::BadCorrection;
using scalar::detail::BadSystem;
using scalar::detail::HasAnchor;

// Canonical owner-outward inner flux: Q = a*(phi_owner-phi_neighbour) + constant.
// constant = -D*T.dot(gradient_face), T = A*(n-d/(d.dot(n))).
template<class Real> struct InteriorFlux { Real conductance = 0, constant = 0; };
template<class Real, class Space, class Law, class Correction, class Flags>
KOKKOS_INLINE_FUNCTION InteriorFlux<Real> interior_flux(const RowData<Space>& data, LocalIndex face,
    const Law& law, const Correction& correction, Flags flags, Real time)
{
    const auto c = interior_context<Real>(data, face, time);
    const Real coefficient = static_cast<Real>(law.coefficient(c));
    const auto f = static_cast<std::size_t>(face);
    const Real a = coefficient*static_cast<Real>(data.area_over_distance(f));
    if (!Kokkos::isfinite(coefficient) || !(coefficient > Real{0}) || !Kokkos::isfinite(a) || !(a > Real{0})) {
        flag(flags, BadDiffusion); return {};
    }
    Real constant = 0;
    if constexpr (corrected<Correction>) {
        const Real increment = static_cast<Real>(correction.interior(c,
            static_cast<Real>(data.non_ortho_x(f)), static_cast<Real>(data.non_ortho_y(f)),
            static_cast<Real>(data.non_ortho_z(f))));
        constant = -coefficient*increment;
        if (!Kokkos::isfinite(increment) || !Kokkos::isfinite(constant)) {
            flag(flags, BadCorrection); return {};
        }
    }
    return {a, constant};
}

template<class Real> struct AffineFlux { Real owner = 0, constant = 0; };
// CorrectionOnly subtracts the zero-increment boundary RHS and requires its matrix slope to stay fixed.
template<bool CorrectionOnly = false, class Real, class Condition, class Law, class Correction, class Flags>
KOKKOS_INLINE_FUNCTION AffineFlux<Real> boundary_flux(const boundary::FaceContext<Real>& c,
    const Condition& condition, const Law& law, const Correction& correction, Flags flags)
{
    const Real coefficient = static_cast<Real>(law.coefficient(c));
    if (!Kokkos::isfinite(coefficient) || !(coefficient > Real{0})) {
        flag(flags, BadDiffusion); return {};
    }
    const Real increment = static_cast<Real>(correction(c));
    if (!Kokkos::isfinite(increment)) { flag(flags, BadCorrection); return {}; }
    const auto sampled = IsotropicDiffusion{UniformCoefficient<Real>{coefficient}};
    const auto relation = boundary::scalar::make_relation(condition, c, increment, sampled);
    const Real owner = sampled.outward_flux(c, relation.gradient_owner);
    Real constant = sampled.outward_flux(c, relation.gradient_constant);
    if (!Kokkos::isfinite(owner) || owner < Real{0} || !Kokkos::isfinite(constant)) {
        flag(flags, BadBoundary); return {};
    } 
    if constexpr (CorrectionOnly) {
        const auto base = boundary::scalar::make_relation(condition, c, Real{0}, sampled);
        if (!Kokkos::isfinite(base.gradient_constant) || relation.gradient_owner != base.gradient_owner) {
            flag(flags, BadCorrection); return {};
        }
        constant = sampled.outward_flux(c, relation.gradient_constant-base.gradient_constant);
    }
    const Real a = CorrectionOnly ? Real{0} : c.area*owner;
    const Real b = c.area*constant;
    if (!Kokkos::isfinite(a) || !Kokkos::isfinite(b)) { flag(flags, BadSystem); return {}; }
    if constexpr (!CorrectionOnly) { if (owner > Real{0}) flag(flags, HasAnchor); }
    return {a, b};
}

/// @brief One iteration owns one row. Correction RHS is added only in the corrected instantiation.
template<class Space, class Real, class Target, class Law, class Correction, class Flags>
struct AddRows {
    RowData<Space> data; Target target; Law law; Correction correction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const auto i = static_cast<std::size_t>(cell);
        using Value = typename decltype(target.values)::non_const_value_type;
        Real diagonal = 0, explicit_rhs = 0;
        for (auto k = data.offsets(i); k < data.offsets(i+1); ++k) {
            const auto face = data.faces(k);
            if (face >= data.n_inner_faces) continue;
            const auto flux = interior_flux<Real>(data, face, law, correction, flags, time);
            const Real a = flux.conductance;
            if (!(a > Real{0})) continue;
            if (!Kokkos::isfinite(static_cast<Value>(a)) || !(static_cast<Value>(a) > Value{0})) {
                flag(flags, BadSystem); continue;
            }
            const auto f = static_cast<std::size_t>(face);
            const bool owner = data.face_owner(f) == cell;
            const auto entry = owner ? target.entries.owner_entries(f) : target.entries.neighbour_entries(f);
            add_checked(target.values, entry, -a, flags);
            diagonal += a;
            if constexpr (corrected<Correction>) explicit_rhs += owner ? -flux.constant : flux.constant;
        }
        add_checked(target.values, target.entries.diagonal(i), diagonal, flags);
        if constexpr (corrected<Correction>) add_checked(target.rhs, i, explicit_rhs, flags);
    }
};

template<class Space, class Real, class Target, class Condition, class Law, class Correction, class Flags>
struct AddBoundary {
    boundary::GeometryView<Space> geometry; Target target; Condition condition; Law law;
    Correction correction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const auto flux = boundary_flux(c, condition, law, correction, flags);
        using Value = typename decltype(target.values)::non_const_value_type;
        if (flux.owner > Real{0} && !(static_cast<Value>(flux.owner) > Value{0})) {
            flag(flags, BadSystem); return;
        }
        const auto i = static_cast<std::size_t>(c.owner);
        atomic_add_checked(target.values, target.entries.diagonal(i), flux.owner, flags);
        atomic_add_checked(target.rhs, i, -flux.constant, flags);
    }
};

/// @brief Adds -A_interior*phi and its explicit correction without a graph or face work buffer.
template<class Space, class Real, class Field, class Output, class Law, class Correction, class Flags>
struct ResidualRows {
    RowData<Space> data; Field field; Output output; Law law; Correction correction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const auto i = static_cast<std::size_t>(cell);
        const Real own = static_cast<Real>(field(i));
        if (!Kokkos::isfinite(own)) { flag(flags, BadSystem); return; }
        Real residual = 0;
        for (auto k = data.offsets(i); k < data.offsets(i+1); ++k) {
            const auto face = data.faces(k);
            if (face >= data.n_inner_faces) continue;
            const auto f = static_cast<std::size_t>(face);
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

template<class Space, class Real, class Field, class Output, class Condition, class Law, class Correction, class Flags>
struct ResidualBoundary {
    boundary::GeometryView<Space> geometry; Field field; Output output;
    Condition condition; Law law; Correction correction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const auto flux = boundary_flux(c, condition, law, correction, flags);
        const auto i = static_cast<std::size_t>(c.owner);
        const Real value = flux.owner == Real{0} ? flux.constant :
            flux.owner*static_cast<Real>(field(i))+flux.constant;
        atomic_add_checked(output, i, -value, flags);
    }
};

/// @brief Adds only the explicit interior correction to an initialized owned RHS/residual prefix.
template<class Space, class Real, class Output, class Law, class Correction, class Flags>
struct CorrectionRows {
    RowData<Space> data; Output output; Law law; Correction correction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        Real value = 0;
        const auto i = static_cast<std::size_t>(cell);
        for (auto k = data.offsets(i); k < data.offsets(i+1); ++k) {
            const auto face = data.faces(k);
            if (face >= data.n_inner_faces) continue;
            const auto f = static_cast<std::size_t>(face);
            const auto flux = interior_flux<Real>(data, face, law, correction, flags, time);
            value += data.face_owner(f) == cell ? -flux.constant : flux.constant;
        }
        add_checked(output, i, value, flags);
    }
};

template<class Space, class Real, class Output, class Condition, class Law, class Correction, class Flags>
struct CorrectionBoundary {
    boundary::GeometryView<Space> geometry; Output output; Condition condition; Law law;
    Correction correction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const auto flux = boundary_flux<true>(c, condition, law, correction, flags);
        atomic_add_checked(output, static_cast<std::size_t>(c.owner), -flux.constant, flags);
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
        const auto i = static_cast<std::size_t>(cell);
        sum += static_cast<AccumulationReal>(source(i))*static_cast<AccumulationReal>(mesh.geometry.cell_volume(i));
    }
};

template<class Space, class Real, class Condition, class Law, class Field, class Correction>
struct IntegrateBoundary {
    boundary::GeometryView<Space> geometry; Condition condition; Law law; Field field; Correction correction; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face, AccumulationReal& sum) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const auto r = boundary::scalar::make_relation(condition, c, static_cast<Real>(correction(c)), law);
        sum += static_cast<AccumulationReal>(c.area)*static_cast<AccumulationReal>(
            law.outward_flux(c, r.normal_gradient(static_cast<Real>(field(static_cast<std::size_t>(c.owner))))));
    }
};

} // namespace krona::transport::scalar::diffusion::detail
