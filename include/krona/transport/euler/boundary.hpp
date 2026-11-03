#pragma once
#include <krona/boundary/physical/evaluate.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace krona::transport::euler {
namespace detail {
// Preserve Euler's status-based invalid-exterior path when reusing checked physical closures.
struct ExteriorErrors {
    bool* invalid;
    KOKKOS_INLINE_FUNCTION void fail(boundary::physical::Error,const char*) const { *invalid=true; }
    KOKKOS_INLINE_FUNCTION bool failed() const { return *invalid; }
};
template<class C,class R,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluated_exterior(const C& condition,const boundary::FaceContext<R>& context,
    boundary::physical::Primitive<R,true> inside,const Model& model,const Law& law,const Correction& correction)
{
    bool invalid=false;
    const auto result=boundary::physical::evaluate(condition,context,inside,model,law,correction,ExteriorErrors{&invalid});
    if (invalid) { inside.pressure=Kokkos::quiet_NaN_v<R>; return inside; }
    return result.exterior;
}
template<class C>inline constexpr bool wall=false;
template<class R>inline constexpr bool wall<boundary::physical::SlipWall<R>> =true;
template<class R>inline constexpr bool wall<boundary::physical::AdiabaticWall<R>> =true;
template<class R>inline constexpr bool wall<boundary::physical::IsothermalWall<R>> =true;
template<class R>inline constexpr bool wall<boundary::physical::HeatFluxWall<R>> =true;
template<class C>inline constexpr bool prepared_trace=false;
template<class R>inline constexpr bool prepared_trace<boundary::physical::PressureOutlet<R>> =true;
template<class R>inline constexpr bool prepared_trace<boundary::physical::TotalConditionsInlet<R>> =true;
template<class R>inline constexpr bool prepared_trace<boundary::physical::SupersonicInlet<R>> =true;
template<class R>inline constexpr bool prepared_trace<boundary::physical::VelocityInlet<R>> =true;
}

/// @brief Consumes current physical-face slots for inlet/pressure conditions; mechanical walls mirror the interior trace.
/// No ghost-cell array. Custom conditions use physical::evaluate or a caller-supplied replacement policy.
struct PreparedExterior {
    template<class R,class C,class Model>
    KOKKOS_INLINE_FUNCTION auto operator()(const boundary::FaceContext<R>& context,const C& condition,
        boundary::physical::Primitive<R,true> inside,boundary::physical::Primitive<R,true> face,const Model& model)const
    {
        namespace physical=boundary::physical;
        using Numeric=physical::condition_type_t<C>;
        if constexpr(detail::prepared_trace<Numeric>)return face;
        else if constexpr(detail::wall<Numeric>){
            const auto u=physical::sample_wall_velocity(condition,context);
            // The existing mechanical mirror is independent of wall thermal data or a flux-to-gradient law.
            return detail::evaluated_exterior(physical::SlipWall<R>{{static_cast<R>(u.x),static_cast<R>(u.y),static_cast<R>(u.z)}},
                context,inside,model,boundary::scalar::NoFluxLaw{},physical::Corrections<>{});
        }else return detail::evaluated_exterior(condition,context,inside,model,boundary::scalar::NoFluxLaw{},physical::Corrections<>{});
    }
};
/// @brief Re-evaluate the existing boundary law from the reconstructed interior trace.
/// Mechanical mirrors bypass thermal laws; custom coupled conditions may use the supplied law/correction.
template<class Law=boundary::scalar::NoFluxLaw,class Correction=boundary::physical::Corrections<>>
struct EvaluatedExterior {
    Law law{};
    Correction correction{};
    template<class R,class C,class Model>
    KOKKOS_INLINE_FUNCTION auto operator()(const boundary::FaceContext<R>& context,const C& condition,
        boundary::physical::Primitive<R,true> inside,boundary::physical::Primitive<R,true> face,const Model& model)const
    {
        using Numeric=boundary::physical::condition_type_t<C>;
        if constexpr(detail::wall<Numeric>||std::is_same_v<Numeric,boundary::physical::Symmetry>)
            return PreparedExterior{}(context,condition,inside,face,model);
        else return detail::evaluated_exterior(condition,context,inside,model,law,correction);
    }
};
template<class L>EvaluatedExterior(L)->EvaluatedExterior<L>;
template<class L,class C>EvaluatedExterior(L,C)->EvaluatedExterior<L,C>;
} // namespace krona::transport::euler