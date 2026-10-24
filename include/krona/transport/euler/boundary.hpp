#pragma once
#include <krona/boundary/physical/evaluate.hpp>

namespace krona::transport::euler {
namespace detail {
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
        if constexpr(detail::prepared_trace<C>)return face;
        else if constexpr(detail::wall<C>){
            const auto u=condition.velocity;
            // The existing mechanical mirror is independent of wall thermal data or a flux-to-gradient law.
            return physical::evaluate(physical::SlipWall<R>{{static_cast<R>(u.x),static_cast<R>(u.y),static_cast<R>(u.z)}},
                context,inside,model).exterior;
        }else return physical::evaluate(condition,context,inside,model).exterior;
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
        if constexpr(detail::wall<C>||std::is_same_v<C,boundary::physical::Symmetry>)
            return PreparedExterior{}(context,condition,inside,face,model);
        else return boundary::physical::evaluate(condition,context,inside,model,law,correction).exterior;
    }
};
template<class L>EvaluatedExterior(L)->EvaluatedExterior<L>;
template<class L,class C>EvaluatedExterior(L,C)->EvaluatedExterior<L,C>;
} // namespace krona::transport::euler
