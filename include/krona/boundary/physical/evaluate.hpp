#pragma once

#include <krona/boundary/physical/models.hpp>
#include <krona/boundary/physical/relations.hpp>

namespace krona::boundary::physical {

namespace detail {
template<class Real,bool Thermal>
KOKKOS_INLINE_FUNCTION void check_state(const Primitive<Real,Thermal>& s)
{
    if (!Kokkos::isfinite(s.pressure) || !Kokkos::isfinite(s.velocity.x) ||
        !Kokkos::isfinite(s.velocity.y) || !Kokkos::isfinite(s.velocity.z))
        Kokkos::abort("Physical boundary: nonfinite primitive state");
    if constexpr (Thermal) if (!(s.temperature > Real{0}) || !Kokkos::isfinite(s.temperature))
        Kokkos::abort("Physical boundary: nonpositive or nonfinite temperature");
}

template<class C,class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto no_slip(C wall,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model&,const Law& law,const Correction& correction)
{
    auto face = inside;
    face.pressure += static_cast<Real>(correction.pressure(c));
    face.velocity = wall.velocity;
    if constexpr (Model::thermal) {
        const auto relation = wall_temperature_relation(wall,c,
            static_cast<Real>(correction.temperature(c)),law);
        face.temperature = relation.face_value(inside.temperature);
    }
    // Thermal Dirichlet data act on the viscous trace, not on the inviscid mirror state.
    auto exterior = inside;
    const auto n = normal(c);
    exterior.velocity = inside.velocity-n*(Real{2}*(inside.velocity-wall.velocity).dot(n));
    check_state(face); check_state(exterior);
    return Evaluation{face,exterior};
}

template<class Real,class Model>
KOKKOS_INLINE_FUNCTION Primitive<Real,Model::thermal> inlet(
    TotalConditionsInlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model)
{
    if constexpr (Model::compressible && Model::has_characteristics)
        return model.characteristics.inlet(model.eos,c,inside,specified);
    else {
        Kokkos::abort("Physical boundary: total inlet requires a compatible characteristic policy");
        return inside;
    }
}

template<class Real,class Model>
KOKKOS_INLINE_FUNCTION Primitive<Real,Model::thermal> backflow(
    PressureOutlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model)
{
    if (specified.backflow.mode == BackflowMode::Reject)
        Kokkos::abort("Physical boundary: reverse flow requires explicit backflow data");
    if constexpr (Model::compressible) {
        return inlet(specified.backflow.reservoir,c,inside,model);
    } else {
        const auto n = normal(c);
        const Real mu = specified.backflow.direction.dot(n);
        if (!(mu < Real{0})) Kokkos::abort("Physical boundary: backflow direction must point into the domain");
        inside.pressure = specified.pressure;
        inside.velocity = specified.backflow.direction*(inside.velocity.dot(n)/mu);
        if constexpr (Model::thermal) inside.temperature = specified.backflow.temperature;
        return inside;
    }
}
}

/// Custom extension point. The callable receives typed context/state/model and explicit helpers.
template<class Condition,class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(Condition condition,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction)
{ return condition(c,inside,model,law,correction); }

template<class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(Symmetry,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model&,const Law&,const Correction& correction)
{
    const auto n = normal(c);
    auto face = inside;
    const auto reconstructed = inside.velocity+correction.velocity(c);
    face.velocity = reconstructed-n*reconstructed.dot(n);
    face.pressure += static_cast<Real>(correction.pressure(c));
    if constexpr (Model::thermal) face.temperature += static_cast<Real>(correction.temperature(c));
    auto exterior = inside;
    exterior.velocity = inside.velocity-n*(Real{2}*inside.velocity.dot(n));
    detail::check_state(face); detail::check_state(exterior);
    return Evaluation{face,exterior};
}

template<class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(SlipWall<Real> wall,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction)
{
    auto result = evaluate(Symmetry{},c,inside,model,law,correction);
    const auto n = normal(c);
    result.face.velocity = result.face.velocity+n*wall.velocity.dot(n);
    result.exterior.velocity = inside.velocity-n*(Real{2}*(inside.velocity-wall.velocity).dot(n));
    return result;
}

template<class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(AdiabaticWall<Real> wall,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction)
{ return detail::no_slip(wall,c,inside,model,law,correction); }

template<class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(IsothermalWall<Real> wall,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction)
{ return detail::no_slip(wall,c,inside,model,law,correction); }

template<class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(HeatFluxWall<Real> wall,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction)
{ return detail::no_slip(wall,c,inside,model,law,correction); }

template<class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(TotalConditionsInlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law&,const Correction&)
{
    const auto face = detail::inlet(specified,c,inside,model);
    detail::check_state(face);
    return Evaluation{face,face};
}

template<class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(PressureOutlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law&,const Correction& correction)
{
    auto face = inside;
    const Real un = inside.velocity.dot(normal(c));
    if constexpr (Model::compressible) {
        if constexpr (Model::has_characteristics) {
            const auto qi = model.thermodynamic(inside);
            if (un >= qi.sound_speed) return Evaluation{inside,inside};
            if (un < Real{0}) face = detail::backflow(specified,c,inside,model);
            else {
                face = model.characteristics.outlet(model.eos,c,inside,specified.pressure);
                if (face.velocity.dot(normal(c)) < Real{0}) face = detail::backflow(specified,c,inside,model);
            }
        } else Kokkos::abort("Physical boundary: pressure outlet requires a characteristic policy");
    } else {
        if (un < Real{0}) face = detail::backflow(specified,c,inside,model);
        else {
            face.pressure = specified.pressure;
            face.velocity = inside.velocity+correction.velocity(c);
            if constexpr (Model::thermal) face.temperature += static_cast<Real>(correction.temperature(c));
        }
    }
    detail::check_state(face);
    return Evaluation{face,face};
}

template<class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(VelocityInlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model&,const Law&,const Correction& correction)
{
    if constexpr (Model::compressible)
        Kokkos::abort("Physical boundary: VelocityInlet currently targets incompressible flow");
    auto face = inside;
    face.pressure += static_cast<Real>(correction.pressure(c));
    face.velocity = specified.velocity;
    if constexpr (Model::thermal) face.temperature = specified.temperature;
    detail::check_state(face);
    return Evaluation{face,face};
}

template<class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(SupersonicInlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law&,const Correction&)
{
    auto face = inside;
    if constexpr (Model::compressible) {
        face = {specified.pressure,specified.velocity,specified.temperature};
        if (specified.mach>Real{0})
            face.velocity=specified.direction*(specified.mach*model.thermodynamic(face).sound_speed);
        if (!(face.velocity.dot(normal(c)) < -model.thermodynamic(face).sound_speed))
            Kokkos::abort("Physical boundary: prescribed supersonic inlet is not supersonic in the inward normal direction");
    } else Kokkos::abort("Physical boundary: supersonic inlet requires compressible flow");
    return Evaluation{face,face};
}

template<class Real,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluate(SupersonicOutlet,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law&,const Correction&)
{
    if constexpr (Model::compressible) {
        if (!(inside.velocity.dot(normal(c)) >= model.thermodynamic(inside).sound_speed))
            Kokkos::abort("Physical boundary: supersonic outlet has an incoming characteristic");
    } else Kokkos::abort("Physical boundary: supersonic outlet requires compressible flow");
    return Evaluation{inside,inside};
}

/// Direct device use, e.g. inside a custom residual kernel.
template<class Condition,class Real,class Model>
KOKKOS_INLINE_FUNCTION auto evaluate(Condition condition,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model)
{ return evaluate(condition,c,inside,model,scalar::NoFluxLaw{},Corrections<>{}); }

} // namespace krona::boundary::physical
