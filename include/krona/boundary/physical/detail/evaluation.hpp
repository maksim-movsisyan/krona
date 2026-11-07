#pragma once

#include <krona/boundary/physical/models.hpp>
#include <krona/boundary/physical/relations.hpp>
#include <krona/boundary/physical/detail/parameters.hpp>

namespace krona::boundary::physical::detail {

template<class C,class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto no_slip(C wall,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model&,const Law& law,const Correction& correction,[[maybe_unused]] const Errors& errors)
{
    auto face = inside;
    face.pressure += static_cast<Real>(correction.pressure(c));
    face.velocity = wall.velocity;
    if constexpr (Model::thermal) {
        if constexpr (std::is_same_v<C,IsothermalWall<Real>>) {
            face.temperature=wall.temperature; // Dirichlet trace does not read temperature gradients.
        } else {
            const Real increment=static_cast<Real>(correction.temperature(c));
            if (!require_device(Kokkos::isfinite(increment),errors,Error::Relation,"Physical boundary: invalid temperature correction"))
                return Evaluation{inside,inside};
            const auto relation=wall_temperature_relation(wall,c,increment,law);
            if (!require_device(scalar::finite(relation),errors,Error::Relation,"Physical boundary: invalid wall temperature relation"))
                return Evaluation{inside,inside};
            face.temperature=relation.face_value(inside.temperature);
        }
    }
    // Thermal Dirichlet data act on the viscous trace, not on the inviscid mirror state.
    auto exterior = inside;
    const auto n = normal(c);
    exterior.velocity = inside.velocity-n*(Real{2}*(inside.velocity-wall.velocity).dot(n));
    
    return Evaluation{face,exterior};
}

template<class Real,class Model,class Errors>
KOKKOS_INLINE_FUNCTION Primitive<Real,Model::thermal> inlet(
    TotalConditionsInlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,[[maybe_unused]] const Errors& errors)
{
    if constexpr (Model::compressible && Model::has_characteristics)
    {
        if constexpr (requires { model.characteristics.inlet(model.eos,c,inside,specified,errors); })
            return model.characteristics.inlet(model.eos,c,inside,specified,errors);
        else return model.characteristics.inlet(model.eos,c,inside,specified);
    }
    else {
        errors.fail(Error::Unsupported,"Physical boundary: total inlet requires a compatible characteristic policy");
        return inside;
    }
}

template<class Real,class Model,class Errors>
KOKKOS_INLINE_FUNCTION Primitive<Real,Model::thermal> backflow(
    PressureOutlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,[[maybe_unused]] const Errors& errors)
{
    if (specified.backflow.mode == BackflowMode::Reject) {
        errors.fail(Error::Characteristics,"Physical boundary: reverse flow requires explicit backflow data"); return inside;
    }
    if constexpr (Model::compressible) {
        return inlet(specified.backflow.reservoir,c,inside,model,errors);
    } else {
        const auto n = normal(c);
        const Real mu = specified.backflow.direction.dot(n);
        if (!(mu < Real{0})) {
            errors.fail(Error::Characteristics,"Physical boundary: backflow direction must point into the domain"); return inside;
        }
        inside.pressure = specified.pressure;
        inside.velocity = specified.backflow.direction*(inside.velocity.dot(n)/mu);
        if constexpr (Model::thermal) inside.temperature = specified.backflow.temperature;
        return inside;
    }
}

/// Custom extension point. The callable receives typed context/state/model and explicit helpers.
template<class Condition,class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(Condition condition,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction,[[maybe_unused]] const Errors& errors)
{
    if constexpr (requires { condition(c,inside,model,law,correction,errors); })
        return condition(c,inside,model,law,correction,errors);
    else return condition(c,inside,model,law,correction);
}

template<class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(Symmetry,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model&,const Law&,const Correction& correction,[[maybe_unused]] const Errors& errors)
{
    const auto n = normal(c);
    auto face = inside;
    const auto reconstructed = inside.velocity+correction.velocity(c);
    face.velocity = reconstructed-n*reconstructed.dot(n);
    face.pressure += static_cast<Real>(correction.pressure(c));
    if constexpr (Model::thermal) face.temperature += static_cast<Real>(correction.temperature(c));
    auto exterior = inside;
    exterior.velocity = inside.velocity-n*(Real{2}*inside.velocity.dot(n));
    
    return Evaluation{face,exterior};
}

template<class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(SlipWall<Real> wall,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction,[[maybe_unused]] const Errors& errors)
{
    auto result = evaluate_numeric(Symmetry{},c,inside,model,law,correction,errors);
    const auto n = normal(c);
    result.face.velocity = result.face.velocity+n*wall.velocity.dot(n);
    result.exterior.velocity = inside.velocity-n*(Real{2}*(inside.velocity-wall.velocity).dot(n));
    return result;
}

template<class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(AdiabaticWall<Real> wall,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction,[[maybe_unused]] const Errors& errors)
{ return no_slip(wall,c,inside,model,law,correction,errors); }

template<class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(IsothermalWall<Real> wall,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction,[[maybe_unused]] const Errors& errors)
{ return no_slip(wall,c,inside,model,law,correction,errors); }

template<class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(HeatFluxWall<Real> wall,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction,[[maybe_unused]] const Errors& errors)
{ return no_slip(wall,c,inside,model,law,correction,errors); }

template<class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(TotalConditionsInlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law&,const Correction&,[[maybe_unused]] const Errors& errors)
{
    const auto face = inlet(specified,c,inside,model,errors);
    
    return Evaluation{face,face};
}

template<class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(PressureOutlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law&,const Correction& correction,[[maybe_unused]] const Errors& errors)
{
    auto face = inside;
    const Real un = inside.velocity.dot(normal(c));
    if constexpr (Model::compressible) {
        if constexpr (Model::has_characteristics) {
            const auto qi = model.thermodynamic(inside);
            if (un >= qi.sound_speed) return Evaluation{inside,inside};
            if (un < Real{0}) face = backflow(specified,c,inside,model,errors);
            else {
                if constexpr (requires { model.characteristics.outlet(model.eos,c,inside,specified.pressure,errors); })
                    face=model.characteristics.outlet(model.eos,c,inside,specified.pressure,errors);
                else face=model.characteristics.outlet(model.eos,c,inside,specified.pressure);
                if (face.velocity.dot(normal(c)) < Real{0}) face = backflow(specified,c,inside,model,errors);
            }
        } else errors.fail(Error::Unsupported,"Physical boundary: pressure outlet requires a characteristic policy");
    } else {
        if (un < Real{0}) face = backflow(specified,c,inside,model,errors);
        else {
            face.pressure = specified.pressure;
            face.velocity = inside.velocity+correction.velocity(c);
            if constexpr (Model::thermal) face.temperature += static_cast<Real>(correction.temperature(c));
        }
    }
    
    return Evaluation{face,face};
}

template<class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(VelocityInlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model&,const Law&,const Correction& correction,[[maybe_unused]] const Errors& errors)
{
    if constexpr (Model::compressible)
        errors.fail(Error::Unsupported,"Physical boundary: VelocityInlet currently targets incompressible flow");
    auto face = inside;
    face.pressure += static_cast<Real>(correction.pressure(c));
    face.velocity = specified.velocity;
    if constexpr (Model::thermal) face.temperature = specified.temperature;
    
    return Evaluation{face,face};
}

template<class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(SupersonicInlet<Real> specified,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law&,const Correction&,[[maybe_unused]] const Errors& errors)
{
    auto face = inside;
    if constexpr (Model::compressible) {
        face = {specified.pressure,specified.velocity,specified.temperature};
        if (specified.mach>Real{0})
            face.velocity=specified.direction*(specified.mach*model.thermodynamic(face).sound_speed);
        if (!(face.velocity.dot(normal(c)) < -model.thermodynamic(face).sound_speed))
            errors.fail(Error::Characteristics,"Physical boundary: prescribed supersonic inlet is not supersonic in the inward normal direction");
    } else errors.fail(Error::Unsupported,"Physical boundary: supersonic inlet requires compressible flow");
    return Evaluation{face,face};
}

template<class Real,class Model,class Law,class Correction,class Errors>
KOKKOS_INLINE_FUNCTION auto evaluate_numeric(SupersonicOutlet,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law&,const Correction&,[[maybe_unused]] const Errors& errors)
{
    if constexpr (Model::compressible) {
        if (!(inside.velocity.dot(normal(c)) >= model.thermodynamic(inside).sound_speed))
            errors.fail(Error::Characteristics,"Physical boundary: supersonic outlet has an incoming characteristic");
    } else errors.fail(Error::Unsupported,"Physical boundary: supersonic outlet requires compressible flow");
    return Evaluation{inside,inside};
}


} // namespace krona::boundary::physical::detail
