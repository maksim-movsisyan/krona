#pragma once

#include <krona/boundary/physical/detail/evaluation.hpp>
#include <krona/boundary/physical/profiles.hpp>

namespace krona::boundary::physical {

/// @brief Device evaluation. Samples each provider once, validates, then evaluates the numeric closure.
/// Manager kernels supply checked errors. Direct callers default to Kokkos::abort on BC errors.
/// EOS/material/custom-policy routines retain their own documented domain/error contracts.
template<class Condition,class Real,class Model,class Law,class Correction,class Errors=detail::AbortErrors>
KOKKOS_INLINE_FUNCTION auto evaluate(const Condition& condition,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model,const Law& law,const Correction& correction,
    const Errors& errors={})
{
    auto numeric=sample_condition(condition,c);
    const auto failed=Evaluation{inside,inside};
    if (!detail::prepare_device(numeric,model,errors,is_profiled<Condition>)||
        !detail::valid_state(inside,model,errors,Error::Interior)) return failed;
    if constexpr (std::is_same_v<decltype(numeric),HeatFluxWall<Real>>&&!scalar::FluxLaw<Law,Real>) {
        errors.fail(Error::Unsupported,"Physical boundary: heat-flux wall requires an explicit flux law");
        return failed;
    }
    auto result=detail::evaluate_numeric(numeric,c,inside,model,law,correction,errors);
    if (!errors.failed()) {
        (void)detail::valid_state(result.face,model,errors,Error::State);
        (void)detail::valid_state(result.exterior,model,errors,Error::State);
    }
    return result;
}

/// @brief Direct use without a flux law or reconstruction correction.
template<class Condition,class Real,class Model>
KOKKOS_INLINE_FUNCTION auto evaluate(const Condition& condition,const FaceContext<Real>& c,
    Primitive<Real,Model::thermal> inside,const Model& model)
{ return evaluate(condition,c,inside,model,scalar::NoFluxLaw{},Corrections<>{}); }

} // namespace krona::boundary::physical
