#pragma once

#include <krona/boundary/physical/evaluate.hpp>
#include <krona/boundary/physical/fields.hpp>

namespace krona::boundary::physical::detail {

template<class Model,class Reader,class Sink,class Law,class Correction>
struct Evaluate {
    Model model; Reader reader; Sink sink; Law law; Correction correction;
    template<class Real,class Condition,class State>
    KOKKOS_INLINE_FUNCTION auto evaluate_local(const FaceContext<Real>& c,
        const Condition& condition,const State& inside) const
    {
        if constexpr (requires { model.at(c,condition); })
            return physical::evaluate(condition,c,inside,model.at(c,condition),law,correction);
        else return physical::evaluate(condition,c,inside,model,law,correction);
    }
    template<class Real,class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c,const Condition& condition) const
    {
        const auto inside=reader(c,condition);
        const auto result=evaluate_local(c,condition,inside);
        if constexpr (Model::compressible) {
            if (!(result.face.pressure>Real{0}) || !(result.exterior.pressure>Real{0}))
                Kokkos::abort("Physical boundary: nonpositive compressible boundary pressure");
        }
        sink(c,result);
    }
};

template<class Fields>
struct Fill {
    Fields fields;
    template<class Real,class State>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c,const Evaluation<State>& result) const
    { fields.write(c.extended_index,result.face); }
};

} // namespace krona::boundary::physical::detail
