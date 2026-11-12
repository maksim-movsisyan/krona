#pragma once

#include <krona/boundary/physical/evaluate.hpp>
#include <krona/boundary/physical/fields.hpp>

namespace krona::boundary::physical::detail {

template<class Model,class Reader,class Sink,class Law,class Correction,class Flags>
struct Evaluate {
    Model model; Reader reader; Sink sink; Law law; Correction correction; Flags flags;

    template<class Real,class Condition,class State,class Errors>
    KOKKOS_INLINE_FUNCTION auto evaluate_local(const FaceContext<Real>& c,
        const Condition& condition,const State& inside,const Errors& errors) const {
        if constexpr (requires { model.at(c,condition); }) {
            const auto local=model.at(c,condition);
            static_assert(decltype(local)::thermal==Model::thermal&&decltype(local)::compressible==Model::compressible,
                          "Physical boundary: local model capabilities must match the manager model");
            return physical::evaluate(condition,c,inside,local,law,correction,errors);
        } else return physical::evaluate(condition,c,inside,model,law,correction,errors);
    }
    
    template<class Real,class Condition>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c,const Condition& condition) const {
        bool failed=false;
        const DeviceErrors<Flags> errors{flags,&failed};
        const auto inside=reader(c,condition);
        const auto result=evaluate_local(c,condition,inside,errors);
        if (!failed) sink(c,result,errors);
    }
};

// Adapt ordinary public sinks while keeping checked output filling internal.
template<class Sink>
struct CheckedSink {
    Sink sink;
    template<class Context,class Result,class Errors>
    KOKKOS_INLINE_FUNCTION void operator()(const Context& c,const Result& result,const Errors&) const
    { sink(c,result); }
};
template<class Fields>
struct Fill {
    Fields fields;
    template<class Real,class State,class Errors>
    KOKKOS_INLINE_FUNCTION void operator()(const FaceContext<Real>& c,const Evaluation<State>& result,const Errors& errors) const {
        if (!fields.can_store(result.face)) { errors.fail(Error::Output,"Physical boundary: unrepresentable primitive output"); return; }
        fields.write(c.extended_index,result.face);
    }
};

} // namespace krona::boundary::physical::detail
