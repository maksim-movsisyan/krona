#pragma once
#include <krona/boundary/physical/evaluate.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <type_traits>

namespace krona::transport::euler {

namespace detail {
// Custom closures can use the prepared face argument.
template<class Closure>
inline constexpr bool uses_prepared_face = true;

// Preserve Euler's status-based invalid-exterior path when reusing checked physical closures.
struct ExteriorErrors {
    bool* invalid;
    KOKKOS_INLINE_FUNCTION void fail(boundary::physical::Error,const char*) const { *invalid=true; }
    KOKKOS_INLINE_FUNCTION bool failed() const { return *invalid; }
};

template<class C,class R,class Model,class Law,class Correction>
KOKKOS_INLINE_FUNCTION auto evaluated_exterior(const C& condition,const boundary::FaceContext<R>& context,
                                               boundary::physical::Primitive<R,true> inside,const Model& model,
                                               const Law& law,const Correction& correction) {
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
} // namespace detail

/// @brief Uses the prepared primitive face value for inlet and pressure-outlet conditions.
/// Mechanical walls reflect relative normal velocity using the interior trace.
/// Other conditions use physical::evaluate to construct an exterior state.
struct PreparedExterior {
    /// @brief Constructs an exterior state using the face slot when the condition calls for it.
    /// @param face Prepared primitive value stored at context.extended_index.
    /// @tparam R Scalar type used for states and geometry.
    /// @tparam C Typed boundary condition from the patch.
    /// @tparam Model Physical model for this face.
    /// @param context Face geometry and time for sampling profiles.
    /// @param condition Condition assigned to the patch.
    /// @param inside Reconstructed interior trace, or the cell value after fallback.
    /// @param model Supplies the EOS and physical boundary policies.
    template<class R,class C,class Model>
    KOKKOS_INLINE_FUNCTION auto operator()(const boundary::FaceContext<R>& context, const C& condition,
                                           boundary::physical::Primitive<R,true> inside,
                                           boundary::physical::Primitive<R,true> face, const Model& model) const {
        namespace physical=boundary::physical;
        using Numeric=physical::condition_type_t<C>;
        if constexpr(detail::prepared_trace<Numeric>) return face;
        else if constexpr(detail::wall<Numeric>){
            const auto u=physical::sample_wall_velocity(condition, context);
            // The existing mechanical mirror is independent of wall thermal data or a flux-to-gradient law.
            return detail::evaluated_exterior(physical::SlipWall<R>{{static_cast<R>(u.x),static_cast<R>(u.y),static_cast<R>(u.z)}},
                context,inside,model,boundary::scalar::NoFluxLaw{},physical::Corrections<>{});
        } else return detail::evaluated_exterior(condition,context,inside,model,boundary::scalar::NoFluxLaw{},physical::Corrections<>{});
    }
};

/// @brief Re-evaluate the existing boundary law from the reconstructed interior trace.
/// Mechanical mirrors bypass thermal laws. This closure doesn't read prepared face slots.
/// @tparam Law Converts wall heat flux to a gradient for coupled custom conditions.
/// @tparam Correction Supplies corrections to the boundary evaluation.
template<class Law=boundary::scalar::NoFluxLaw,class Correction=boundary::physical::Corrections<>>
struct EvaluatedExterior {
    Law law{};
    Correction correction{};

    /// @brief Evaluates the boundary condition at the current interior trace.
    /// @param face Placeholder accepted for the closure interface.
    /// @tparam R Scalar type used for states and geometry.
    /// @tparam C Typed boundary condition from the patch.
    /// @tparam Model Physical model for this face.
    /// @param context Face geometry and time for sampling profiles.
    /// @param condition Condition assigned to the patch.
    /// @param inside Reconstructed interior trace, or the cell value after fallback.
    /// @param model Supplies the EOS and physical boundary policies.
    template<class R,class C,class Model>
    KOKKOS_INLINE_FUNCTION auto operator()(const boundary::FaceContext<R>& context, const C& condition,
                                           boundary::physical::Primitive<R,true> inside,
                                           boundary::physical::Primitive<R,true> face, const Model& model) const {
        using Numeric=boundary::physical::condition_type_t<C>;
        if constexpr (detail::wall<Numeric>||std::is_same_v<Numeric,boundary::physical::Symmetry>)
            return PreparedExterior{}(context,condition,inside,face,model);
        else return detail::evaluated_exterior(condition,context,inside,model,law,correction);
    }
};

namespace detail {

template<class Law, class Correction>
inline constexpr bool uses_prepared_face<EvaluatedExterior<Law, Correction>> = false;

} // namespace detail

template<class L>EvaluatedExterior(L)->EvaluatedExterior<L>;
template<class L,class C>EvaluatedExterior(L,C)->EvaluatedExterior<L,C>;
} // namespace krona::transport::euler
