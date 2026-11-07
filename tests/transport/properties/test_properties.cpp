#include <common/test_support.hpp>
#include <krona/transport/properties/properties.hpp>
#include <krona/thermodynamics/ideal_gas.hpp>

#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace p = krona::transport::properties;
using krona::test::Exec;
using krona::test::check;
using krona::test::near;

/// @brief Provides only the temperature member needed by thermal coefficient laws.
template<class R> struct Temperature { R temperature; };

/// @brief Proves that constant coefficient laws require no temperature member.
struct ColdState {};

/// @brief Compares a coefficient using a tolerance scaled by the reference magnitude.
template<class R>
void relative(R actual, R expected, R tolerance, std::string_view message)
{ near(actual, expected, tolerance*std::abs(expected), message); }
 
/// @brief Checks constant coefficients without thermal state access, including zero values.
void constants(const Exec&)
{
    const auto model = p::TransportProperties{p::ConstantViscosity{0.001}, p::ConstantConductivity{0.6}};
    model.validate();
    const auto result = model.evaluate<double>(ColdState{});
    check(result.valid(), "constant coefficients valid without thermal state");
    near(result.dynamic_viscosity, 0.001, 0., "dynamic viscosity");
    near(result.thermal_conductivity, 0.6, 0., "thermal conductivity");
    const auto zero = p::TransportProperties{p::ConstantViscosity{0.}, p::ConstantConductivity{0.}};
    zero.validate();
    check(zero(ColdState{}).valid(), "zero constant coefficients allowed");

    // Unused thermal data must not invalidate a constant-coefficient model.
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    check(model(Temperature<double>{nan}).valid(), "constants do not inspect unused temperature");
}
 
/// @brief Checks the viscosity curve, reference value and large-temperature arithmetic.
void sutherland(const Exec&)
{
    const auto law = p::SutherlandViscosity{1.716e-5, 273.15, 110.4};
    law.validate();
    relative(law.evaluate<double>(Temperature<double>{273.15}), 1.716e-5, 2.e-15, "reference viscosity");
    double previous = 0;
    for (double t : {150., 250., 300., 600., 1500.}) {
        // Independent direct expression from the model definition.
        const double expected = 1.716e-5*std::pow(t/273.15, 1.5)*(273.15+110.4)/(t+110.4);
        const double mu = law.evaluate<double>(Temperature<double>{t});
        relative(mu, expected, 3.e-15, "Sutherland curve");
        check(mu > previous, "positive monotonically increasing curve");
        previous = mu;
    }
    const auto no_shift = p::SutherlandViscosity{2., 4., 0.};
    no_shift.validate();
    near(no_shift.evaluate<double>(Temperature<double>{16.}), 4., 0., "S=0 gives square-root law");

    // A representable large-temperature result must survive intermediate arithmetic.
    const float high = law.evaluate<float>(Temperature<float>{1.e30f});
    const double expected = 1.716e-5*std::pow(1.e30/273.15, 1.5)*(273.15+110.4)/(1.e30+110.4);
    check(std::isfinite(high), "avoid overflow in T^(3/2) at representable high temperature");
    relative(static_cast<double>(high), expected, 5.e-7, "high-temperature arithmetic");
}

/// @brief Checks conductivity derived from viscosity, heat capacity and constant Prandtl number.
void prandtl(const Exec&)
{
    const auto cp = p::ConstantHeatCapacity{1005.};
    const auto law = p::ConstantPrandtl{0.72, cp};
    law.validate();
    const double mu = 1.8e-5;
    const double conductivity = law.evaluate<double>(ColdState{}, mu);
    relative(mu*1005./conductivity, 0.72, 2.e-15, "recover Pr from conductivity");
    near(law.evaluate<double>(ColdState{}, 0.), 0., 0., "zero viscosity gives zero conductivity");
    const auto combined = p::TransportProperties{p::ConstantViscosity{mu}, law};
    combined.validate();
    near(combined.evaluate<double>(ColdState{}).thermal_conductivity, conductivity, 0., "same combined result");
}

/// @brief Supplies an independent temperature-dependent heat-capacity policy.
struct LinearHeatCapacity {
    template<class R, class State>
    KOKKOS_INLINE_FUNCTION R evaluate(const State& state) const
    { return R{1000} + static_cast<R>(0.1)*(static_cast<R>(state.temperature)-R{300}); }
};

/// @brief Counts viscosity evaluations to detect duplicate policy calls.
struct CountViscosity {
    int* calls;
    template<class R, class State> KOKKOS_INLINE_FUNCTION R evaluate(const State&) const { ++*calls; return static_cast<R>(0.002); }
};

/// @brief Counts heat-capacity evaluations used by the conductivity policy.
struct CountHeatCapacity {
    int* calls;
    template<class R, class State> KOKKOS_INLINE_FUNCTION R evaluate(const State&) const { ++*calls; return R{1200}; }
};

/// @brief Checks custom heat-capacity policies and exactly one evaluation of each coefficient.
void custom(const Exec&)
{
    const auto law = p::TransportProperties{p::ConstantViscosity{2.e-5},
        p::ConstantPrandtl{0.8, LinearHeatCapacity{}}};
    law.validate(); // Custom policy intentionally has no validate(): no configurable parameters.
    const auto result = law.evaluate<double>(Temperature<double>{500.});
    relative(result.thermal_conductivity, 2.e-5*1020./0.8, 2.e-15, "state-dependent cp policy");

    // Reuse the computed viscosity when evaluating conductivity, rather than calling it twice.
    int viscosity_calls = 0, cp_calls = 0;
    const auto counted = p::TransportProperties{CountViscosity{&viscosity_calls},
        p::ConstantPrandtl{0.75, CountHeatCapacity{&cp_calls}}};
    const auto counted_result = counted.evaluate<double>(ColdState{});
    check(counted_result.valid(), "custom model result");
    check(viscosity_calls == 1 && cp_calls == 1, "viscosity and cp evaluated once");
    relative(counted_result.thermal_conductivity, 3.2, 2.e-15, "computed viscosity passed to conductivity");
}

/// @brief Checks explicit arithmetic types, promotion and the default ComputeReal selection.
void precision(const Exec&)
{
    const auto law = p::TransportProperties{p::SutherlandViscosity{1.716e-5, 273.15, 110.4},
        p::ConstantPrandtl{0.72, p::ConstantHeatCapacity{1005.}}};
    const auto value = law.evaluate<float>(Temperature<double>{300.});
    static_assert(std::is_same_v<std::remove_cv_t<decltype(value)>, p::Coefficients<float>>);
    const auto reference = law.evaluate<double>(Temperature<double>{300.});
    relative(static_cast<double>(value.dynamic_viscosity), reference.dynamic_viscosity, 5.e-7, "float evaluation");
    relative(static_cast<double>(value.thermal_conductivity), reference.thermal_conductivity, 5.e-7, "float conductivity");
    const auto stored_float = p::TransportProperties{p::ConstantViscosity{1.8e-5f}, p::ConstantConductivity{0.025f}};
    const auto promoted = stored_float.evaluate<double>(ColdState{});
    static_assert(std::is_same_v<std::remove_cv_t<decltype(promoted)>, p::Coefficients<double>>);
    const auto default_value = law(Temperature<double>{300.});
    static_assert(std::is_same_v<std::remove_cv_t<decltype(default_value)>, p::Coefficients<krona::ComputeReal>>);
    check(default_value.valid() && promoted.valid(), "default and promoted precision");
}

/// @brief Supplies prescribed invalid values for runtime result validation.
struct InvalidHeatCapacity {
    double value;
    template<class R, class State> KOKKOS_INLINE_FUNCTION R evaluate(const State&) const { return static_cast<R>(value); }
};

/// @brief Checks invalid temperatures, heat capacities and nonfinite or negative coefficients.
void invalid_states(const Exec&)
{
    const auto law = p::TransportProperties{p::SutherlandViscosity{1.716e-5, 273.15, 110.4},
        p::ConstantPrandtl{0.72, p::ConstantHeatCapacity{1005.}}};
    for (double t : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        check(!law.evaluate<double>(Temperature<double>{t}).valid(), "invalid temperature propagates");

    // Zero viscosity must not mask an invalid heat-capacity result.
    for (double cp : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        const auto bad = p::TransportProperties{p::ConstantViscosity{0.}, p::ConstantPrandtl{1., InvalidHeatCapacity{cp}}};
        check(!bad.evaluate<double>(ColdState{}).valid(), "invalid cp cannot hide behind zero viscosity");
    }
    check(!p::Coefficients<double>{-1., 0.}.valid(), "negative viscosity");
    check(!p::Coefficients<double>{0., -1.}.valid(), "negative conductivity");
    check(!p::Coefficients<double>{std::numeric_limits<double>::infinity(), 1.}.valid(), "nonfinite coefficient");
}

/// @brief Checks transport properties using a thermodynamic state and explicitly supplied EOS heat capacity.
void eos(const Exec&)
{
    const krona::thermodynamics::IdealGas<double> gas{};
    gas.validate();
    const auto law = p::TransportProperties{p::SutherlandViscosity{1.716e-5, 273.15, 110.4},
        p::ConstantPrandtl{0.72, p::ConstantHeatCapacity{gas.cp()}}};
    law.validate();
    const auto state = gas.from_pT(101325., 300.);
    const auto result = law.evaluate<double>(state);
    check(result.valid(), "existing thermodynamic state accepted");
    relative(result.dynamic_viscosity*gas.cp()/result.thermal_conductivity, 0.72, 2.e-15, "EOS cp used explicitly");
}

/// @brief Evaluates transport coefficients at a different temperature in each kernel entry.
template<class R, class Model, class Output>
struct EvaluateBatch {
    Model model;
    Output output;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const
    { output(i) = model.template evaluate<R>(Temperature<R>{R{250} + R{10}*static_cast<R>(i)}); }
};

/// @brief Checks completed device coefficients against independent viscosity and conductivity formulas.
template<class R, class Model>
void batch(const Exec& exec, Model model)
{
    constexpr int count = 37;
    using View = Kokkos::View<p::Coefficients<R>*, typename Exec::memory_space>;
    View output("transport/properties", count);
    krona::test::complete(exec, [&] {
        Kokkos::parallel_for("transport/properties/evaluate", Kokkos::RangePolicy<Exec>(exec, 0, count),
            EvaluateBatch<R, Model, View>{model, output});
    });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
    for (int i = 0; i < count; ++i) {
        check(host(i).valid(), "device coefficients valid");
        const double t = 250. + 10.*i;
        const double mu = 1.716e-5*std::pow(t/273.15, 1.5)*(273.15+110.4)/(t+110.4);
        const double cp = 1000. + 0.1*(t-300.);
        const double tolerance = 16.*static_cast<double>(Kokkos::epsilon_v<R>);
        relative(static_cast<double>(host(i).dynamic_viscosity), mu, tolerance, "device viscosity");
        relative(static_cast<double>(host(i).thermal_conductivity), mu*cp/0.72, tolerance, "device conductivity");
    }
}

/// @brief Exercises custom transport-property policies in float and double device kernels.
void device(const Exec& exec)
{
    const auto model = p::TransportProperties{p::SutherlandViscosity{1.716e-5, 273.15, 110.4},
        p::ConstantPrandtl{0.72, LinearHeatCapacity{}}};
    model.validate();
    batch<float>(exec, model);
    batch<double>(exec, model);
}

/// @brief Rejects a negative configured viscosity.
void reject_mu(const Exec&) { p::ConstantViscosity{-1.}.validate(); }

/// @brief Rejects a negative configured conductivity.
void reject_k(const Exec&) { p::ConstantConductivity{-1.}.validate(); }

/// @brief Rejects zero configured heat capacity.
void reject_cp(const Exec&) { p::ConstantHeatCapacity{0.}.validate(); }

/// @brief Rejects zero configured Prandtl number.
void reject_pr(const Exec&) { p::ConstantPrandtl{0., p::ConstantHeatCapacity{1005.}}.validate(); }

/// @brief Rejects a nonpositive reference viscosity.
void reject_sutherland_mu(const Exec&) { p::SutherlandViscosity{0., 273.15, 110.4}.validate(); }

/// @brief Rejects a nonpositive reference temperature.
void reject_sutherland_temperature(const Exec&) { p::SutherlandViscosity{1.716e-5, 0., 110.4}.validate(); }

/// @brief Rejects a negative Sutherland shift.
void reject_sutherland_shift(const Exec&) { p::SutherlandViscosity{1.716e-5, 273.15, -1.}.validate(); }

// Simulate validation in an external/user policy which throws instead of using MPI helpers.
/// @brief Emulates an external policy whose validator throws instead of using MPI helpers.
struct ThrowingValidation {
    const char* message;
    void validate() const { throw std::runtime_error(message); }
};

/// @brief Checks that an exception from custom viscosity validation becomes a fatal diagnostic.
void reject_custom_viscosity(const Exec&)
{
    const auto model = p::TransportProperties{
        ThrowingValidation{"custom viscosity validation failed"}, p::ConstantConductivity{1.}};
    static_assert(noexcept(model.validate()));
    model.validate();
    check(false, "throwing viscosity validator unexpectedly returned");
}
 
/// @brief Checks that an exception from custom heat-capacity validation becomes a fatal diagnostic.
void reject_custom_cp(const Exec&)
{
    const auto model = p::ConstantPrandtl{0.72, ThrowingValidation{"custom heat capacity validation failed"}};
    static_assert(noexcept(model.validate()));
    model.validate();
    check(false, "throwing heat-capacity validator unexpectedly returned");
}

/// @brief Runs property cases or one isolated configuration-rejection case.
int main(int argc, char** argv)
{
    const std::array cases{
        krona::test::Case{"constants", constants}, krona::test::Case{"sutherland", sutherland},
        krona::test::Case{"prandtl", prandtl}, krona::test::Case{"custom", custom},
        krona::test::Case{"precision", precision}, krona::test::Case{"invalid_states", invalid_states},
        krona::test::Case{"eos", eos}, krona::test::Case{"device", device}};
    const std::array rejection_cases{
        krona::test::Case{"reject_mu", reject_mu}, krona::test::Case{"reject_k", reject_k},
        krona::test::Case{"reject_cp", reject_cp}, krona::test::Case{"reject_pr", reject_pr},
        krona::test::Case{"reject_sutherland_mu", reject_sutherland_mu},
        krona::test::Case{"reject_sutherland_temperature", reject_sutherland_temperature},
        krona::test::Case{"reject_sutherland_shift", reject_sutherland_shift},
        krona::test::Case{"reject_custom_viscosity", reject_custom_viscosity},
        krona::test::Case{"reject_custom_cp", reject_custom_cp}};
    if (argc == 2 && std::string_view(argv[1]).starts_with("reject_"))
        return krona::test::run(argc, argv, "transport.properties", rejection_cases);
    return krona::test::run(argc, argv, "transport.properties", cases);
}
