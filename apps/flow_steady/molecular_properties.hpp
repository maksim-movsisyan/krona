#pragma once

#include <krona/transport/compressible/config.hpp>

#include <krona/thermodynamics/ideal_gas.hpp>
#include <krona/transport/properties/properties.hpp>
#include <krona/transport/properties/thermal.hpp>
#include <krona/transport/properties/viscosity.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace krona::apps::flow_steady {

namespace properties = transport::properties;
namespace ct = transport::compressible;

/// @brief Selects a viscosity correlation without duplicating the steady solver kernels.
template<class R>
struct Viscosity {
    ct::ViscosityModel kind;
    properties::ConstantViscosity<R> constant;
    properties::SutherlandViscosity<R> sutherland;

    void validate() const {
        if (kind == ct::ViscosityModel::Constant) constant.validate();
        else sutherland.validate();
    }

    template<class Real, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State& state) const {
        return kind == ct::ViscosityModel::Constant ? constant.template evaluate<Real>(state)
                                               : sutherland.template evaluate<Real>(state);
    }
};

/// @brief Selects constant conductivity or a constant Prandtl number.
template<class R>
struct Conductivity {
    ct::ConductivityModel kind;
    properties::ConstantConductivity<R> constant;
    properties::ConstantPrandtl<properties::ConstantHeatCapacity<R>, R> prandtl;

    void validate() const {
        if (kind == ct::ConductivityModel::Constant) constant.validate();
        else prandtl.validate();
    }

    template<class Real, class State>
    KOKKOS_INLINE_FUNCTION Real evaluate(const State& state, Real mu) const {
        return kind == ct::ConductivityModel::Constant ? constant.template evaluate<Real>(state, mu)
                                                  : prandtl.template evaluate<Real>(state, mu);
    }
};

template<class R>
auto molecular_properties(const ct::MolecularOptions& input, const thermodynamics::IdealGas<R>& gas) {
    const auto scalar = [](double value) {
        mpi::require(value >= 0 && value <= static_cast<double>(Kokkos::finite_max_v<R>),
                     "Steady flow: unrepresentable molecular property");
        const auto converted = static_cast<R>(value);
        mpi::require(value == 0 || converted > R{0}, "Steady flow: molecular property rounded to zero");
        return converted;
    };
    Viscosity<R> viscosity{input.viscosity, {}, {}};
    if (input.viscosity == ct::ViscosityModel::Constant)
        viscosity.constant = {scalar(input.dynamic_viscosity)};
    else
        viscosity.sutherland = {scalar(input.reference_viscosity), scalar(input.reference_temperature),
                                scalar(input.sutherland_temperature)};
    Conductivity<R> conductivity{input.conductivity, {}, {}};
    if (input.conductivity == ct::ConductivityModel::Constant)
        conductivity.constant = {scalar(input.thermal_conductivity)};
    else
        conductivity.prandtl = {scalar(input.prandtl), {gas.cp()}};
    return properties::TransportProperties{viscosity, conductivity};
}

} // namespace krona::apps::flow_steady
