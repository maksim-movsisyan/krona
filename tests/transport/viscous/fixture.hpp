#pragma once

#include "common/test_support.hpp"

#include <krona/thermodynamics/ideal_gas.hpp>
#include <krona/transport/viscous/boundary_jacobian.hpp>
#include <krona/transport/properties/properties.hpp>
#include <krona/transport/properties/thermal.hpp>
#include <krona/transport/properties/viscosity.hpp>

namespace viscous_test {

namespace vt = krona::transport::viscous;
namespace ph = krona::boundary::physical;
namespace props = krona::transport::properties;
namespace test = krona::test;
using Exec = test::Exec;
using R = krona::ComputeReal;
using A = krona::AccumulationReal;

inline constexpr auto properties = props::TransportProperties{
    props::ConstantViscosity<R>{R{2}}, props::ConstantConductivity<R>{R{3}}};
inline constexpr krona::thermodynamics::IdealGas<R> gas{static_cast<R>(1.4), R{1}};

inline double tolerance() {
    return 256 * static_cast<double>(Kokkos::epsilon_v<R>);
}

void rows(const Exec& exec);
void distributed(const Exec& exec);

} // namespace viscous_test
