#pragma once

#include <krona/boundary/physical/detail/tabulated_parameters.hpp>
#include <krona/boundary/physical/manager.hpp>

#include <utility>

namespace krona::boundary::physical {
namespace detail {

/// @brief Derives the manager alternatives from the same bind overloads used by the CSV loader.
template<class Model, class Exec, class Real, class... CustomConditions>
struct TabulatedManagerTypes {
    using Parameters = TabulatedParameters<Model, Exec, Real>;

    template<class... NumericConditions>
    using With = Manager<Model, Exec, Real,
        decltype(std::declval<Parameters&>().bind(std::declval<const NumericConditions&>()))...,
        CustomConditions...>;
};

} // namespace detail

/// @brief Manager with file-backed alternatives for the parameterized built-in conditions.
/// @tparam Model Flow model used to evaluate conditions and validate their compatibility.
/// @tparam Exec Kokkos execution space used for the profile arrays and boundary kernels.
/// @tparam Real Floating-point type of the samples and boundary calculations.
/// @tparam CustomConditions Extra condition types the application can bind in C++.
template<class Model, class Exec = execution::ExecutionSpace, class Real = ComputeReal, class... CustomConditions>
using TabulatedManager = typename detail::TabulatedManagerTypes<Model, Exec, Real, CustomConditions...>::template With<
    SlipWall<double>, AdiabaticWall<double>, IsothermalWall<double>, HeatFluxWall<double>,
    VelocityInlet<double>, TotalConditionsInlet<double>, SupersonicInlet<double>, PressureOutlet<double>>;

/// @brief Collectively binds the configuration and samples each CSV at local face centroids.
/// @tparam Model Flow model used by the manager.
/// @tparam Exec Kokkos execution space used to upload the profile arrays.
/// @tparam Real Floating-point type of one sample.
/// @tparam CustomConditions Manager alternatives, including the types produced by the CSV bindings.
/// @param manager Manager built on the layout for this host mesh, typically a TabulatedManager.
/// @param config Same configuration in the same patch order on every rank.
/// @param host Host mesh used to build the layout. It supplies the local face centroids.
/// @param comm Mesh communicator. Rank zero opens CSV paths relative to its working directory.
/// @details Complete previous boundary work before rebinding. Copies complete before return.
/// Call finalize() on the manager before submitting boundary work.
template<class Model, class Exec, class Real, class... CustomConditions>
void configure_tabulated(Manager<Model, Exec, Real, CustomConditions...>& manager, const Configuration& config,
                         const mesh::host::MeshPart& host, mpi::CommView comm) {
    mpi::require(!comm.is_null() && host.rank == comm.rank() && host.nprocs == comm.size(),
                 "Physical boundary: CSV configuration requires the mesh communicator");
    manager.configure(config, [&](const TabulatedConditionConfig& spec, PatchRange range, const auto& assign) {
        detail::TabulatedParameters<Model, Exec, Real> values{spec, range, host, comm, manager.execution_space()};
        std::visit([&](const auto& condition) {
            if constexpr (requires { values.bind(condition); })
                assign(values.bind(condition));
            else
                mpi::fatal("Physical boundary: condition does not accept CSV profiles");
        }, spec.numeric);
        values.require_complete();
    });
}

} // namespace krona::boundary::physical
