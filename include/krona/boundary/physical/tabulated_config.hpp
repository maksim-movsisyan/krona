#pragma once

#include <krona/boundary/detail/tabulated_values.hpp>
#include <krona/boundary/physical/manager.hpp>

#include <algorithm>

namespace krona::boundary::physical {
namespace detail {

template<class Real, class Space>
using TabulatedScalar = boundary::detail::TabulatedPatchValues<Real, Space>;

template<class Real, class Space>
struct TabulatedVector {
    TabulatedScalar<Real, Space> x, y, z;

    KOKKOS_INLINE_FUNCTION Vector<Real> operator()(const FaceContext<Real>& context) const {
        return {x(context), y(context), z(context)};
    }

    template<class Exec>
    void validate(PatchRange range) const {
        x.template validate<Exec>(range);
        y.template validate<Exec>(range);
        z.template validate<Exec>(range);
    }
};

template<class Real, class Space>
struct TabulatedBackflow {
    BackflowMode mode = BackflowMode::Reject;
    TabulatedScalar<Real, Space> pressure, temperature;
    TabulatedVector<Real, Space> direction;
    bool allow_choking = false;

    KOKKOS_INLINE_FUNCTION Backflow<Real> operator()(const FaceContext<Real>& context) const {
        Backflow<Real> result;
        result.mode = mode;
        if (mode == BackflowMode::TotalConditions) {
            result.reservoir = {pressure(context), temperature(context), direction(context), allow_choking};
        } else if (mode == BackflowMode::DirectionTemperature) {
            result.direction = direction(context);
            result.temperature = temperature(context);
        }
        return result;
    }

    template<class Exec>
    void validate(PatchRange range) const {
        pressure.template validate<Exec>(range);
        temperature.template validate<Exec>(range);
        direction.template validate<Exec>(range);
    }
};

template<class Real, class Space>
using TabulatedSupersonic = SupersonicParameters<Real, TabulatedScalar<Real, Space>, TabulatedScalar<Real, Space>,
    TabulatedVector<Real, Space>, TabulatedScalar<Real, Space>, TabulatedVector<Real, Space>>;

template<class Model, class Exec, class Real, class Scalar, class VectorValue, class... CustomConditions>
using TabulatedManagerImpl = Manager<Model, Exec, Real,
    decltype(slip_wall<Real>(VectorValue{})),
    decltype(adiabatic_wall<Real>(VectorValue{})),
    decltype(isothermal_wall<Real>(Scalar{}, VectorValue{})),
    decltype(heat_flux_wall<Real>(Scalar{}, VectorValue{})),
    decltype(velocity_inlet<Real>(VectorValue{}, Scalar{})),
    decltype(total_conditions_inlet<Real>(Scalar{}, Scalar{}, VectorValue{})),
    Profiled<SupersonicInlet<Real>, TabulatedSupersonic<Real, typename Exec::memory_space>>,
    decltype(pressure_outlet<Real>(Scalar{}, profiled<Backflow<Real>>(TabulatedBackflow<Real, typename Exec::memory_space>{}))),
    CustomConditions...>;

/// @brief Resolves each configured parameter to a constant or a sampled local array.
template<class Exec, class Real>
class TabulatedParameters {
public:
    using Space = typename Exec::memory_space;

    TabulatedParameters(const TabulatedConditionConfig& spec, PatchRange range, const mesh::host::MeshPart& host,
                        mpi::CommView comm, const Exec& exec)
        : spec_(spec), range_(range), host_(host), comm_(comm), exec_(exec) {}

    [[nodiscard]] TabulatedScalar<Real, Space> scalar(ProfileParameter parameter, double constant,
                                                      std::size_t component = 0) {
        const auto found = std::ranges::find_if(spec_.profiles, [&](const ParameterProfile& profile) {
            return profile.parameter == parameter && profile.component == component;
        });
        const bool mach_required = parameter == ProfileParameter::Mach &&
                                   (constant > 0 || found != spec_.profiles.end());
        const auto convert = [&](double value) {
            mpi::require(std::isfinite(value) &&
                         std::abs(value) <= static_cast<double>(std::numeric_limits<Real>::max()),
                         "Physical boundary: parameter is not representable in the selected precision");
            const auto converted = static_cast<Real>(value);
            mpi::require(!mach_required || converted > Real{0},
                         "Physical boundary: prescribed Mach number must be positive");
            return converted;
        };
        if (found == spec_.profiles.end()) {
            TabulatedScalar<Real, Space> result;
            result.constant = convert(constant);
            return result;
        }
        ++used_;
        return boundary::detail::load_patch_values<Real>(found->path, range_, host_, comm_, exec_, convert);
    }

    [[nodiscard]] TabulatedVector<Real, Space> vector(ProfileParameter parameter, Vector<double> constant) {
        return {scalar(parameter, constant.x, 0), scalar(parameter, constant.y, 1),
                scalar(parameter, constant.z, 2)};
    }

    template<class Model>
    [[nodiscard]] auto backflow(const Backflow<double>& data) {
        using P = ProfileParameter;
        TabulatedBackflow<Real, Space> result;
        result.mode = data.mode;
        if (data.mode == BackflowMode::TotalConditions) {
            mpi::require(Model::compressible, "Physical boundary: incompressible backflow uses DIRECTION_TEMPERATURE");
            result.pressure = scalar(P::TotalPressure, data.reservoir.total_pressure);
            result.temperature = scalar(P::TotalTemperature, data.reservoir.total_temperature);
            result.direction = vector(P::Direction, data.reservoir.direction);
            result.allow_choking = data.reservoir.allow_choking;
        } else if (data.mode == BackflowMode::DirectionTemperature) {
            mpi::require(!Model::compressible, "Physical boundary: compressible backflow requires total reservoir conditions");
            result.direction = vector(P::Direction, data.direction);
            result.temperature = scalar(P::Temperature, data.temperature);
        }
        return profiled<Backflow<Real>>(result);
    }

    void require_complete() const {
        mpi::require(used_ == spec_.profiles.size(), "Physical boundary: unused or duplicate CSV profile parameter");
    }

private:
    const TabulatedConditionConfig& spec_;
    PatchRange range_;
    const mesh::host::MeshPart& host_;
    mpi::CommView comm_;
    const Exec& exec_;
    std::size_t used_ = 0;
};

} // namespace detail

/// @brief Physical manager with file-backed alternatives for all parameterized conditions.
template<class Model, class Exec = execution::ExecutionSpace, class Real = ComputeReal, class... CustomConditions>
using TabulatedManager = detail::TabulatedManagerImpl<Model, Exec, Real,
    detail::TabulatedScalar<Real, typename Exec::memory_space>,
    detail::TabulatedVector<Real, typename Exec::memory_space>, CustomConditions...>;

/// @brief Collectively binds physical conditions and samples CSVs at local face centroids.
/// Use identical configurations on all ranks and the host mesh used to build the layout.
/// Relative paths use rank 0's working directory. Copies complete before return.
/// Complete prior boundary kernels before rebinding, and finalize before dispatch.
template<class Model, class Exec, class Real, class... CustomConditions>
void configure_tabulated(Manager<Model, Exec, Real, CustomConditions...>& manager, const Configuration& config,
                         const mesh::host::MeshPart& host, mpi::CommView comm) {
    using P = ProfileParameter;
    using Space = typename Exec::memory_space;
    mpi::require(!comm.is_null() && host.rank == comm.rank() && host.nprocs == comm.size(),
                 "Physical boundary: CSV configuration requires the mesh communicator");
    manager.configure(config, [&](const TabulatedConditionConfig& spec, PatchRange range, const auto& assign) {
        detail::TabulatedParameters<Exec, Real> values{spec, range, host, comm, manager.execution_space()};
        std::visit([&](const auto& c) {
            using C = std::remove_cvref_t<decltype(c)>;
            // Separate statements fix the order of collective reads for conditions with multiple files.
            if constexpr (std::is_same_v<C, SlipWall<double>>) {
                assign(slip_wall<Real>(values.vector(P::Velocity, c.velocity)));
            } else if constexpr (std::is_same_v<C, AdiabaticWall<double>>) {
                assign(adiabatic_wall<Real>(values.vector(P::Velocity, c.velocity)));
            } else if constexpr (std::is_same_v<C, IsothermalWall<double>>) {
                auto temperature = values.scalar(P::Temperature, c.temperature);
                assign(isothermal_wall<Real>(std::move(temperature), values.vector(P::Velocity, c.velocity)));
            } else if constexpr (std::is_same_v<C, HeatFluxWall<double>>) {
                auto flux = values.scalar(P::Flux, c.flux);
                assign(heat_flux_wall<Real>(std::move(flux), values.vector(P::Velocity, c.velocity)));
            } else if constexpr (std::is_same_v<C, VelocityInlet<double>>) {
                auto velocity = values.vector(P::Velocity, c.velocity);
                assign(velocity_inlet<Real>(std::move(velocity), values.scalar(P::Temperature, c.temperature)));
            } else if constexpr (std::is_same_v<C, TotalConditionsInlet<double>>) {
                auto pressure = values.scalar(P::TotalPressure, c.total_pressure);
                auto temperature = values.scalar(P::TotalTemperature, c.total_temperature);
                auto direction = values.vector(P::Direction, c.direction);
                assign(total_conditions_inlet<Real>(std::move(pressure), std::move(temperature),
                                                    std::move(direction), c.allow_choking));
            } else if constexpr (std::is_same_v<C, SupersonicInlet<double>>) {
                assign(profiled<SupersonicInlet<Real>>(detail::TabulatedSupersonic<Real, Space>{
                    values.scalar(P::Pressure, c.pressure), values.scalar(P::Temperature, c.temperature),
                    values.vector(P::Velocity, c.velocity), values.scalar(P::Mach, c.mach),
                    values.vector(P::Direction, c.direction)}));
            } else if constexpr (std::is_same_v<C, PressureOutlet<double>>) {
                auto pressure = values.scalar(P::Pressure, c.pressure);
                assign(pressure_outlet<Real>(std::move(pressure), values.template backflow<Model>(c.backflow)));
            } else {
                mpi::fatal("Physical boundary: condition does not accept CSV profiles");
            }
        }, spec.numeric);
        values.require_complete();
    });
}

} // namespace krona::boundary::physical
