#pragma once

#include <krona/boundary/detail/config_binding.hpp>
#include <krona/boundary/detail/tabulated_values.hpp>
#include <krona/boundary/physical/config.hpp>
#include <krona/boundary/physical/detail/parameters.hpp>
#include <krona/boundary/physical/profiles.hpp>

#include <algorithm>
#include <utility>

namespace krona::boundary::physical::detail {

/// @brief Shared constant or sampled patch storage used for one scalar parameter.
/// @tparam Real Type of the stored values.
/// @tparam Space Kokkos memory space holding the samples.
template<class Real, class Space>
using TabulatedScalar = boundary::detail::TabulatedPatchValues<Real, Space>;

/// @brief Groups three independently configured scalar providers into a vector.
/// @tparam Real Floating-point type of the vector and stored samples.
/// @tparam Space Kokkos memory space holding any sampled components.
template<class Real, class Space>
struct TabulatedVector {
    TabulatedScalar<Real, Space> x, y, z;

    /// @brief Reads the three Cartesian components of this face's vector.
    /// @param context Face used to index any sampled array.
    KOKKOS_INLINE_FUNCTION Vector<Real> operator()(const FaceContext<Real>& context) const {
        return {x(context), y(context), z(context)};
    }

    /// @brief Checks that all three providers are usable on the assigned patch.
    /// @tparam Exec Execution space that will read the values.
    /// @param range Local face range assigned to this parameter.
    template<class Exec>
    void validate(PatchRange range) const {
        x.template validate<Exec>(range);
        y.template validate<Exec>(range);
        z.template validate<Exec>(range);
    }
};

/// @brief Builds numeric backflow data using the parameters of the configured mode.
/// @tparam Real Floating-point type of the condition.
/// @tparam Space Kokkos memory space of any profile arrays.
template<class Real, class Space>
struct TabulatedBackflow {
    BackflowMode mode = BackflowMode::Reject;
    TabulatedScalar<Real, Space> pressure, temperature;
    TabulatedVector<Real, Space> direction;
    bool allow_choking = false;

    /// @brief Evaluates the parameters required by the selected mode.
    /// @param context Current face and evaluation time.
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

    /// @brief Validates the individual providers once on the host.
    /// @tparam Exec Execution space that will read the providers.
    /// @param range Local patch to which the outlet is bound.
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

/// @brief Resolves each configured parameter to a constant or a sampled local array.
/// @tparam Model Flow model used to check the allowed backflow mode.
/// @tparam Exec Kokkos execution space uploading the local samples.
/// @tparam Real Arithmetic and storage type of the sampled data.
template<class Model, class Exec, class Real>
class TabulatedParameters {
public:
    using Space = typename Exec::memory_space;

    /// @brief Opens a binder for one patch. File reads occur when bind() asks for a parameter.
    /// @param spec Numeric defaults and the list of CSV overrides.
    /// @param range Local face interval for sampling.
    /// @param host Host mesh supplying local face centroids.
    /// @param comm Mesh communicator used by the profile loader.
    /// @param exec Execution instance receiving the sampled data.
    TabulatedParameters(const TabulatedConditionConfig& spec, PatchRange range, const mesh::host::MeshPart& host,
                        mpi::CommView comm, const Exec& exec)
        : spec_(spec), range_(range), host_(host), comm_(comm), exec_(exec) {}


    // Each binding lists the parameters its condition accepts. Separate
    // statements and braced initialization keep collective reads ordered.
    [[nodiscard]] auto bind(const SlipWall<double>& wall) {
        return slip_wall<Real>(vector(P::Velocity, wall.velocity));
    }

    [[nodiscard]] auto bind(const AdiabaticWall<double>& wall) {
        return adiabatic_wall<Real>(vector(P::Velocity, wall.velocity));
    }

    [[nodiscard]] auto bind(const IsothermalWall<double>& wall) {
        auto temperature = scalar(P::Temperature, wall.temperature);
        return isothermal_wall<Real>(std::move(temperature), vector(P::Velocity, wall.velocity));
    }

    [[nodiscard]] auto bind(const HeatFluxWall<double>& wall) {
        auto flux = scalar(P::Flux, wall.flux);
        return heat_flux_wall<Real>(std::move(flux), vector(P::Velocity, wall.velocity));
    }

    [[nodiscard]] auto bind(const VelocityInlet<double>& inlet) {
        auto velocity = vector(P::Velocity, inlet.velocity);
        return velocity_inlet<Real>(std::move(velocity), scalar(P::Temperature, inlet.temperature));
    }

    [[nodiscard]] auto bind(const TotalConditionsInlet<double>& inlet) {
        auto pressure = scalar(P::TotalPressure, inlet.total_pressure);
        auto temperature = scalar(P::TotalTemperature, inlet.total_temperature);
        auto direction = vector(P::Direction, inlet.direction);
        return total_conditions_inlet<Real>(std::move(pressure), std::move(temperature),
                                           std::move(direction), inlet.allow_choking);
    }

    [[nodiscard]] auto bind(const SupersonicInlet<double>& inlet) {
        return profiled<SupersonicInlet<Real>>(TabulatedSupersonic<Real, Space>{
            scalar(P::Pressure, inlet.pressure), scalar(P::Temperature, inlet.temperature),
            vector(P::Velocity, inlet.velocity), scalar(P::Mach, inlet.mach),
            vector(P::Direction, inlet.direction)});
    }

    [[nodiscard]] auto bind(const PressureOutlet<double>& outlet) {
        auto pressure = scalar(P::Pressure, outlet.pressure);
        return pressure_outlet<Real>(std::move(pressure), backflow(outlet.backflow));
    }

    /// @brief Checks that every requested CSV override was consumed exactly once.
    void require_complete() const {
        mpi::require(used_ == spec_.profiles.size(), "Physical boundary: unused or duplicate CSV profile parameter");
    }

private:
    using P = ProfileParameter;

    // Resolve one parameter by its meaning and vector component. Unmatched entries use the numeric value.
    [[nodiscard]] TabulatedScalar<Real, Space> scalar(ProfileParameter parameter, double constant,
                                                      std::size_t component = 0) {
        const auto found = std::ranges::find_if(spec_.profiles, [&](const ParameterProfile& profile) {
            return profile.parameter == parameter && profile.component == component;
        });
        const bool mach_required = parameter == ProfileParameter::Mach &&
                                   (constant > 0 || found != spec_.profiles.end());
        const auto convert = [&](double value) {
            const auto converted = boundary::detail::checked_parameter<Real>(value, "Physical boundary");
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

    [[nodiscard]] auto backflow(const Backflow<double>& data) {
        if (const char* issue = incompatible_backflow<Model>(data.mode))
            mpi::fatal(issue);
        TabulatedBackflow<Real, Space> result;
        result.mode = data.mode;
        if (data.mode == BackflowMode::TotalConditions) {
            result.pressure = scalar(P::TotalPressure, data.reservoir.total_pressure);
            result.temperature = scalar(P::TotalTemperature, data.reservoir.total_temperature);
            result.direction = vector(P::Direction, data.reservoir.direction);
            result.allow_choking = data.reservoir.allow_choking;
        } else if (data.mode == BackflowMode::DirectionTemperature) {
            result.direction = vector(P::Direction, data.direction);
            result.temperature = scalar(P::Temperature, data.temperature);
        }
        return profiled<Backflow<Real>>(result);
    }

    const TabulatedConditionConfig& spec_;
    PatchRange range_;
    const mesh::host::MeshPart& host_;
    mpi::CommView comm_;
    const Exec& exec_;
    std::size_t used_ = 0;
};

} // namespace krona::boundary::physical::detail
