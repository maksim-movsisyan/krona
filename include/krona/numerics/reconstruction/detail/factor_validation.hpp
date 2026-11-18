#pragma once

#include <krona/numerics/reconstruction/detail/validation.hpp>
#include <krona/numerics/reconstruction/schemes/limited.hpp>
#include <krona/numerics/reconstruction/types.hpp>

#include <string>
#include <type_traits>

namespace krona::numerics::reconstruction::detail {

template <class Formula>
constexpr bool is_limited = false;
template <class Formula, class Factors>
constexpr bool is_limited<Limited<Formula, Factors>> = true;

template <class Exec, class Formula, class Values>
void validate_unlimited_formula(const Formula& formula, const Values&, std::size_t cells) {
    static_assert(!is_limited<Formula>, "Prepare factors from the unlimited formula");
    validate_formula<Exec, Values::components>(formula, cells, [](const char* name) {
        return std::string("Reconstruction: ") + name + " extent is too short";
    });
}

template <class Exec, class Provider>
void require_factor_storage(const Provider& input, std::size_t extent, const char* message) {
    require_accessible<Exec, Provider>();
    mpi::require(input.size() >= extent, message);
}

template <class Exec, class Profiles, class Output, class Coupling>
CellRange validate_factor_launch(const Profiles& profiles, const Output& output, CellRange range, Coupling) {
    using Values = decltype(profiles.values);
    static_assert(Values::components > 0);
    static_assert(std::is_same_v<Coupling, Componentwise> || std::is_same_v<Coupling, SharedMinimum>);
    static_assert(Output::components == Values::components ||
                  (std::is_same_v<Coupling, SharedMinimum> && Output::components == 1));

    require_accessible<Exec, decltype(profiles.stencil)>();
    mpi::require(Kokkos::is_initialized() && !Kokkos::is_finalized(), "Reconstruction: inactive Kokkos");

    const auto cells = profiles.stencil.size();
    if (range.end == std::numeric_limits<std::size_t>::max())
        range.end = cells;
    constexpr auto bad_range = "Reconstruction: invalid range or input/output extents";
    mpi::require(range.begin <= range.end && range.end <= cells, bad_range);
    require_factor_storage<Exec>(profiles.values, Kokkos::max(cells, profiles.stencil.required_input_size()),
                                 bad_range);
    require_factor_storage<Exec>(output, cells, bad_range);
    return range;
}

} // namespace krona::numerics::reconstruction::detail
