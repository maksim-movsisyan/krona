#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>

#include <type_traits>

namespace krona::numerics::reconstruction::detail {

// A custom formula can expose borrowed inputs to the same host checks as built-in formulas.
template <class Formula, class Visitor>
void visit_inputs(const Formula& formula, Visitor&& visit) {
    if constexpr (requires { formula.visit_inputs(visit); })
        formula.visit_inputs(visit);
}

template <class Exec, class Provider>
constexpr void require_accessible() {
    if constexpr (requires { typename Provider::memory_space; }) {
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Provider::memory_space>::accessible,
                      "Reconstruction: provider memory is inaccessible from this execution space");
    }
}

template <class Exec, class Provider, class Message>
void validate_extent(const Provider& provider, std::size_t count, const Message& diagnostic) {
    require_accessible<Exec, Provider>();
    if constexpr (requires { provider.size(); })
        mpi::require(provider.size() >= count, diagnostic);
}

template <class Exec, std::size_t Components, class Formula, class Diagnostic>
void validate_formula(const Formula& formula, std::size_t cells, const Diagnostic& diagnostic) {
    visit_inputs(formula, [&](const char* name, const auto& provider, bool shared = false) {
        using Provider = std::remove_cvref_t<decltype(provider)>;
        if constexpr (requires { Provider::components; }) {
            // A limiter factor may be shared across variables. Other packs follow value order.
            mpi::require(Provider::components == Components || (shared && Provider::components == 1),
                         "Reconstruction: field component counts do not match");
        }
        require_accessible<Exec, Provider>();
        if constexpr (requires { provider.size(); }) {
            if (provider.size() < cells)
                mpi::fatal(diagnostic(name));
        }
    });
}

} // namespace krona::numerics::reconstruction::detail
