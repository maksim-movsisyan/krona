#pragma once

#include <chrono>

namespace krona::solvers::flow::detail {

/// @brief Measures phase boundaries on the host when iteration timing is enabled.
class IterationTimer {
    using Clock = std::chrono::steady_clock;
public:
    explicit IterationTimer(bool active)
        : start_(active ? Clock::now() : Clock::time_point{}), previous_(start_), active_(active) {}

    [[nodiscard]] bool active() const noexcept { return active_; }

    [[nodiscard]] double lap() {
        if (!active_) return 0;
        const auto now = Clock::now();
        const auto elapsed = std::chrono::duration<double>(now - previous_).count();
        previous_ = now;
        return elapsed;
    }

    [[nodiscard]] double total() const {
        return active_ ? std::chrono::duration<double>(Clock::now() - start_).count() : 0;
    }

private:
    Clock::time_point start_, previous_;
    bool active_;
};

} // namespace krona::solvers::flow::detail
