#include "common/test_support.hpp"

#include <krona/flow/conservative.hpp>

#include <Kokkos_NumericTraits.hpp>

#include <array>

namespace {

using namespace krona::test;
namespace flow = krona::flow;

void conservative(const Exec& exec) {
    Kokkos::View<bool*, Exec::memory_space> checks("flow conversion checks", 3);
    complete(exec, [&] {
        Kokkos::parallel_for(
            "test/flow_conservative", Policy(exec, 0, 1), KOKKOS_LAMBDA(Index) {
                using S = krona::ComputeReal;
                const auto q = flow::to_conservative(S{2}, krona::math::Vector3<S>{3, -4, 0}, S{-20});
                checks(0) = q.density == S{2} && q.momentum.x == S{6} && q.momentum.y == S{-8} &&
                            q.momentum.z == S{0} && q.total_energy_density == S{-15};

                // Keep the material's energy reference. A negative internal energy can be valid.
                flow::InternalEnergyState<S> result;
                checks(1) = flow::try_split_conservative(q, result) && result.density == S{2} &&
                            result.velocity.x == S{3} && result.velocity.y == S{-4} &&
                            result.velocity.z == S{0} && result.internal_energy == S{-20};

                const flow::Conservative<S> empty{};
                const flow::Conservative<S> invalid_momentum{S{1}, {Kokkos::quiet_NaN_v<S>, 0, 0}, S{1}};
                const flow::Conservative<S> overflow{S{1}, {Kokkos::finite_max_v<S>, 0, 0}, S{1}};
                checks(2) = !flow::try_split_conservative(empty, result) &&
                            !flow::try_split_conservative(invalid_momentum, result) &&
                            !flow::try_split_conservative(overflow, result) && result.density == S{2} &&
                            result.velocity.x == S{3} && result.internal_energy == S{-20};
            });
    });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, checks);
    check(host(0), "Conservative state uses momentum and total energy per volume");
    check(host(1), "Splitting recovers velocity and permits a shifted energy reference");
    check(host(2), "Invalid states leave the output unchanged");
}
} // namespace

int main(int argc, char** argv) {
    constexpr std::array cases{Case{"conservative", conservative}};
    return krona::test::run(argc, argv, "flow", cases);
}