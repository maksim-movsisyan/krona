#include "common/test_support.hpp"
#include <krona/fields/transfer.hpp>

#include <array>
#include <cstdint>

namespace {
using namespace krona::test;
namespace f = krona::fields;
using Space = typename Exec::memory_space;

/// @brief Round-trips a mixed field prefix through strided DOFs for either field layout.
template <class Layout> void roundtrip(const Exec& exec) {
    // Twelve DOFs hold three four-component cells; the fields have two extra cells.
    Kokkos::View<float*, Space> rho("rho", 5), rho_out("rho_out", 5);
    Kokkos::View<double* [3], Layout, Space> u("u", 5), u_out("u_out", 5);
    Kokkos::View<double* [2], Kokkos::LayoutRight, Space> storage("dofs", 12);
    const auto dofs = Kokkos::subview(storage, Kokkos::ALL(), 1);

    // Sentinels detect writes to the unused DOF column and the destination field tails.
    Kokkos::deep_copy(exec, storage, -19.0);
    Kokkos::deep_copy(exec, rho_out, -7.0f);
    Kokkos::deep_copy(exec, u_out, -8.0);
    Kokkos::parallel_for(
        "test/transfer_input", Kokkos::RangePolicy<Exec>(exec, 0, 5), KOKKOS_LAMBDA(int i) {
            rho(i) = static_cast<float>(100 * i);
            for (int c = 0; c < 3; ++c)
                u(i, c) = static_cast<double>(100 * i + c + 1);
        });

    // Combine a scalar view and a vector field into a read-only input pack.
    using U = f::FieldView<double, 3, Space, Layout>;
    const U velocity(typename U::view_type(u.data(), 5));
    const U velocity_out(typename U::view_type(u_out.data(), 5));
    Kokkos::View<const float*, Space> readonly = rho;
    const auto input = f::pack(readonly, velocity.as_const());
    const auto output = f::pack(rho_out, velocity_out);

    // Queue both transfers on one instance; wait once before reading their results.
    f::copy_to_dofs_async(exec, input, dofs);
    f::copy_from_dofs_async(exec, dofs, output);
    exec.fence("test/transfer complete");
    const auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, storage);
    const auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rho_out);
    const auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u_out);

    // DOFs use cell-major order, and only the selected field prefix is overwritten.
    for (std::size_t i = 0; i < 12; ++i) {
        equal(hs(i, 0), -19.0, "Unselected DOF storage is preserved");
        equal(hs(i, 1), static_cast<double>(100 * (i / 4) + i % 4), "Cell-major DOF order and stride");
    }
    for (std::size_t i = 0; i < 5; ++i) {
        equal(hr(i), i < 3 ? static_cast<float>(100 * i) : -7.0f, "Density prefix and extended tail");
        for (int c = 0; c < 3; ++c)
            equal(hu(i, c), i < 3 ? static_cast<double>(100 * i + static_cast<std::size_t>(c) + 1) : -8.0,
                  "Velocity prefix and extended tail");
    }
}

/// @brief Preserves integers beyond double's exact range when transferring a mixed pack.
void exact_types(const Exec& exec) {
    // 2^53 + 1 would lose precision if an integer component passed through double.
    constexpr std::int64_t big = 9007199254740993LL;
    Kokkos::View<std::int64_t*, Space> integer("integer", 2), target("integer_out", 2),
        dofs("integer_dofs", 4);
    Kokkos::View<float*, Space> real("float", 2), real_out("float_out", 2);
    Kokkos::deep_copy(exec, integer, big);
    Kokkos::deep_copy(exec, real, 3.0f);

    // Blocking transfers also exercise nested packs and conversion of the float component.
    const auto input = f::pack(integer, f::pack(real));
    f::copy_to_dofs(exec, input.as_const(), dofs);
    f::copy_from_dofs(exec, dofs, f::pack(target, real_out));
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, target);
    const auto r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, real_out);
    for (int i = 0; i < 2; ++i) {
        equal(h(i), big, "Each source/destination uses its exact scalar type");
        equal(r(i), 3.0f, "Mixed destination conversion");
    }
}

/// @brief Treats zero-length DOFs as an empty prefix and preserves the destination.
void empty(const Exec& exec) {
    Kokkos::View<double*, Space> dofs("empty", 0), values("values", 3);
    Kokkos::deep_copy(exec, values, 7.0);

    // The blocking return transfer completes earlier work on the same instance.
    f::copy_to_dofs_async(exec, values, dofs);
    f::copy_from_dofs(exec, dofs, values);
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
    for (int i = 0; i < 3; ++i)
        equal(h(i), 7.0, "Empty transfer preserves destination");
}

/// @brief Rejects a DOF count that is not divisible by the pack's component count.
void mismatch(const Exec& exec) {
    Kokkos::View<double*, Space> values("values", 3), dofs("wrong_width", 5);
    f::copy_to_dofs_async(exec, f::pack(values, values), dofs);
}

/// @brief Rejects packing more cells than the source field contains.
void short_input(const Exec& exec) {
    Kokkos::View<double*, Space> values("values", 1), dofs("dofs", 2);
    f::copy_to_dofs_async(exec, values, dofs);
}

/// @brief Rejects unpacking more cells than the destination field contains.
void short_output(const Exec& exec) {
    Kokkos::View<double*, Space> values("values", 1), dofs("dofs", 2);
    f::copy_from_dofs_async(exec, dofs, values);
}
} // namespace

/// @brief Runs transfer cases; the CMake wrapper checks fatal-contract diagnostics.
int main(int argc, char** argv) {
    constexpr std::array cases{Case{"left", roundtrip<Kokkos::LayoutLeft>},
                               Case{"right", roundtrip<Kokkos::LayoutRight>},
                               Case{"types", exact_types},
                               Case{"empty", empty},
                               Case{"mismatch", mismatch},
                               Case{"short_input", short_input},
                               Case{"short_output", short_output}};
    return krona::test::run(argc, argv, "fields.transfer", cases);
}
