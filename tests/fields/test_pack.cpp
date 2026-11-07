#include <common/test_support.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/fields/pack.hpp>

#include <array>
#include <type_traits>

namespace {
using namespace krona::test;
using namespace krona::fields;
using Space = typename Exec::memory_space;

/// @brief Packs scalar and vector fields without copying their SoA or AoS storage.
void fields(const Exec& exec) {
    // The pack flattens one scalar, three vector components and one component pair.
    FieldStorage<double, 1, Exec> scalar("scalar", 5, FieldLocation::Custom, exec);
    FieldStorage<double, 3, Exec, Kokkos::LayoutRight> vector("vector", 5, FieldLocation::Custom, exec);
    FieldStorage<double, 2, Exec> pair("pair", 5, FieldLocation::Custom, exec);
    const auto s = scalar.view();
    const auto v = vector.view();
    const auto p = pair.view();
    const auto packed = pack(s, v, p);

    // Component descriptors retain the original pointers, strides and reference types.
    static_assert(decltype(packed)::components == 6 && decltype(packed)::writable);
    static_assert(std::is_same_v<decltype(packed(0, 0)), double&>);
    static_assert(std::is_trivially_copyable_v<decltype(packed)>);
    equal(packed.size(), std::size_t{5}, "Common entity count");
    equal(packed.component<0>().data(), s.data(), "Pack borrows scalar storage");
    equal(packed.component<2>().data(), v.component(1).data(), "Pack borrows vector component");
    equal(packed.component<2>().stride(0), v.component(1).stride(0), "AoS stride is preserved");

    // Write through the pack, then inspect the three original allocations.
    auto hs = scalar.create_host_buffer();
    auto hv = vector.create_host_buffer();
    auto hp = pair.create_host_buffer();
    complete(exec, [&] {
        Kokkos::parallel_for(
            "test/pack_fields", Policy(exec, 0, 5), KOKKOS_LAMBDA(Index i) {
                const auto row = static_cast<std::size_t>(i);
                for (std::size_t c = 0; c < packed.components; ++c)
                    packed(row, c) = static_cast<double>(10 * row + c);
            });
        Kokkos::deep_copy(exec, hs, s.kokkos_view());
        Kokkos::deep_copy(exec, hv, v.kokkos_view());
        Kokkos::deep_copy(exec, hp, p.kokkos_view());
    });

    // Component offsets must follow argument order across field boundaries.
    for (std::size_t i = 0; i < 5; ++i) {
        equal(hs(i), static_cast<double>(10 * i), "Scalar is first");
        for (int c = 0; c < 3; ++c)
            equal(hv(i, c), static_cast<double>(10 * i + 1 + static_cast<std::size_t>(c)),
                  "Vector components follow");
        for (int c = 0; c < 2; ++c)
            equal(hp(i, c), static_cast<double>(10 * i + 4 + static_cast<std::size_t>(c)),
                  "Pair components are last");
    }
}

/// @brief Preserves exact types for component writes and a common type for mixed reads.
void mixed(const Exec& exec) {
    FieldStorage<float, 1, Exec> first("first", 3, FieldLocation::Custom, exec);
    FieldStorage<double, 2, Exec> middle("middle", 3, FieldLocation::Custom, exec);
    FieldStorage<int, 1, Exec> last("last", 3, FieldLocation::Custom, exec);
    const auto packed = pack(first.view(), middle.view(), last.view());

    // Runtime-indexed reads return a value; typed accessors retain writable references.
    static_assert(decltype(packed)::components == 4 && !decltype(packed)::homogeneous);
    static_assert(!decltype(packed)::writable);
    static_assert(std::is_same_v<decltype(packed(0, 0)), double>);
    static_assert(std::is_same_v<decltype(packed.get<0>(0)), float&>);
    static_assert(std::is_same_v<decltype(packed.get<3>(0)), int&>);
    static_assert(std::is_same_v<typename decltype(packed.component<3>())::value_type, int>);
    static_assert(std::is_trivially_copyable_v<decltype(packed)>);
    Kokkos::View<double*, Space> sums("pack/sums", 3);
    auto host_sums = Kokkos::create_mirror(sums);
    auto host_first = first.create_host_buffer();
    auto host_middle = middle.create_host_buffer();
    auto host_last = last.create_host_buffer();

    // Write each native type, then sum all components through runtime indexing.
    complete(exec, [&] {
        Kokkos::parallel_for(
            "test/pack_mixed", Policy(exec, 0, 3), KOKKOS_LAMBDA(Index i) {
                const auto row = static_cast<std::size_t>(i);
                packed.get<0>(row) = 1.5f;
                packed.get<1>(row) = 2.25;
                packed.component<2>()(row) = 3.5;
                packed.get<3>(row) = 4;
                double sum = 0;
                for (std::size_t c = 0; c < packed.components; ++c)
                    sum += packed(row, c);
                sums(row) = sum;
            });
        Kokkos::deep_copy(exec, host_sums, sums);
        Kokkos::deep_copy(exec, host_first, first.view().kokkos_view());
        Kokkos::deep_copy(exec, host_middle, middle.view().kokkos_view());
        Kokkos::deep_copy(exec, host_last, last.view().kokkos_view());
    });
    for (std::size_t i = 0; i < 3; ++i) {
        equal(host_sums(i), 11.25, "Mixed reads use the common numeric type");
        equal(host_first(i), 1.5f, "Float writes reach the original field");
        equal(host_middle(i, 0), 2.25, "First double component");
        equal(host_middle(i, 1), 3.5, "Second double component");
        equal(host_last(i), 4, "Integer writes reach the original field");
    }
}

/// @brief Checks per-component constness and read-only aliases of shared storage.
void read_only(const Exec& exec) {
    Kokkos::View<double*, Space> first("pack/first", 2), second("pack/second", 2);
    const auto packed = pack(first, second);
    const auto partially_const = pack(Kokkos::View<const double*, Space>{first}, second);

    // A partially const pack still allows typed writes to its mutable component.
    static_assert(std::is_same_v<decltype(partially_const(0, 0)), const double&>);
    static_assert(std::is_same_v<decltype(partially_const.get<0>(0)), const double&>);
    static_assert(std::is_same_v<decltype(partially_const.get<1>(0)), double&>);
    static_assert(!decltype(partially_const)::writable);

    // Freezing the whole pack removes writes without allocating new storage.
    const auto frozen = packed.as_const();
    static_assert(std::is_same_v<decltype(frozen(0, 0)), const double&>);
    static_assert(std::is_same_v<decltype(frozen.get<1>(0)), const double&>);
    static_assert(!std::is_assignable_v<decltype(frozen.component<1>()(0)), double>);
    equal(frozen.component<0>().data(), first.data(), "as_const does not copy");
    Kokkos::View<double*, Space> output("pack/output", 2);
    auto host = Kokkos::create_mirror(output);

    // The frozen alias must observe a later write through the mutable descriptor.
    complete(exec, [&] {
        Kokkos::deep_copy(exec, first, 2.0);
        Kokkos::deep_copy(exec, second, 3.0);
        Kokkos::parallel_for(
            "test/pack_const", Policy(exec, 0, 2), KOKKOS_LAMBDA(Index i) {
                const auto row = static_cast<std::size_t>(i);
                partially_const.get<1>(row) = 7.0;
                output(row) = frozen(row, 0) + frozen.get<1>(row);
            });
        Kokkos::deep_copy(exec, host, output);
    });
    for (std::size_t i = 0; i < 2; ++i)
        equal(host(i), 9.0, "Const descriptors observe shared data");
}

/// @brief Restricts all components to the shortest input and accepts empty packs.
void extents(const Exec& exec) {
    Kokkos::View<int*, Space> short_values("pack/short", 2), long_values("pack/long", 5);
    const auto packed = pack(short_values, long_values);
    equal(packed.size(), std::size_t{2}, "Pack exposes the minimum extent");
    equal(packed.component<1>().extent(0), std::size_t{2}, "Extracted component uses the pack extent");

    // A sentinel beyond the common extent detects writes to the longer input's tail.
    auto host = Kokkos::create_mirror(long_values);
    complete(exec, [&] {
        Kokkos::deep_copy(exec, short_values, 0);
        Kokkos::deep_copy(exec, long_values, -1);
        Kokkos::parallel_for(
            "test/pack_extents", Policy(exec, 0, 2),
            KOKKOS_LAMBDA(Index i) { packed(static_cast<std::size_t>(i), 1) = 8; });
        Kokkos::deep_copy(exec, host, long_values);
    });
    for (std::size_t i = 0; i < 5; ++i)
        equal(host(i), i < 2 ? 8 : -1, "Long component tail is unchanged");

    // An empty input also makes nested and read-only forms empty.
    Kokkos::View<int*, Space> empty;
    const auto zero = pack(empty, long_values);
    check(zero.empty() && zero.size() == 0, "One empty input makes an empty pack");
    check(zero.as_const().empty(), "Read-only empty pack");
    check(pack(zero).empty(), "An existing empty pack can be packed again");
    equal(pack(empty).size(), std::size_t{0}, "Null storage is valid for an empty input");
}

/// @brief Writes selected noncontiguous columns while preserving all other values.
void strided(const Exec& exec) {
    // Select columns in reverse order so both stride and component order matter.
    Kokkos::View<float**, Kokkos::LayoutRight, Space> data("pack/strided", 4, 6);
    const auto a = Kokkos::subview(data, Kokkos::ALL(), 1);
    const auto b = Kokkos::subview(data, Kokkos::ALL(), 4);
    const auto packed = pack(b, a);
    check(a.stride(0) > 1, "Input exercises nonunit stride");

    // Sentinels expose any write outside the two selected columns.
    auto host = Kokkos::create_mirror(data);
    complete(exec, [&] {
        Kokkos::deep_copy(exec, data, -1.0f);
        Kokkos::parallel_for(
            "test/pack_strided", Policy(exec, 0, 4), KOKKOS_LAMBDA(Index i) {
                const auto row = static_cast<std::size_t>(i);
                packed(row, 0) = 10.0f + static_cast<float>(i);
                packed.component<1>()(row) = 20.0f + static_cast<float>(i);
            });
        Kokkos::deep_copy(exec, host, data);
    });
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t c = 0; c < 6; ++c) {
            const float expected = c == 4   ? 10.0f + static_cast<float>(i)
                                   : c == 1 ? 20.0f + static_cast<float>(i)
                                            : -1.0f;
            equal(host(i, c), expected, "Only selected strided components are modified");
        }
}

/// @brief Flattens nested mixed-type packs without changing component order or aliases.
void nested(const Exec& exec) {
    FieldStorage<float, 2, Exec> field("pair", 3, FieldLocation::Custom, exec);
    Kokkos::View<int*, Space> flag("pack/flag", 4);

    // Adapt the pair first, then prepend a longer integer component.
    const auto single = field_pack(field.view());
    static_assert(decltype(single)::components == 2);
    const auto packed = pack(flag, single);
    static_assert(decltype(packed)::components == 3);
    equal(packed.size(), std::size_t{3}, "Nested pack retains the common extent");
    equal(packed.component<1>().data(), field.view().component(0).data(),
          "Nested descriptors preserve order");

    // Exact typed reads and common-type reads must survive nested const conversion.
    const auto frozen = packed.as_const();
    static_assert(std::is_same_v<decltype(frozen.get<0>(0)), const int&>);
    static_assert(std::is_same_v<decltype(frozen.get<1>(0)), const float&>);
    static_assert(std::is_same_v<decltype(frozen(0, 0)), float>);
    Kokkos::View<float*, Space> output("pack/nested_sum", 3);
    auto host = Kokkos::create_mirror(output);
    complete(exec, [&] {
        Kokkos::parallel_for(
            "test/pack_nested", Policy(exec, 0, 3), KOKKOS_LAMBDA(Index i) {
                const auto row = static_cast<std::size_t>(i);
                packed.get<0>(row) = 2;
                packed.get<1>(row) = 1.25f;
                packed.get<2>(row) = 2.5f;
                output(row) = frozen(row, 0) + frozen(row, 1) + frozen(row, 2);
            });
        Kokkos::deep_copy(exec, host, output);
    });
    for (std::size_t i = 0; i < 3; ++i)
        equal(host(i), 5.75f, "Nested mixed pack in a kernel");
}
} // namespace

/// @brief Runs the selected pack case through the shared test harness.
int main(int argc, char** argv) {
    const std::array cases{Case{"fields", fields},   Case{"mixed", mixed},     Case{"const", read_only},
                           Case{"extents", extents}, Case{"strided", strided}, Case{"nested", nested}};
    return krona::test::run(argc, argv, "fields.pack", cases);
}
