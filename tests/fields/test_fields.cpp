#include "../common/test_support.hpp"

#include <krona/fields/fields_manager.hpp>

#include <array>
#include <type_traits>
#include <utility>

namespace {

using namespace krona::test;
using namespace krona::fields;
 
/// @brief Completes a manager snapshot before returning host-readable values.
template<class T = Real, int Components = 1, class Layout = Kokkos::LayoutLeft>
auto host_copy(FieldsManager<>& fields, const std::string& name)
{
    auto host = fields.copy_to_host_async<T, Components, Layout>(name);
    complete(fields.execution_space(), [] {});
    return host;
}

/// @brief Locations, typed lookup and zero initialization follow the registry contract.
void registry(const Exec& exec)
{
    FieldsManager<> fields({.own_cells = 2, .cells = 3, .faces = 7, .nodes = 6}, exec);
    complete(exec, [&] {
        fields.add_field("owned", FieldLocation::OwnCell);
        fields.add_field("density", FieldLocation::Cell);
        fields.add_field<Real, 3>("velocity", FieldLocation::Cell);
        fields.add_field<int>("face_flag", FieldLocation::Face);
        fields.add_field("node_value", FieldLocation::Node);
        fields.add_field("scratch", 4, FieldLocation::Custom);
    });
    equal(fields.get_field_size("owned"), std::size_t{2}, "Owned field size");
    equal(fields.get_field_size("density"), std::size_t{3}, "Cell size includes ghosts");
    equal(fields.get_field_size("face_flag"), std::size_t{7}, "Local face count");
    equal(fields.get_field_size("node_value"), std::size_t{6}, "Local node count");
    equal(fields.get_field_size("scratch"), std::size_t{4}, "Explicit custom count");
    check(fields.has_field("density") && !fields.has_field("missing"), "Name lookup");
    check(fields.get_field("missing") == nullptr, "Missing metadata lookup is optional");
    check(fields.find_field<Real>("missing") == nullptr, "Missing typed lookup is optional");
    check(fields.find_field<int>("density") == nullptr, "Wrong value type is rejected");
    check(fields.find_field<Real, 2>("velocity") == nullptr, "Wrong component count is rejected");
    check(fields.find_field<Real, 3, Kokkos::LayoutRight>("velocity") == nullptr, "Wrong layout is rejected");
    equal(fields.get_field("velocity")->component_count(), 3, "Component metadata");
    check(fields.get_field("velocity")->location() == FieldLocation::Cell, "Location metadata");

    const auto read_only = std::as_const(fields).get_field_view<Real>("density");
    static_assert(std::is_same_v<decltype(read_only.data()), const Real*>);
    static_assert(!std::is_copy_constructible_v<FieldsManager<>>);
    static_assert(!std::is_move_constructible_v<FieldsManager<>>);
    static_assert(decltype(read_only)::view_type::memory_traits::is_unmanaged);
    static_assert(!std::is_assignable_v<decltype(read_only[0]), Real>);
    const auto density = fields.get_field_view<Real>("density");
    static_assert(std::is_assignable_v<decltype(density[0]), Real>);
    static_assert(std::is_convertible_v<decltype(density), decltype(read_only)>);
    static_assert(!std::is_convertible_v<decltype(read_only), decltype(density)>);
    equal(density.as_const().data(), read_only.data(), "Read-only conversion aliases the field");

    auto host = host_copy(fields, "density");
    for (std::size_t cell = 0; cell < host.extent(0); ++cell)
        equal(host(cell), Real{0}, "Default initialization includes all cell slots");
    auto flags = host_copy<int>(fields, "face_flag");
    for (std::size_t face = 0; face < flags.extent(0); ++face)
        equal(flags(face), 0, "Integer fields are zero-initialized too");
}

/// @brief Two-component SoA views survive registry growth and work in ordered kernels.
void soa(const Exec& exec)
{
    FieldsManager<> fields({.own_cells = 5, .cells = 5}, exec);
    auto state = fields.add_field<Real, 2>("state", FieldLocation::Cell,
                                         FieldInitialization::Uninitialized);
    auto component = state.component(1);
    const auto* original = state.data();
    for (int field = 0; field < 64; ++field)
        fields.add_field<int>("unrelated_" + std::to_string(field), 2,
                              FieldLocation::Custom, FieldInitialization::Uninitialized);
    equal(state.data(), original, "Adding a field does not invalidate cached views");
    equal(fields.get_field_view<Real, 2>("state").data(), original,
          "Registry growth preserves the original field allocation");
    equal(state.size(), std::size_t{5}, "FieldView size counts entities");
    equal(state.kokkos_view().size(), std::size_t{10}, "Kokkos View size counts scalar values");
    equal(component.stride(0), std::size_t{1}, "A fixed SoA component is contiguous");
    static_assert(std::is_same_v<typename decltype(state)::array_layout, Kokkos::LayoutLeft>);

    auto host = fields.create_host_buffer<Real, 2>("state");
    complete(exec, [&] {
        Kokkos::parallel_for("test/soa_write", Policy(exec, 0, 5), KOKKOS_LAMBDA(Index cell) {
            for (int c = 0; c < 2; ++c) state(static_cast<std::size_t>(cell), c) = static_cast<Real>(10 * cell + c);
        });
        Kokkos::parallel_for("test/component_write", Policy(exec, 0, 5), KOKKOS_LAMBDA(Index cell) {
            component(cell) += Real{100};
        });
        Kokkos::deep_copy(exec, host, state.kokkos_view());
    });
    for (std::size_t cell = 0; cell < 5; ++cell) {
        equal(host(cell, 0), static_cast<Real>(10 * cell), "First component is unchanged");
        equal(host(cell, 1), static_cast<Real>(10 * cell + 101), "Subview writes the second component");
    }
}

/// @brief Uploads, fills and independent snapshots preserve all components.
void transfers(const Exec& exec)
{
    FieldsManager<> fields({.own_cells = 4, .cells = 4}, exec);
    auto velocity = fields.add_field<Real, 3>("velocity", FieldLocation::Cell,
                                            FieldInitialization::Uninitialized);
    auto upload = fields.create_host_buffer<Real, 3>("velocity");
    for (std::size_t cell = 0; cell < 4; ++cell)
        for (int c = 0; c < 3; ++c)
            upload(cell, c) = static_cast<Real>(10 * cell + static_cast<std::size_t>(c));

    // Both operations use the manager's instance, with no fence between them.
    complete(exec, [&] {
        fields.copy_from_host_async<Real, 3>("velocity", upload);
        Kokkos::parallel_for("test/after_upload", Policy(exec, 0, 4), KOKKOS_LAMBDA(Index cell) {
            for (int c = 0; c < 3; ++c) velocity(static_cast<std::size_t>(cell), c) *= Real{2};
        });
    });
    auto snapshot = host_copy<Real, 3>(fields, "velocity");
    for (std::size_t cell = 0; cell < 4; ++cell)
        for (int c = 0; c < 3; ++c)
            equal(snapshot(cell, c), Real{2} * upload(cell, c), "Upload is visible to the next kernel");

    // Host buffers are separate allocations in CPU builds as well.
    snapshot(0, 0) = Real{99};
    auto unchanged = host_copy<Real, 3>(fields, "velocity");
    equal(unchanged(0, 0), Real{0}, "Editing a snapshot does not edit the field");
    complete(exec, [&] {
        fields.fill_field<Real, 3>("velocity", Real{7});
        Kokkos::deep_copy(exec, upload, velocity.kokkos_view());
    });
    equal(snapshot(0, 0), Real{99}, "A later fill does not edit the snapshot");
    for (std::size_t cell = 0; cell < 4; ++cell)
        for (int c = 0; c < 3; ++c)
            equal(upload(cell, c), Real{7}, "Fill updates every entity and component");
}

/// @brief Vector aliases and full tensors use their documented component ordering.
void tensor(const Exec& exec)
{
    FieldsManager<> fields({.own_cells = 2, .cells = 3}, exec);
    auto velocity = fields.add_field<Real, 3>("velocity", FieldLocation::Cell,
                                            FieldInitialization::Uninitialized);
    auto stress = fields.add_field<Real, 9>("stress", FieldLocation::OwnCell,
                                          FieldInitialization::Uninitialized);
    auto host_velocity = fields.create_host_buffer<Real, 3>("velocity");
    auto host_stress = fields.create_host_buffer<Real, 9>("stress");
    complete(exec, [&] {
        Kokkos::parallel_for("test/vector", Policy(exec, 0, 3), KOKKOS_LAMBDA(Index cell) {
            velocity.x(static_cast<std::size_t>(cell)) = static_cast<Real>(cell + 1);
            velocity.y(static_cast<std::size_t>(cell)) = static_cast<Real>(cell + 11);
            velocity.z(static_cast<std::size_t>(cell)) = static_cast<Real>(cell + 21);
        });
        Kokkos::parallel_for("test/tensor", Policy(exec, 0, 2), KOKKOS_LAMBDA(Index cell) {
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    stress(static_cast<std::size_t>(cell), row, column) = static_cast<Real>(100 * cell + 10 * row + column);
        });
        Kokkos::deep_copy(exec, host_velocity, velocity.kokkos_view());
        Kokkos::deep_copy(exec, host_stress, stress.kokkos_view());
    });
    equal(host_velocity(2, 0), Real{3}, "x maps to component zero");
    equal(host_velocity(2, 1), Real{13}, "y maps to component one");
    equal(host_velocity(2, 2), Real{23}, "z maps to component two");
    for (std::size_t cell = 0; cell < 2; ++cell)
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                equal(host_stress(cell, row * 3 + column),
                      static_cast<Real>(100 * cell + static_cast<std::size_t>(10 * row + column)),
                      "Full tensor uses row-major component numbering");
}

/// @brief Empty fields remain valid registry entries and accept empty kernel ranges.
void empty(const Exec& exec)
{
    FieldsManager<> fields({}, exec);
    complete(exec, [&] {
        fields.add_field("density", FieldLocation::Cell);
        fields.add_field<Real, 3>("velocity", FieldLocation::Node);
        fields.add_field<int>("scratch", 0, FieldLocation::Custom);
    });
    const auto density = fields.get_field_view<Real>("density");
    check(density.empty(), "Zero-sized field has an empty view");
    check(fields.has_field("scratch"), "An empty custom field still exists");
    auto host = host_copy(fields, "density");
    equal(host.extent(0), std::size_t{0}, "Empty host snapshot has zero entities");
    auto host_velocity = fields.create_host_buffer<Real, 3>("velocity");
    equal(host_velocity.extent(0), std::size_t{0}, "Empty vector has no entities");
    equal(host_velocity.extent(1), std::size_t{3}, "Empty vector retains its component count");
    complete(exec, [&] {
        Kokkos::parallel_for("test/empty_field", Policy(exec, 0, 0), KOKKOS_LAMBDA(Index cell) {
            density(static_cast<std::size_t>(cell)) = Real{1};
        });
        fields.fill_field<Real>("density", Real{2});
        fields.fill_field<Real, 3>("velocity", Real{3});
        fields.copy_from_host_async<Real, 3>("velocity", host_velocity);
    });
    check(density.empty(), "Empty operations preserve the view shape");
}

/// @brief Fill values convert to the selected field type without deducing it.
void fill_types(const Exec& exec)
{
    using Default = krona::StorageReal;
    FieldsManager<> fields({.own_cells = 3, .cells = 4}, exec);
    fields.add_field("scalar", FieldLocation::Cell);
    fields.add_field<int>("integer", FieldLocation::Cell);
    fields.add_field<float>("single", FieldLocation::Cell);
    fields.add_field<double>("double", FieldLocation::Cell);

    const double const_value = 3.5;
    double mutable_value = 4.5;
    fields.fill_field("scalar", 0); // An int literal must not select an int field.
    auto zeros = fields.copy_to_host_async<>("scalar");
    fields.fill_field("scalar", 2.5); // Also exercises double-to-float in FP32 builds.
    auto literals = fields.copy_to_host_async<>("scalar");
    fields.fill_field("scalar", const_value);
    auto constants = fields.copy_to_host_async<>("scalar");
    fields.fill_field("scalar", mutable_value);
    auto variables = fields.copy_to_host_async<>("scalar");
    fields.fill_field("scalar", Default{5});
    auto temporaries = fields.copy_to_host_async<>("scalar");
    fields.fill_field<int>("integer", 7.0);
    fields.fill_field<float>("single", 1.25);
    fields.fill_field<double>("double", 6);
    auto integers = fields.copy_to_host_async<int>("integer");
    auto singles = fields.copy_to_host_async<float>("single");
    auto doubles = fields.copy_to_host_async<double>("double");
    fields.fence();

    for (std::size_t cell = 0; cell < 4; ++cell) {
        equal(zeros(cell), Default{0}, "An integer literal uses the default field type");
        equal(literals(cell), Default{2.5}, "A double literal converts to the default field type");
        equal(constants(cell), Default{3.5}, "Fill accepts a const value");
        equal(variables(cell), Default{4.5}, "Fill accepts a mutable value");
        equal(temporaries(cell), Default{5}, "Fill accepts a temporary value");
        equal(integers(cell), 7, "Explicit integer fill converts its argument");
        equal(singles(cell), 1.25f, "Explicit float fill converts its argument");
        equal(doubles(cell), 6.0, "Explicit double fill converts its argument");
    }
}

/// @brief AoS raw storage, component subviews and read-only views agree on indexing.
void aos(const Exec& exec)
{
    using Layout = Kokkos::LayoutRight;
    FieldsManager<> fields({.own_cells = 5, .cells = 5}, exec);
    auto velocity = fields.add_field<Real, 3, Layout>(
        "velocity", FieldLocation::Cell, FieldInitialization::Uninitialized);
    auto stress = fields.add_field<Real, 9, Layout>(
        "stress", FieldLocation::Cell, FieldInitialization::Uninitialized);
    auto z = velocity.component(2);
    const auto* original = velocity.data();
    const FieldView<const Real, 3, typename Exec::memory_space, Layout> read_only = velocity;
    auto* raw = velocity.data();
    static_assert(std::is_same_v<typename decltype(velocity)::array_layout, Layout>);
    static_assert(decltype(z)::memory_traits::is_unmanaged);
    static_assert(std::is_same_v<decltype(read_only.component(1).data()), const Real*>);
    equal(velocity.kokkos_view().stride(0), std::size_t{3}, "AoS entity stride equals component count");
    equal(velocity.kokkos_view().stride(1), std::size_t{1}, "AoS components are contiguous");
    equal(z.stride(0), std::size_t{3}, "A fixed AoS component retains the entity stride");
    equal(z.extent(0), std::size_t{5}, "A component subview keeps all entities");
    equal(read_only.data(), original, "Read-only AoS conversion shares the allocation");

    auto host_velocity = fields.create_host_buffer<Real, 3, Layout>("velocity");
    auto host_stress = fields.create_host_buffer<Real, 9, Layout>("stress");
    complete(exec, [&] {
        Kokkos::parallel_for("test/aos_raw_write", Policy(exec, 0, 5), KOKKOS_LAMBDA(Index cell) {
            for (int c = 0; c < 3; ++c)
                raw[3 * static_cast<std::size_t>(cell) + static_cast<std::size_t>(c)] =
                    static_cast<Real>(10 * cell + c);
        });
        Kokkos::parallel_for("test/aos_component", Policy(exec, 0, 5), KOKKOS_LAMBDA(Index cell) {
            velocity.x(static_cast<std::size_t>(cell)) += Real{1};
            velocity.y(static_cast<std::size_t>(cell)) += Real{2};
            z(cell) += Real{100};
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    stress(static_cast<std::size_t>(cell), row, column) =
                        read_only(static_cast<std::size_t>(cell), column) + static_cast<Real>(10 * row);
        });
        Kokkos::deep_copy(exec, host_velocity, velocity.kokkos_view());
        Kokkos::deep_copy(exec, host_stress, stress.kokkos_view());
    });
    for (std::size_t cell = 0; cell < 5; ++cell) {
        const std::array expected{
            static_cast<Real>(10 * cell + 1), static_cast<Real>(10 * cell + 3),
            static_cast<Real>(10 * cell + 102)};
        for (int c = 0; c < 3; ++c)
            equal(host_velocity(cell, c), expected[static_cast<std::size_t>(c)],
                  "AoS accessors and strided component writes address the right elements");
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                equal(host_stress(cell, row * 3 + column),
                      expected[static_cast<std::size_t>(column)] + static_cast<Real>(10 * row),
                      "AoS tensors retain logical row-major component numbering");
    }

    complete(exec, [&] { fields.fill_field<Real, 3, Layout>("velocity", 8); });
    auto filled = host_copy<Real, 3, Layout>(fields, "velocity");
    for (std::size_t cell = 0; cell < 5; ++cell)
        for (int c = 0; c < 3; ++c)
            equal(filled(cell, c), Real{8}, "AoS fill covers every component");
}

/// @brief Extended fields include distinct owned, ghost and appended boundary slots.
void extended(const Exec& exec)
{
    FieldsManager<> fields({.own_cells = 2, .cells = 4, .faces = 9, .bfaces = 3, .nodes = 6}, exec);
    auto scalar = fields.add_field("extended", FieldLocation::CellExtnd);
    fields.add_field("cells", FieldLocation::Cell);
    fields.add_field("explicit_extended", 7, FieldLocation::CellExtnd);
    equal(scalar.size(), std::size_t{7}, "Extended fields include all cells and boundary faces");
    equal(fields.get_field_size("cells"), std::size_t{4}, "Cell fields exclude appended boundary entries");
    equal(fields.get_field_size("explicit_extended"), std::size_t{7}, "Explicit extended size matches FieldSizes");
    equal(fields.sizes().bfaces, std::size_t{3}, "Boundary-face count is retained");
    check(fields.get_field("extended")->location() == FieldLocation::CellExtnd,
          "Extended location is retained in metadata");
    auto initial = host_copy(fields, "extended");
    for (std::size_t i = 0; i < 7; ++i)
        equal(initial(i), Real{0}, "Default initialization includes appended boundary entries");

    auto component = scalar.component(0);
    equal(component.data(), scalar.data(), "A scalar component aliases the scalar field");
    complete(exec, [&] {
        Kokkos::parallel_for("test/extended_write", Policy(exec, 0, 7), KOKKOS_LAMBDA(Index i) {
            scalar[static_cast<std::size_t>(i)] = static_cast<Real>(i + 1);
            scalar(static_cast<std::size_t>(i)) += Real{10};
            component(i) += Real{20};
        });
        Kokkos::parallel_for("test/extended_boundary", Policy(exec, 4, 7), KOKKOS_LAMBDA(Index i) {
            scalar(static_cast<std::size_t>(i), 0) += Real{100};
        });
    });
    auto populated = host_copy(fields, "extended");
    for (std::size_t i = 0; i < 7; ++i)
        equal(populated(i), static_cast<Real>(i + 31 + (i >= 4 ? 100 : 0)),
              "Boundary updates preserve the owned and ghost cell portions");

    complete(exec, [&] { fields.fill_field("extended", -2); });
    auto filled = host_copy(fields, "extended");
    for (std::size_t i = 0; i < 7; ++i)
        equal(filled(i), Real{-2}, "Fill includes owned, ghost and boundary entries");
}

/// @brief Standalone storage owns data and exposes independent host buffers.
void storage(const Exec& exec)
{
    using Storage = FieldStorage<Real, 3>;
    static_assert(!std::is_copy_constructible_v<Storage>);
    static_assert(!std::is_move_constructible_v<Storage>);
    Storage field("standalone", 4, FieldLocation::Custom, exec);
    auto values = field.view();
    auto read_only = std::as_const(field).view();
    static_assert(std::is_same_v<decltype(read_only.data()), const Real*>);
    equal(values.data(), read_only.data(), "Const storage exposes the same allocation read-only");
    check(field.name() == "standalone", "Standalone name is retained");
    check(field.location() == FieldLocation::Custom, "Standalone location is retained");
    equal(field.size(), std::size_t{4}, "Standalone size counts entities");
    equal(field.component_count(), 3, "Standalone component metadata");
    check(field.byte_size() >= std::size_t{12} * sizeof(Real), "Byte size covers all scalar values");

    auto host = field.create_host_buffer();
    check(host.data() != values.data(), "A host buffer is independent even with host field storage");
    complete(exec, [&] {
        field.fill(exec, Real{4});
        Kokkos::parallel_for("test/standalone_storage", Policy(exec, 0, 4), KOKKOS_LAMBDA(Index i) {
            values.y(static_cast<std::size_t>(i)) += static_cast<Real>(i);
        });
        Kokkos::deep_copy(exec, host, read_only.kokkos_view());
    });
    for (std::size_t i = 0; i < 4; ++i) {
        equal(host(i, 0), Real{4}, "Standalone fill initializes x");
        equal(host(i, 1), Real{4} + static_cast<Real>(i), "Standalone views support ordered writes");
        equal(host(i, 2), Real{4}, "Standalone fill initializes z");
    }
}

} // namespace

int main(int argc, char** argv)
{
    const std::array cases{
        Case{"registry", registry}, Case{"soa", soa}, Case{"transfers", transfers},
        Case{"tensor", tensor}, Case{"empty", empty}, Case{"fill_types", fill_types},
        Case{"aos", aos}, Case{"extended", extended}, Case{"storage", storage}};
    return krona::test::run(argc, argv, "fields", cases);
}