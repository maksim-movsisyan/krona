#include "../common/test_support.hpp"

#include <krona/fields/fields_manager.hpp>

#include <array>
#include <type_traits>

namespace {

using namespace krona::test;
using namespace krona::fields;
 
/// @brief Completes a manager snapshot before returning host-readable values.
template<class T = Real, int Components = 1>
auto host_copy(FieldsManager<>& fields, const std::string& name)
{
    auto host = fields.copy_to_host_async<T, Components>(name);
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
    const auto density = fields.get_field_view<Real>("density");
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
    fields.add_field<int>("unrelated", 2, FieldLocation::Custom, FieldInitialization::Uninitialized);
    equal(state.data(), original, "Adding a field does not invalidate cached views");
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
    complete(exec, [&] {
        Kokkos::parallel_for("test/empty_field", Policy(exec, 0, 0), KOKKOS_LAMBDA(Index cell) {
            density(static_cast<std::size_t>(cell)) = Real{1};
        });
        fields.fill_field<Real>("density", Real{2});
    });
    check(density.empty(), "Empty operations preserve the view shape");
}

} // namespace

int main(int argc, char** argv)
{
    const std::array cases{
        Case{"registry", registry}, Case{"soa", soa}, Case{"transfers", transfers},
        Case{"tensor", tensor}, Case{"empty", empty}};
    return krona::test::run(argc, argv, "fields", cases);
}
