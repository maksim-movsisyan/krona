#pragma once

#include <krona/fields/field_storage.hpp>

#include <memory>
#include <unordered_map>

namespace krona::fields {

/// @brief Host registry for typed fields; resolve names once before kernel dispatch.
/// Adding fields preserves existing views. Finish kernels/copies before destruction.
template<class Exec = execution::ExecutionSpace>
class FieldsManager {
public:
    explicit FieldsManager(FieldSizes sizes = {}, Exec exec = {})
        : sizes_(sizes), exec_(exec)
    {
        execution::require_kokkos();
        mpi::require(sizes_.own_cells <= sizes_.cells, "Fields: owned cells exceed total cells");
    }
    FieldsManager(const FieldsManager&) = delete;
    FieldsManager& operator=(const FieldsManager&) = delete;
    FieldsManager(FieldsManager&&) = delete;
    FieldsManager& operator=(FieldsManager&&) = delete;

    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    [[nodiscard]] FieldSizes sizes() const noexcept { return sizes_; }

    /// @brief Creates a mesh-associated field; zero initialization is queued by default.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    auto add_field(const std::string& name, FieldLocation location,
                   FieldInitialization initialization = FieldInitialization::Zero)
    {
        mpi::require(location != FieldLocation::Custom, "Custom fields require an explicit size");
        return add_field<T, Components, Layout>(name, entity_count(location), location, initialization);
    }

    /// @brief Accepts explicit sizes; mesh-associated counts must match FieldSizes.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    auto add_field(const std::string& name, std::size_t count, FieldLocation location,
                   FieldInitialization initialization = FieldInitialization::Zero)
    {
        if (name.empty()) detail::field_error(name, "empty field name");
        if (registry_.contains(name)) detail::field_error(name, "duplicate field");
        if (location != FieldLocation::Custom && count != entity_count(location))
            detail::field_error(name, "size does not match field location");
        mpi::require(initialization == FieldInitialization::Zero ||
                        initialization == FieldInitialization::Uninitialized,
                        "Fields: invalid initialization mode");
        using Storage = FieldStorage<T, Components, Exec, Layout>;
        auto field = std::make_unique<Storage>(name, count, location, exec_);
        auto* storage = field.get();
        // Commit ownership before dispatch: an insertion failure has no pending fill.
        registry_.emplace(name, std::move(field));
        if (initialization == FieldInitialization::Zero) {
            try { storage->fill(exec_, T{}); }
            catch (...) { exec_.fence("KRONA fields: drain failed initialization"); throw; }
        }
        return storage->view();
    }

    /// @brief Returns typed storage or nullptr on missing name/type/components/layout.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] auto* find_field(const std::string& name) noexcept
    {
        using Storage = FieldStorage<T, Components, Exec, Layout>;
        const auto it = registry_.find(name);
        return it == registry_.end() ? nullptr : dynamic_cast<Storage*>(it->second.get());
    }
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] const auto* find_field(const std::string& name) const noexcept
    {
        using Storage = FieldStorage<T, Components, Exec, Layout>;
        const auto it = registry_.find(name);
        return it == registry_.end() ? nullptr : dynamic_cast<const Storage*>(it->second.get());
    }

    /// @brief Strict mutable lookup; no name lookup or RTTI occurs inside kernels.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] auto get_field_view(const std::string& name)
    {
        auto* field = find_field<T, Components, Layout>(name);
        if (!field) detail::field_error(name, "missing field or type/components/layout mismatch");
        return field->view();
    }
    /// @brief A const manager exposes read-only values, including through data().
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] auto get_field_view(const std::string& name) const
    {
        const auto* field = find_field<T, Components, Layout>(name);
        if (!field) detail::field_error(name, "missing field or type/components/layout mismatch");
        return field->view();
    }

    /// @brief Enqueues a uniform fill of all entities and components.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    void fill_field(const std::string& name, const T& value)
    {
        auto* field = find_field<T, Components, Layout>(name);
        if (!field) detail::field_error(name, "invalid fill field or type/components/layout mismatch");
        field->fill(exec_, value);
    }

    /// @brief Returns immutable metadata or nullptr; stored names cannot be changed.
    [[nodiscard]] const AbstractField* get_field(const std::string& name) const noexcept
    {
        const auto it = registry_.find(name);
        return it == registry_.end() ? nullptr : it->second.get();
    }
    [[nodiscard]] bool has_field(const std::string& name) const noexcept { return registry_.contains(name); }
    [[nodiscard]] std::size_t get_field_size(const std::string& name) const noexcept
    {
        const auto* field = get_field(name);
        return field ? field->size() : 0;
    }

    /// @brief Allocates independent host storage; no values are copied or initialized.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] auto create_host_buffer(const std::string& name) const
    {
        const auto* field = find_field<T, Components, Layout>(name);
        if (!field) detail::field_error(name, "invalid host buffer field");
        return field->create_host_buffer();
    }

    /// @brief Enqueues a host snapshot; wait on execution_space() before reading it.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] auto copy_to_host_async(const std::string& name) const
    {
        auto source = get_field_view<T, Components, Layout>(name).kokkos_view();
        auto host = create_host_buffer<T, Components, Layout>(name);
        try { Kokkos::deep_copy(exec_, host, source); }
        catch (...) { exec_.fence("KRONA fields: drain failed host snapshot"); throw; }
        return host;
    }

    /// @brief Enqueues an upload; source must stay alive and unchanged until completion.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft,
             class HostView>
    void copy_from_host_async(const std::string& name, const HostView& source)
    {
        static_assert(Kokkos::is_view<HostView>::value);
        static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename HostView::memory_space>::accessible);
        static_assert(std::is_same_v<typename HostView::non_const_value_type, T>);
        auto destination = get_field_view<T, Components, Layout>(name).kokkos_view();
        static_assert(HostView::rank == decltype(destination)::rank);
        if (source.extent(0) != destination.extent(0) ||
            source.extent(1) != destination.extent(1))
            detail::field_error(name, "host upload shape mismatch");
        Kokkos::deep_copy(exec_, destination, source);
    }

    void fence() const
    {
        exec_.fence("KRONA fields: synchronization");
    }

private:
    FieldSizes sizes_;
    Exec exec_;
    std::unordered_map<std::string, std::unique_ptr<AbstractField>> registry_;

    [[nodiscard]] std::size_t entity_count(FieldLocation location) const noexcept
    {
        switch (location) {
        case FieldLocation::OwnCell: return sizes_.own_cells;
        case FieldLocation::Cell: return sizes_.cells;
        case FieldLocation::CellExtnd: return sizes_.cells + sizes_.bfaces;
        case FieldLocation::Face: return sizes_.faces;
        case FieldLocation::Node: return sizes_.nodes;
        case FieldLocation::Custom: break;
        }
        krona::mpi::fatal("Fields: invalid mesh-associated field location");
    }
};

} // namespace krona::fields
