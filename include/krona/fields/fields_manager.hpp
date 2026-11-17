#pragma once

#include <krona/fields/field_storage.hpp>

#include <memory>
#include <unordered_map>

namespace krona::fields {

/// @brief Owns named fields and looks them up on the host before kernels use their views.
/// @tparam Exec Kokkos execution space for the fields. Its memory space holds their values.
/// Adding fields preserves existing views. Complete kernels and copies before destroying the manager.
template<class Exec = execution::ExecutionSpace>
class FieldsManager {
public:
    /// @brief Creates an empty registry with fixed local entity counts.
    /// @param sizes Local counts used to size mesh fields. Total cells include owned cells and MPI ghosts.
    /// For CellExtnd, cells + bfaces must fit in std::size_t.
    /// @param exec Instance used for field allocations, fills and copies.
    explicit FieldsManager(FieldSizes sizes = {}, Exec exec = {}) : sizes_(sizes), exec_(exec) {
        execution::require_kokkos();
        mpi::require(sizes_.own_cells <= sizes_.cells, "Fields: owned cells exceed total cells");
    }

    FieldsManager(const FieldsManager&) = delete;
    FieldsManager& operator=(const FieldsManager&) = delete;
    FieldsManager(FieldsManager&&) = delete;
    FieldsManager& operator=(FieldsManager&&) = delete;

    /// @brief Returns the instance to use when queueing dependent kernels.
    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    /// @brief Returns a copy of the local counts the manager was created with.
    [[nodiscard]] FieldSizes sizes() const noexcept { return sizes_; }

    /// @brief Creates a mesh field using the count associated with its location.
    /// @tparam T Type of one stored value, for example StorageReal.
    /// @tparam Components Number of values per entity, for example 3 for velocity.
    /// @tparam Layout Kokkos layout of the field values.
    /// @param name Unique, nonempty name within this manager.
    /// @param location Mesh entries to allocate. Use the explicit-count overload for Custom.
    /// @param initialization Zero queues a fill with T{}. Uninitialized leaves the values unwritten.
    /// @return Writable view. Read initialized values after the fill completes or in later work
    /// queued on execution_space().
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    auto add_field_async(const std::string& name, FieldLocation location,
                   FieldInitialization initialization = FieldInitialization::Zero) {
        mpi::require(location != FieldLocation::Custom, "Custom fields require an explicit size");
        return add_field_async<T, Components, Layout>(name, entity_count(location), location, initialization);
    }

    /// @brief Creates a field with an explicit entity count.
    /// @tparam T Type of one stored value, for example StorageReal.
    /// @tparam Components Number of values per entity.
    /// @tparam Layout Kokkos layout of the field values.
    /// @param name Unique, nonempty name within this manager.
    /// @param count Number of entities. For mesh fields, it must match the count for location.
    /// @param location Kind of entry. For caller-defined entries, use Custom.
    /// @param initialization Zero queues a fill with T{}. Uninitialized leaves the values unwritten.
    /// @return Writable view that borrows the new field from this manager.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    auto add_field_async(const std::string& name, std::size_t count, FieldLocation location,
                   FieldInitialization initialization = FieldInitialization::Zero) {
        if (name.empty()) detail::field_error(name, "empty field name");
        if (registry_.contains(name)) detail::field_error(name, "duplicate field");
        if (location != FieldLocation::Custom && count != entity_count(location))
            detail::field_error(name, "size does not match field location");
        mpi::require(initialization == FieldInitialization::Zero ||
                     initialization == FieldInitialization::Uninitialized, "Fields: invalid initialization mode");
        using Storage = FieldStorage<T, Components, Exec, Layout>;
        auto field = std::make_unique<Storage>(name, count, location, exec_);
        auto* storage = field.get();
        // Register the storage before queueing a fill so the manager keeps the buffer alive.
        registry_.emplace(name, std::move(field));
        if (initialization == FieldInitialization::Zero) {
            try { storage->fill_async(exec_, T{}); }
            catch (...) { exec_.fence("KRONA fields: drain failed initialization"); throw; }
        }
        return storage->view();
    }

    /// @brief Finds storage when the name, stored type, component count and layout all match.
    /// @tparam T Registered scalar type, using the original non-const type.
    /// @tparam Components Registered number of values per entity.
    /// @tparam Layout Registered Kokkos layout.
    /// @param name Field name to look up on the host.
    /// @return Pointer to the matching storage, or nullptr.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] auto* find_field(const std::string& name) noexcept {
        using Storage = FieldStorage<T, Components, Exec, Layout>;
        const auto it = registry_.find(name);
        return it == registry_.end() ? nullptr : dynamic_cast<Storage*>(it->second.get());
    }
    /// @brief Same lookup as the mutable overload, with a pointer to const storage.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] const auto* find_field(const std::string& name) const noexcept {
        using Storage = FieldStorage<T, Components, Exec, Layout>;
        const auto it = registry_.find(name);
        return it == registry_.end() ? nullptr : dynamic_cast<const Storage*>(it->second.get());
    }

    /// @brief Returns a writable view, or aborts if the field is missing or its type does not match.
    /// @tparam T Registered scalar type.
    /// @tparam Components Registered number of values per entity.
    /// @tparam Layout Registered Kokkos layout.
    /// @param name Field name to resolve on the host before passing the view to a kernel.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] auto get_field_view(const std::string& name) {
        auto* field = find_field<T, Components, Layout>(name);
        if (!field) detail::field_error(name, "missing field or type/components/layout mismatch");
        return field->view();
    }
    /// @brief Looks up the field as above and returns a read-only view.
    /// Use the registered non-const scalar type for T, even through a const manager.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] auto get_field_view(const std::string& name) const {
        const auto* field = find_field<T, Components, Layout>(name);
        if (!field) detail::field_error(name, "missing field or type/components/layout mismatch");
        return field->view();
    }

    /// @brief Enqueues a uniform fill of all entities and components.
    /// @tparam T Registered scalar type. The fill value converts to this type when passed in.
    /// @tparam Components Registered number of values per entity.
    /// @tparam Layout Registered Kokkos layout.
    /// @param name Field to fill on this manager's execution-space instance.
    /// @param value Value to write to each entry.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    void fill_field_async(const std::string& name, const std::type_identity_t<T>& value) {
        auto* field = find_field<T, Components, Layout>(name);
        if (!field) detail::field_error(name, "invalid fill field or type/components/layout mismatch");
        field->fill_async(exec_, value);
    }

    /// @brief Returns a read-only pointer to the field's metadata, or nullptr if the name is missing.
    [[nodiscard]] const AbstractField* get_field(const std::string& name) const noexcept {
        const auto it = registry_.find(name);
        return it == registry_.end() ? nullptr : it->second.get();
    }
    [[nodiscard]] bool has_field(const std::string& name) const noexcept { return registry_.contains(name); }
    /// @brief Returns the entity count, or zero for a missing name. Use has_field to identify empty fields.
    [[nodiscard]] std::size_t get_field_size(const std::string& name) const noexcept {
        const auto* field = get_field(name);
        return field ? field->size() : 0;
    }

    /// @brief Allocates a separate, uninitialized host buffer with the field's shape and layout.
    /// @tparam T Registered scalar type.
    /// @tparam Components Registered number of values per entity.
    /// @tparam Layout Registered Kokkos layout.
    /// @param name Field whose shape and layout the buffer should match.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] auto create_host_buffer(const std::string& name) const {
        const auto* field = find_field<T, Components, Layout>(name);
        if (!field) detail::field_error(name, "invalid host buffer field");
        return field->create_host_buffer();
    }

    /// @brief Creates a host buffer and queues a copy of the field into it.
    /// @tparam T Registered scalar type.
    /// @tparam Components Registered number of values per entity.
    /// @tparam Layout Registered Kokkos layout.
    /// @param name Field to copy.
    /// @return Independent host buffer. Keep it alive and wait on this manager before accessing its values.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft>
    [[nodiscard]] auto copy_to_host_async(const std::string& name) const {
        auto source = get_field_view<T, Components, Layout>(name).kokkos_view();
        auto host = create_host_buffer<T, Components, Layout>(name);
        try { Kokkos::deep_copy(exec_, host, source); }
        catch (...) { exec_.fence("KRONA fields: drain failed host snapshot"); throw; }
        return host;
    }

    /// @brief Queues a copy from host-accessible storage into the field.
    /// @tparam T Registered scalar type.
    /// @tparam Components Registered number of values per entity.
    /// @tparam Layout Registered Kokkos layout.
    /// @tparam HostView Type of the source Kokkos view. Its scalar type and rank must match the field.
    /// @param name Destination field.
    /// @param source View with matching extents and a layout Kokkos can copy from.
    /// Keep its allocation alive and unchanged until the copy finishes.
    template<class T = StorageReal, int Components = 1, class Layout = Kokkos::LayoutLeft, class HostView>
    void copy_from_host_async(const std::string& name, const HostView& source) {
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

    /// @brief Waits for work queued on this manager's execution-space instance.
    void fence() const { exec_.fence("KRONA fields: synchronization"); }

private:
    FieldSizes sizes_;
    Exec exec_;
    std::unordered_map<std::string, std::unique_ptr<AbstractField>> registry_;

    [[nodiscard]] std::size_t entity_count(FieldLocation location) const noexcept {
        switch (location) {
        case FieldLocation::OwnCell: return sizes_.own_cells;
        case FieldLocation::Cell: return sizes_.cells;
        // The caller maps each boundary face to a slot after the owned and ghost cells.
        case FieldLocation::CellExtnd: return sizes_.cells + sizes_.bfaces;
        case FieldLocation::Face: return sizes_.faces;
        case FieldLocation::Node: return sizes_.nodes;
        case FieldLocation::Custom: break;
        }
        krona::mpi::fatal("Fields: invalid mesh-associated field location");
    }
};

} // namespace krona::fields
