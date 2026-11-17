#pragma once

#include <krona/execution/utils.hpp>
#include <krona/fields/fields_view.hpp>
#include <krona/mpi/error.hpp> 

#include <cstdio>
#include <limits>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

namespace krona::fields {

/// @brief Describes what each entry represents. FieldsManager uses it to choose the field length.
enum class FieldLocation {
    OwnCell,    ///< One entry per cell owned by this rank.
    Cell,       ///< Owned cells followed by MPI ghost cells.
    CellExtnd,  ///< Cell entries followed by one extra slot per boundary face.
    Face,       ///< One entry per local face, including partition interfaces and boundaries.
    Node,       ///< One entry per local node.
    Custom      ///< Caller-defined entries with an explicit count.
};

/// @brief Selects whether FieldsManager queues a fill with T{} or leaves the values untouched.
enum class FieldInitialization { Zero, Uninitialized };

/// @brief Counts on this rank, supplied by the caller when constructing a FieldsManager.
struct FieldSizes {
    std::size_t own_cells = 0; ///< Cells owned by this rank.
    std::size_t cells = 0;     ///< Total of owned and MPI ghost cells.
    std::size_t faces = 0;     ///< All local faces, including the boundary faces.
    std::size_t bfaces = 0;    ///< Boundary faces, used to size the tail of a CellExtnd field.
    std::size_t nodes = 0;     ///< All local nodes.
};

namespace detail {

[[noreturn]] inline void field_error(std::string_view name, const char* reason,
                                     std::source_location where = std::source_location::current()) noexcept {
    char message[512]{};
    const int length = static_cast<int>(name.size() < 256 ? name.size() : 256);
    std::snprintf(message, sizeof(message), "Fields: %s ('%.*s')", reason, length, name.data());
    krona::mpi::fatal(message, where);
}

} // namespace detail

/// @brief Common metadata for every field in the host registry.
class AbstractField {
public:
    AbstractField(std::string name, FieldLocation location) : name_(std::move(name)), location_(location) {}
    virtual ~AbstractField() = default;
    [[nodiscard]] std::string_view name() const noexcept { return name_; }
    [[nodiscard]] FieldLocation location() const noexcept { return location_; }
    /// @brief Returns the number of entities in the field.
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    /// @brief Returns the number of values per entity.
    [[nodiscard]] virtual int component_count() const noexcept = 0;
    /// @brief Returns the allocated value bytes including padding.
    [[nodiscard]] virtual std::size_t byte_size() const noexcept = 0;
private:
    std::string name_;
    FieldLocation location_;
};

/// @brief Owns the values for a single field in Exec::memory_space.
/// @tparam T Mutable, trivially copyable type of one value, such as float or int.
/// @tparam Components Number of values per entity, fixed at compile time.
/// @tparam Exec Kokkos execution space for allocation and fills. Its memory space holds the values.
/// @tparam Layout LayoutLeft groups values by component. LayoutRight groups them by entity.
/// Complete kernels and copies that use the field before destroying the storage.
template<class T, int Components = 1, class Exec = execution::ExecutionSpace, class Layout = Kokkos::LayoutLeft>
class FieldStorage final : public AbstractField {
    static_assert(Components > 0 && !std::is_const_v<T>);
    static_assert(std::is_trivially_copyable_v<T>, "Field values must be trivially copyable");
    static_assert(std::is_same_v<Layout, Kokkos::LayoutLeft> ||
                  std::is_same_v<Layout, Kokkos::LayoutRight>,
                  "Owned fields support LayoutLeft and LayoutRight");
public:
    using value_type = T;
    using memory_space = typename Exec::memory_space;
    using array_layout = Layout;
    using buffer_type = detail::FieldArray<value_type, Components, Kokkos::Device<Exec, memory_space>, array_layout, true>;
    using view_type = FieldView<value_type, Components, memory_space, array_layout>;
    using const_view_type = FieldView<const value_type, Components, memory_space, array_layout>;

    /// @brief Allocates the requested number of entities with uninitialized values.
    /// @param name Field name, also used as the Kokkos allocation label.
    /// @param count Number of entities, each with Components values.
    /// @param location Metadata describing the entries. With standalone storage the caller chooses count.
    /// @param exec Execution-space instance used to allocate the values.
    FieldStorage(std::string name, std::size_t count, FieldLocation location, const Exec& exec)
        : AbstractField(std::move(name), location) {
        execution::require_kokkos();
        constexpr std::size_t bytes_per_entity = sizeof(T) * static_cast<std::size_t>(Components);
        mpi::require(count <= std::numeric_limits<std::size_t>::max() / bytes_per_entity,
                     "Field allocation size overflows size_t");
        buffer_ = buffer_type(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string{this->name()}), count);
    }

    FieldStorage(const FieldStorage&) = delete;
    FieldStorage& operator=(const FieldStorage&) = delete;
    FieldStorage(FieldStorage&&) = delete;
    FieldStorage& operator=(FieldStorage&&) = delete;

    [[nodiscard]] std::size_t size() const noexcept override { return buffer_.extent(0); }
    [[nodiscard]] int component_count() const noexcept override { return Components; }
    /// @brief Returns the footprint of the value array, including any layout padding.
    [[nodiscard]] std::size_t byte_size() const noexcept override { return buffer_.span() * sizeof(T); }

    /// @brief Borrows a writable view. Keep this storage alive through completion of its users.
    [[nodiscard]] view_type view() & noexcept { return view_type{typename view_type::view_type{buffer_}}; }
    [[nodiscard]] view_type view() && = delete;

    /// @brief Borrows a read-only view of the same allocation.
    [[nodiscard]] const_view_type view() const & noexcept { return const_view_type{typename const_view_type::view_type{buffer_}}; }
    [[nodiscard]] const_view_type view() const && = delete;

    /// @brief Queues a fill of every entity and component.
    /// @param exec Instance used for the fill. Wait on it before reading values on the host
    /// or using them from another instance.
    /// @param value Value to write to each entry.
    void fill_async(const Exec& exec, const T& value)
    { Kokkos::deep_copy(exec, buffer_, value); }

    /// @brief Allocates an independent, uninitialized host buffer with matching layout.
    [[nodiscard]] auto create_host_buffer() const
    { return Kokkos::create_mirror(Kokkos::WithoutInitializing, buffer_); }
private:
    buffer_type buffer_;
};

} // namespace krona::fields
