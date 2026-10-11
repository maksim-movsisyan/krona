#pragma once

#include <krona/fields/fields_view.hpp>
#include <krona/mpi/error.hpp> 

#include <cstdio>
#include <limits>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

namespace krona::fields {

enum class FieldLocation { OwnCell, Cell, CellExtnd, Face, Node, Custom };
enum class FieldInitialization { Zero, Uninitialized };

/// @brief Local entity counts; Cell includes owned and MPI ghost cells only.
struct FieldSizes {
    std::size_t own_cells = 0, cells = 0, faces = 0, bfaces = 0, nodes = 0;
};

namespace detail {

[[noreturn]] inline void field_error(std::string_view name, const char* reason,
                                    std::source_location where = std::source_location::current()) noexcept
{
    char message[512]{};
    const int length = static_cast<int>(name.size() < 256 ? name.size() : 256);
    std::snprintf(message, sizeof(message), "Fields: %s ('%.*s')", reason, length, name.data());
    krona::mpi::fatal(message, where);
}

inline void require_kokkos() noexcept
{
    mpi::require(Kokkos::is_initialized() && !Kokkos::is_finalized(),
            "Fields require active Kokkos");
}

// Check before constructing an execution-space instance, including default arguments.
template<class Exec>
[[nodiscard]] Exec default_execution_space()
{
    require_kokkos();
    return Exec{};
}

} // namespace detail

/// @brief Host-only immutable metadata; virtual dispatch never enters kernels.
class AbstractField {
public:
    AbstractField(std::string name, FieldLocation location) : name_(std::move(name)), location_(location) {}
    virtual ~AbstractField() = default;
    [[nodiscard]] std::string_view name() const noexcept { return name_; }
    [[nodiscard]] FieldLocation location() const noexcept { return location_; }
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual int component_count() const noexcept = 0;
    [[nodiscard]] virtual std::size_t byte_size() const noexcept = 0;
private:
    std::string name_;
    FieldLocation location_;
};

/// @brief Owns one field allocation in Exec::memory_space; no permanent host copy.
/// Finish all users before destruction. Allocation leaves values uninitialized.
template<class T, int Components = 1, class Exec = execution::ExecutionSpace, class Layout = Kokkos::LayoutLeft>
class FieldStorage final : public AbstractField {
    static_assert(Components > 0 && !std::is_const_v<T>);
    static_assert(std::is_trivially_copyable_v<T>, "Field values must be trivially copyable");
    static_assert(std::is_same_v<Layout, Kokkos::LayoutLeft> ||
                  std::is_same_v<Layout, Kokkos::LayoutRight>,
                  "Owned fields support LayoutLeft and LayoutRight");
public:
    using memory_space = typename Exec::memory_space;
    using buffer_type = detail::FieldArray<T, Components, Kokkos::Device<Exec, memory_space>, Layout, true>;
    using view_type = FieldView<T, Components, memory_space, Layout>;
    using const_view_type = FieldView<const T, Components, memory_space, Layout>;

    FieldStorage(std::string name, std::size_t count, FieldLocation location, const Exec& exec)
        : AbstractField(std::move(name), location)
    {
        detail::require_kokkos();
        constexpr auto bytes_per_entity = sizeof(T) * static_cast<std::size_t>(Components);
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
    /// @brief Allocated value bytes, including layout padding; excludes metadata.
    [[nodiscard]] std::size_t byte_size() const noexcept override { return buffer_.span() * sizeof(T); }

    
    [[nodiscard]] view_type view() & noexcept { return view_type{typename view_type::view_type{buffer_}}; }
    [[nodiscard]] view_type view() && = delete;

    [[nodiscard]] const_view_type view() const & noexcept { return const_view_type{typename const_view_type::view_type{buffer_}}; }
    [[nodiscard]] const_view_type view() const && = delete;

    /// @brief Enqueues a fill on exec; caller synchronizes before external access.
    void fill(const Exec& exec, const T& value)
    { Kokkos::deep_copy(exec, buffer_, value); }

    /// @brief Allocates an independent, uninitialized host buffer with matching layout.
    [[nodiscard]] auto create_host_buffer() const
    { return Kokkos::create_mirror(Kokkos::WithoutInitializing, buffer_); }
private:
    buffer_type buffer_;
};

} // namespace krona::fields
