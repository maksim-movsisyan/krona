#pragma once

#include <krona/execution/types.hpp>

#include <cstddef>
#include <type_traits>

namespace krona::fields {

namespace detail {

/// @brief Kokkos array indexed by entity, and by component for a multicomponent field.
/// @tparam T Type of one value, including const for read-only access.
/// @tparam Components Number of values per entity. A scalar uses a rank-one array.
/// @tparam Space Kokkos memory space, or a Kokkos device pairing it with an execution space.
/// @tparam Layout How entities and components are arranged in memory.
/// @tparam Owning True for managed storage, false for a view that borrows it.
template<class T, int Components, class Space, class Layout, bool Owning>
using FieldArray = Kokkos::View<std::conditional_t<Components == 1, T*, T*[Components]>,
                               Layout, Space, Kokkos::MemoryTraits<Owning ? 0 : Kokkos::Unmanaged>>;

} // namespace detail

/// @brief Borrows field values so a kernel can index them by entity and component.
/// @tparam T Type of one value, for example StorageReal for a writable field
/// or const StorageReal for a read-only field.
/// @tparam Components Number of values per entity, for example 1 for a scalar or 3 for velocity.
/// @tparam Space Kokkos memory space where the values live.
/// @tparam Layout Memory layout. LayoutLeft groups values by component (SoA). LayoutRight groups them by entity (AoS).
/// A const FieldView with mutable T can still write values. Keep the allocation alive
/// until all kernels and copies using the view have finished.
template<class T, int Components = 1, class Space = execution::MemorySpace, class Layout = Kokkos::LayoutLeft>
class FieldView {
    static_assert(Components > 0);
public:
    using value_type = T;
    using memory_space = Space;
    using array_layout = Layout;
    using view_type = detail::FieldArray<T, Components, Space, Layout, false>;
    static constexpr int components = Components;

    KOKKOS_INLINE_FUNCTION FieldView() = default;
    /// @brief Borrows an existing Kokkos array with the matching type, shape and layout.
    KOKKOS_INLINE_FUNCTION explicit FieldView(view_type data) : data_(data) {}

    /// @brief Creates a read-only view of the same values.
    /// @tparam U Mutable scalar type corresponding to const T.
    template<class U> requires (std::is_const_v<T> && !std::is_const_v<U> && std::is_same_v<std::remove_const_t<T>, U>)
    KOKKOS_INLINE_FUNCTION FieldView(const FieldView<U, Components, Space, Layout>& other) : data_(other.kokkos_view()) {}

    /// @brief Returns the number of entities, even if each entity has several components.
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return data_.extent(0); }
    KOKKOS_INLINE_FUNCTION bool empty() const { return size() == 0; }

    /// @brief Accesses one value by zero-based entity and component indices.
    /// @param entity Local index in [0, size()).
    /// @param component Index in [0, Components). For a scalar field, pass zero.
    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t entity, int component) const {
        KOKKOS_ASSERT(component >= 0 && component < Components);
        if constexpr (Components == 1) return data_(entity);
        else return data_(entity, component);
    }

    /// @brief Accesses the value at this entity in a scalar field.
    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t entity) const requires (Components == 1)
    { return data_(entity); }
    KOKKOS_INLINE_FUNCTION T& operator[](std::size_t entity) const requires (Components == 1)
    { return data_(entity); }

    /// @brief Accesses the x component of a three-component field.
    KOKKOS_INLINE_FUNCTION T& x(std::size_t entity) const requires (Components == 3)
    { return (*this)(entity, 0); }
    /// @brief Accesses the y component of a three-component field.
    KOKKOS_INLINE_FUNCTION T& y(std::size_t entity) const requires (Components == 3)
    { return (*this)(entity, 1); }
    /// @brief Accesses the z component of a three-component field.
    KOKKOS_INLINE_FUNCTION T& z(std::size_t entity) const requires (Components == 3)
    { return (*this)(entity, 2); }

    /// @brief Accesses a full 3x3 tensor stored in row order (xx, xy, xz, yx, yy, yz, zx, zy, zz).
    /// @param entity Local index in [0, size()).
    /// @param row Zero-based tensor row in [0, 3).
    /// @param column Zero-based tensor column in [0, 3).
    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t entity, int row, int column) const requires (Components == 9) {
        KOKKOS_ASSERT(row >= 0 && row < 3 && column >= 0 && column < 3);
        return (*this)(entity, row * 3 + column);
    }

    /// @brief Borrows one component across all entities as a rank-one Kokkos view.
    /// @param index Component in [0, Components). Use 0 for a scalar or 2 for the z component of a vector.
    /// The distance between consecutive values follows the original layout.
    KOKKOS_INLINE_FUNCTION auto component(int index) const {
        KOKKOS_ASSERT(index >= 0 && index < Components);
        if constexpr (Components == 1) return data_;
        else return Kokkos::subview(data_, Kokkos::ALL(), index);
    }

    /// @brief Returns a read-only view sharing the same allocation.
    KOKKOS_INLINE_FUNCTION auto as_const() const {
        using Result = FieldView<const std::remove_const_t<T>, Components, Space, Layout>;
        return Result{typename Result::view_type{data_}};
    }

    /// @brief Exposes the unmanaged Kokkos descriptor for copies and interoperability.
    KOKKOS_INLINE_FUNCTION view_type kokkos_view() const { return data_; }

    /// @brief Returns the address of the first value. Use the view's strides for indexing.
    KOKKOS_INLINE_FUNCTION T* data() const { return data_.data(); }

private:
    view_type data_;
};

/// @brief FieldView for one scalar per entity.
template<class T, class Space = execution::MemorySpace, class Layout = Kokkos::LayoutLeft>
using ScalarView = FieldView<T, 1, Space, Layout>;
/// @brief FieldView for an x, y, z vector per entity.
template<class T, class Space = execution::MemorySpace, class Layout = Kokkos::LayoutLeft>
using VectorView = FieldView<T, 3, Space, Layout>;
/// @brief FieldView for a full 3x3 tensor per entity.
template<class T, class Space = execution::MemorySpace, class Layout = Kokkos::LayoutLeft>
using TensorView = FieldView<T, 9, Space, Layout>;

} // namespace krona::fields
