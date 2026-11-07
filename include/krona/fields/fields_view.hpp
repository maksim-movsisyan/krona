#pragma once

#include <krona/execution/types.hpp>

#include <cstddef>
#include <type_traits>

namespace krona::fields {

namespace detail {

// Scalars use a rank-one View; multicomponent fields have a static second extent.
template<class T, int Components, class Space, class Layout, bool Owning>
using FieldArray = Kokkos::View<std::conditional_t<Components == 1, T*, T*[Components]>, Layout, Space, Kokkos::MemoryTraits<Owning ? 0 : Kokkos::Unmanaged>>;

} // namespace detail

/// @brief Non-owning, typed field descriptor passed by value to kernels.
/// Constness of T controls write access; constness of the descriptor does not.
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
    KOKKOS_INLINE_FUNCTION explicit FieldView(view_type data) : data_(data) {}

    /// @brief Allows mutable-to-const conversion without copying values.
    template<class U> requires (std::is_const_v<T> && !std::is_const_v<U> && std::is_same_v<std::remove_const_t<T>, U>)
    KOKKOS_INLINE_FUNCTION FieldView(const FieldView<U, Components, Space, Layout>& other) : data_(other.kokkos_view()) {}

    KOKKOS_INLINE_FUNCTION std::size_t size() const { return data_.extent(0); }
    KOKKOS_INLINE_FUNCTION bool empty() const { return size() == 0; }

    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t entity, int component) const
    {
        KOKKOS_ASSERT(component >= 0 && component < Components);
        if constexpr (Components == 1) return data_(entity);
        else return data_(entity, component);
    }

    /// @brief Scalar accessor
    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t entity) const requires (Components == 1)
    { return data_(entity); }
    KOKKOS_INLINE_FUNCTION T& operator[](std::size_t entity) const requires (Components == 1)
    { return data_(entity); }

    /// @brief 3D vector components accessor
    KOKKOS_INLINE_FUNCTION T& x(std::size_t entity) const requires (Components == 3)
    { return (*this)(entity, 0); }
    KOKKOS_INLINE_FUNCTION T& y(std::size_t entity) const requires (Components == 3)
    { return (*this)(entity, 1); }
    KOKKOS_INLINE_FUNCTION T& z(std::size_t entity) const requires (Components == 3)
    { return (*this)(entity, 2); }

    /// @brief Full 3x3 tensor, ordered xx,xy,xz,yx,yy,yz,zx,zy,zz.
    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t entity, int row, int column) const requires (Components == 9)
    {
        KOKKOS_ASSERT(row >= 0 && row < 3 && column >= 0 && column < 3);
        return (*this)(entity, row * 3 + column);
    }

    /// @brief Returns a component View; its stride depends on the field layout.
    KOKKOS_INLINE_FUNCTION auto component(int index) const
    {
        KOKKOS_ASSERT(index >= 0 && index < Components);
        if constexpr (Components == 1) return data_;
        else return Kokkos::subview(data_, Kokkos::ALL(), index);
    }

    KOKKOS_INLINE_FUNCTION auto as_const() const
    {
        using Result = FieldView<const std::remove_const_t<T>, Components, Space, Layout>;
        return Result{typename Result::view_type{data_}};
    }

    /// @brief Exposes the unmanaged Kokkos descriptor for copies and interoperability.
    KOKKOS_INLINE_FUNCTION view_type kokkos_view() const { return data_; }

    /// @brief Returns the base pointer; memory space, strides and lifetime still apply.
    KOKKOS_INLINE_FUNCTION T* data() const { return data_.data(); }

private:
    view_type data_;
};

template<class T, class Space = execution::MemorySpace, class Layout = Kokkos::LayoutLeft>
using ScalarView = FieldView<T, 1, Space, Layout>;
template<class T, class Space = execution::MemorySpace, class Layout = Kokkos::LayoutLeft>
using VectorView = FieldView<T, 3, Space, Layout>;
template<class T, class Space = execution::MemorySpace, class Layout = Kokkos::LayoutLeft>
using TensorView = FieldView<T, 9, Space, Layout>;

} // namespace krona::fields
