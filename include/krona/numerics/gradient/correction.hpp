#pragma once

#include <krona/fields/component.hpp>

#include <cstddef>
#include <type_traits>

namespace krona::numerics::gradient {

/// @brief Borrowed Cartesian gradient components; operator()(cell) returns {gx,gy,gz} by value.
/// Build with cell_gradients(); no allocation, copies of field values, submission or MPI.
template<class X, class Y, class Z>
struct CellGradients {
    using value_type = std::common_type_t<std::remove_const_t<typename X::value_type>,
                                          std::remove_const_t<typename Y::value_type>,
                                          std::remove_const_t<typename Z::value_type>>;
    using memory_space = typename X::memory_space;
    static_assert(std::is_same_v<memory_space, typename Y::memory_space> &&
                  std::is_same_v<memory_space, typename Z::memory_space>);
    static_assert(std::is_floating_point_v<std::remove_const_t<typename X::value_type>> &&
                  std::is_floating_point_v<std::remove_const_t<typename Y::value_type>> &&
                  std::is_floating_point_v<std::remove_const_t<typename Z::value_type>>);

    X x; Y y; Z z;
    std::size_t extent = 0;

    KOKKOS_INLINE_FUNCTION std::size_t size() const { return extent; }

    KOKKOS_INLINE_FUNCTION Kokkos::Array<value_type, 3> operator()(std::size_t cell) const {
        KOKKOS_ASSERT(cell < extent);
        return {static_cast<value_type>(x(cell)),
                static_cast<value_type>(y(cell)),
                static_cast<value_type>(z(cell))};
    }

    /// @brief Optional local host check; no fence or data inspection is performed.
    template<class Exec>
    void validate(std::size_t count) const {
        static_assert(Kokkos::SpaceAccessibility<Exec, memory_space>::accessible);
        mpi::require(extent >= count, "Gradient correction: gradient extent is too short");
        mpi::require(extent == 0 || (x.data && y.data && z.data &&
                     x.stride > 0 && y.stride > 0 && z.stride > 0),
                     "Gradient correction: invalid component storage");
    }
};

/// @brief Adapts three rank-one views, including strided views, without retaining their ownership.
/// The exposed extent is the minimum input length; all inputs share one memory space.
template<class X, class Y, class Z>
[[nodiscard]] auto cell_gradients(X x, Y y, Z z) {
    const auto dx = fields::descriptor_from_view(x);
    const auto dy = fields::descriptor_from_view(y);
    const auto dz = fields::descriptor_from_view(z);
    using RX = fields::ComponentDescriptor<const std::remove_const_t<typename X::value_type>, typename X::memory_space>;
    using RY = fields::ComponentDescriptor<const std::remove_const_t<typename Y::value_type>, typename Y::memory_space>;
    using RZ = fields::ComponentDescriptor<const std::remove_const_t<typename Z::value_type>, typename Z::memory_space>;
    const auto xy = x.extent(0) < y.extent(0) ? x.extent(0) : y.extent(0);
    const auto count = xy < z.extent(0) ? xy : z.extent(0);
    return CellGradients<RX, RY, RZ>{{dx.data, dx.stride}, {dy.data, dy.stride}, {dz.data, dz.stride}, count};
}

/// @brief Selects gradient components 3*component+axis from an existing gradient FieldView.
template<class Field>
[[nodiscard]] auto cell_gradients(Field field, int component = 0) {
    static_assert(Field::components > 0 && Field::components % 3 == 0,
                  "Gradient correction: field must have 3*N components");
    mpi::require(component >= 0 && component < Field::components/3,
                 "Gradient correction: component index is out of range");
    return cell_gradients(field.component(3*component), field.component(3*component+1),
                          field.component(3*component+2));
}

/// @brief Zero projection; reads no gradients. Context adapters use structural members only.
struct NoCorrection {
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real increment(std::size_t, Real, Real, Real) const { return Real{0}; }

    template<class Real>
    KOKKOS_INLINE_FUNCTION Real interior(std::size_t, std::size_t, Real, Real, Real, Real) const
    { return Real{0}; }

    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const
    { return std::remove_cvref_t<decltype(c.tx)>{0}; }

    template<class Context, class Real>
    KOKKOS_INLINE_FUNCTION Real interior(const Context&, Real, Real, Real) const { return Real{0}; }
};

/// @brief Projects cell gradients onto a caller-supplied geometric vector.
/// No area, transport coefficient, flux sign or physical boundary condition is applied here.
/// Gradients is callable as gradients(cell)[axis]; keep all backing storage alive and current.
template<class Gradients>
struct GradientCorrection {
    Gradients gradients;

    /// @brief dot(t, grad(cell)); t may be a boundary tangential displacement.
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real increment(std::size_t cell, Real tx, Real ty, Real tz) const {
        static_assert(std::is_floating_point_v<Real>);
        const auto g = gradients(cell);
        return tx*static_cast<Real>(g[0])+ty*static_cast<Real>(g[1])+tz*static_cast<Real>(g[2]);
    }

    /// @brief dot(t, w*grad(owner)+(1-w)*grad(neighbour)); w multiplies the OWNER gradient.
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real interior(std::size_t owner, std::size_t neighbour,
        Real owner_weight, Real tx, Real ty, Real tz) const {
        static_assert(std::is_floating_point_v<Real>);
        const auto p = gradients(owner), n = gradients(neighbour);
        return tx*(owner_weight*static_cast<Real>(p[0])+(Real{1}-owner_weight)*static_cast<Real>(n[0])) +
               ty*(owner_weight*static_cast<Real>(p[1])+(Real{1}-owner_weight)*static_cast<Real>(n[1])) +
               tz*(owner_weight*static_cast<Real>(p[2])+(Real{1}-owner_weight)*static_cast<Real>(n[2]));
    }

    /// @brief Adapter for any context with owner,tx,ty,tz; no boundary-module dependency.
    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const {
        using Real = std::remove_cvref_t<decltype(c.tx)>;
        return increment<Real>(static_cast<std::size_t>(c.owner), c.tx, c.ty, c.tz);
    }

    /// @brief Adapter for any context with owner,neighbour,owner_weight.
    template<class Context, class Real>
    KOKKOS_INLINE_FUNCTION Real interior(const Context& c, Real tx, Real ty, Real tz) const {
        return interior(static_cast<std::size_t>(c.owner), static_cast<std::size_t>(c.neighbour),
                        static_cast<Real>(c.owner_weight), tx, ty, tz);
    }
};

template<class Gradients>
GradientCorrection(Gradients) -> GradientCorrection<Gradients>;

} // namespace krona::numerics::gradient
