#pragma once

#include <krona/fields/component.hpp>

#include <cstddef>
#include <type_traits>

namespace krona::numerics::gradient {

/// @brief Reads the three Cartesian components of a cell gradient as {gx, gy, gz}.
/// @tparam X Descriptor for the x derivative across cells.
/// @tparam Y Descriptor for the y derivative across cells.
/// @tparam Z Descriptor for the z derivative across cells.
/// Build with cell_gradients() and keep the underlying arrays alive as long as the reader is used.
template <class X, class Y, class Z> 
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

    X x;
    Y y;
    Z z;
    std::size_t extent = 0;

    KOKKOS_INLINE_FUNCTION std::size_t size() const { return extent; }

    KOKKOS_INLINE_FUNCTION Kokkos::Array<value_type, 3> operator()(std::size_t cell) const {
        KOKKOS_ASSERT(cell < extent);
        return {static_cast<value_type>(x(cell)), static_cast<value_type>(y(cell)),
                static_cast<value_type>(z(cell))};
    }

    /// @brief Checks whether the borrowed descriptors can cover the requested cells.
    /// @tparam Exec Execution space that will read the gradients.
    /// @param count Minimum number of cells needed by the caller, including ghosts if it will read them.
    template <class Exec> void validate(std::size_t count) const {
        static_assert(Kokkos::SpaceAccessibility<Exec, memory_space>::accessible);
        mpi::require(extent >= count, "Gradient correction: gradient extent is too short");
        mpi::require(extent == 0 ||
                         (x.data && y.data && z.data && x.stride > 0 && y.stride > 0 && z.stride > 0),
                          "Gradient correction: invalid component storage");
    }
};

/// @brief Borrows gradient components from three rank-one views in the same memory space.
/// @tparam X Kokkos view type holding the x derivative.
/// @tparam Y Kokkos view type holding the y derivative.
/// @tparam Z Kokkos view type holding the z derivative.
/// @param x,y,z Views indexed by local cell ID. Strided component views are supported.
/// @return A read-only reader whose size is the shortest input view.
template <class X, class Y, class Z> 
[[nodiscard]] auto cell_gradients(X x, Y y, Z z) {
    const auto dx = fields::descriptor_from_view(x);
    const auto dy = fields::descriptor_from_view(y);
    const auto dz = fields::descriptor_from_view(z);
    using RX = fields::ComponentDescriptor<const std::remove_const_t<typename X::value_type>,
                                           typename X::memory_space>;
    using RY = fields::ComponentDescriptor<const std::remove_const_t<typename Y::value_type>,
                                           typename Y::memory_space>;
    using RZ = fields::ComponentDescriptor<const std::remove_const_t<typename Z::value_type>,
                                           typename Z::memory_space>;
    const auto xy = x.extent(0) < y.extent(0) ? x.extent(0) : y.extent(0);
    const auto count = xy < z.extent(0) ? xy : z.extent(0);
    return CellGradients<RX, RY, RZ>{{dx.data, dx.stride}, {dy.data, dy.stride}, {dz.data, dz.stride}, count};
}

/// @brief Selects one input variable's three derivatives from a gradient field.
/// @tparam Field Field view type with three consecutive gradient components per input variable.
/// @param field Gradient values laid out by input variable, then by axis.
/// @param component Input variable to select. For grad(u, v, w), selecting 1 reads dv/dx, dv/dy, dv/dz.
template <class Field> 
[[nodiscard]] auto cell_gradients(Field field, int component = 0) {
    static_assert(Field::components > 0 && Field::components % 3 == 0,
                  "Gradient correction: field must have 3*N components");
    mpi::require(component >= 0 && component < Field::components / 3,
                 "Gradient correction: component index is out of range");
    return cell_gradients(field.component(3 * component), field.component(3 * component + 1),
                          field.component(3 * component + 2));
}

/// @brief Supplies a zero correction using the same calls as GradientCorrection.
struct NoCorrection {
    template <class Real> KOKKOS_INLINE_FUNCTION Real increment(std::size_t, Real, Real, Real) const {
        return Real{0};
    }

    template <class Real>
    KOKKOS_INLINE_FUNCTION Real interior(std::size_t, std::size_t, Real, Real, Real, Real) const {
        return Real{0};
    }

    template <class Context> KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const {
        return std::remove_cvref_t<decltype(c.tx)>{0};
    }

    template <class Context, class Real>
    KOKKOS_INLINE_FUNCTION Real interior(const Context&, Real, Real, Real) const {
        return Real{0};
    }
};

/// @brief Computes the dot product of a gradient with the geometric vector supplied by the caller.
/// @tparam Gradients Device-callable reader whose gradients(cell)[axis] returns a derivative.
/// The caller chooses the vector, applies any flux scaling and supplies current gradient values.
template <class Gradients> 
struct GradientCorrection {
    Gradients gradients;

    /// @brief Projects one cell's gradient onto t, for example a tangential boundary displacement.
    /// @tparam Real Floating-point type used for the dot product and its result.
    /// @param cell Local cell ID to read.
    /// @param tx,ty,tz Cartesian components of t.
    template <class Real>
    KOKKOS_INLINE_FUNCTION Real increment(std::size_t cell, Real tx, Real ty, Real tz) const {
        static_assert(std::is_floating_point_v<Real>);
        const auto g = gradients(cell);
        return tx * static_cast<Real>(g[0]) + ty * static_cast<Real>(g[1]) + tz * static_cast<Real>(g[2]);
    }

    /// @brief Interpolates the two cell gradients, then projects the result onto t.
    /// @tparam Real Floating-point type used for interpolation and projection.
    /// @param owner,neighbour Local IDs of the two cells. The neighbour may be an MPI ghost.
    /// @param owner_weight Weight of the owner gradient. The neighbour uses 1 - owner_weight.
    /// @param tx,ty,tz Cartesian components of t.
    template <class Real>
    KOKKOS_INLINE_FUNCTION Real interior(std::size_t owner, std::size_t neighbour, Real owner_weight,
                                         Real tx, Real ty, Real tz) const {
        static_assert(std::is_floating_point_v<Real>);
        const auto p = gradients(owner), n = gradients(neighbour);
        return tx * (owner_weight * static_cast<Real>(p[0]) +
                     (Real{1} - owner_weight) * static_cast<Real>(n[0])) +
               ty * (owner_weight * static_cast<Real>(p[1]) +
                     (Real{1} - owner_weight) * static_cast<Real>(n[1])) +
               tz * (owner_weight * static_cast<Real>(p[2]) +
                     (Real{1} - owner_weight) * static_cast<Real>(n[2]));
    }

    /// @brief Projects the owner gradient using the vector stored in a face context.
    /// @tparam Context Type exposing owner and the vector components tx, ty and tz.
    template <class Context> KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const {
        using Real = std::remove_cvref_t<decltype(c.tx)>;
        return increment<Real>(static_cast<std::size_t>(c.owner), c.tx, c.ty, c.tz);
    }

    /// @brief Uses the adjacent cells and owner weight supplied by a face context.
    /// @tparam Context Type exposing owner, neighbour and owner_weight.
    /// @tparam Real Floating-point type used for the projection and its result.
    template <class Context, class Real>
    KOKKOS_INLINE_FUNCTION Real interior(const Context& c, Real tx, Real ty, Real tz) const {
        return interior(static_cast<std::size_t>(c.owner), static_cast<std::size_t>(c.neighbour),
                        static_cast<Real>(c.owner_weight), tx, ty, tz);
    }
};

template <class Gradients> GradientCorrection(Gradients) -> GradientCorrection<Gradients>;

} // namespace krona::numerics::gradient
