#pragma once

#include <krona/linalg/vector.hpp>
#include <krona/fields/fields_view.hpp>

namespace krona::linalg {
namespace detail {
template<class Field, class Values, bool ToVector>
struct FieldCopy {
    Field field;
    Values values;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t dof) const
    {
        const std::size_t cell = dof/static_cast<std::size_t>(Field::components);
        const int component = static_cast<int>(dof%static_cast<std::size_t>(Field::components));
        if constexpr (ToVector) values(dof, 0) = static_cast<typename Values::non_const_value_type>(field(cell, component));
        else field(cell, component) = static_cast<typename Field::value_type>(values(dof, 0));
    }
};

template<class View>
struct TransferComponent {
    using value_type = typename View::value_type;
    using memory_space = typename View::memory_space;
    static constexpr int components = 1;
    View view;
    [[nodiscard]] std::size_t size() const { return view.extent(0); }
    KOKKOS_INLINE_FUNCTION decltype(auto) operator()(std::size_t cell, int) const { return view(cell); }
};

template<class Field>
auto transfer_field(Field field)
{
    if constexpr (requires { Field::components; }) return field;
    else {
        static_assert(Kokkos::is_view<Field>::value && Field::rank == 1,
                      "Pass a FieldView or a rank-one Kokkos component View");
        return TransferComponent<Field>{field};
    }
}
} // namespace detail

/// @brief Device-copyable descriptors; fields are concatenated within each owned cell.
template<class... Fields> struct FieldPack;

template<>
struct FieldPack<> {
    static constexpr std::size_t components = 0;
    template<class Exec, bool ToVector> void check(std::size_t) const {}
    template<bool ToVector, class Values>
    KOKKOS_INLINE_FUNCTION void copy(std::size_t, Values, std::size_t) const {}
};

template<class First, class... Rest>
struct FieldPack<First, Rest...> {
    First first;
    FieldPack<Rest...> rest;
    static constexpr std::size_t components = static_cast<std::size_t>(First::components)+FieldPack<Rest...>::components;

    template<class Exec, bool ToVector>
    void check(std::size_t owned_cells) const
    {
        static_assert(Kokkos::SpaceAccessibility<Exec, typename First::memory_space>::accessible);
        static_assert(ToVector || !std::is_const_v<typename First::value_type>, "Cannot unpack into a const field");
        mpi::require(first.size() >= owned_cells, "Linalg: field is shorter than the owned cell prefix");
        rest.template check<Exec, ToVector>(owned_cells);
    }

    template<bool ToVector, class Values>
    KOKKOS_INLINE_FUNCTION void copy(std::size_t cell, Values values, std::size_t offset) const
    {
        for (int c = 0; c < First::components; ++c) {
            const std::size_t dof = offset+static_cast<std::size_t>(c);
            if constexpr (ToVector)
                values(dof, 0) = static_cast<typename Values::non_const_value_type>(first(cell, c));
            else first(cell, c) = static_cast<typename First::value_type>(values(dof, 0));
        }
        rest.template copy<ToVector>(cell, values, offset+static_cast<std::size_t>(First::components));
    }
};

inline FieldPack<> field_pack() { return {}; }

/// @brief Accepts FieldViews and individual rank-one component Views, including strided subviews.
template<class First, class... Rest>
auto field_pack(First first, Rest... rest)
{
    auto field = detail::transfer_field(first);
    auto tail = field_pack(rest...);
    return FieldPack<decltype(field), decltype(detail::transfer_field(rest))...>{field, tail};
}

namespace detail {
template<class Pack, class Values, bool ToVector>
struct FieldsCopy {
    Pack fields;
    Values values;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const
    { fields.template copy<ToVector>(cell, values, cell*Pack::components); }
};
} // namespace detail

/// @brief Packs all owned cells in one asynchronous kernel; no intermediate storage.
/// Vector DOFs must follow cell order, then the supplied field/component order.
/// Fields must outlive the submitted work and must not alias the destination vector.
template<class Scalar, class Exec, class... Fields>
void copy_from_fields(Vector<Scalar, Exec>& vector, FieldPack<Fields...> fields, const Exec& exec = {})
{
    constexpr auto width = FieldPack<Fields...>::components;
    static_assert(width > 0, "At least one field is required");
    mpi::require(vector.local_size()%width == 0, "Linalg: field pack/DOF size mismatch");
    const auto owned_cells = vector.local_size()/width;
    fields.template check<Exec, true>(owned_cells);
    auto values = vector.overwrite(exec);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("linalg/fields_to_vector", Policy(exec, 0, owned_cells),
        detail::FieldsCopy<decltype(fields), decltype(values), true>{fields, values});
}

/// @brief Unpacks in one asynchronous kernel; all ghost/boundary tails are untouched.
/// Destination fields/components must not overlap each other or the source vector.
template<class Scalar, class Exec, class... Fields>
void copy_to_fields(const Vector<Scalar, Exec>& vector, FieldPack<Fields...> fields, const Exec& exec = {})
{
    constexpr auto width = FieldPack<Fields...>::components;
    static_assert(width > 0, "At least one field is required");
    mpi::require(vector.local_size()%width == 0, "Linalg: field pack/DOF size mismatch");
    const auto owned_cells = vector.local_size()/width;
    fields.template check<Exec, false>(owned_cells);
    auto values = vector.read(exec);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("linalg/vector_to_fields", Policy(exec, 0, owned_cells),
        detail::FieldsCopy<decltype(fields), decltype(values), false>{fields, values});
}

/// @brief Convenience form: copy_from_fields(vector, exec, field1, field2, ...).
template<class Scalar, class Exec, class First, class... Rest>
void copy_from_fields(Vector<Scalar, Exec>& vector, const Exec& exec, First first, Rest... rest)
{ copy_from_fields(vector, field_pack(first, rest...), exec); }

template<class Scalar, class Exec, class First, class... Rest>
void copy_to_fields(const Vector<Scalar, Exec>& vector, const Exec& exec, First first, Rest... rest)
{ copy_to_fields(vector, field_pack(first, rest...), exec); }

/// @brief Packs owned cells into cell-major DOFs, independently of field layout.
/// The vector's local row order must be cell0/comp0,...,cell0/compN-1,cell1/comp0,... .
template<class Scalar, class Exec, class T, int N, class Space, class Layout>
void copy_from_field(Vector<Scalar, Exec>& vector, fields::FieldView<T, N, Space, Layout> field, const Exec& exec = {})
{
    static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible);
    mpi::require(vector.local_size()%static_cast<std::size_t>(N) == 0 &&
                 field.size() >= vector.local_size()/static_cast<std::size_t>(N), "Linalg: field/DOF size mismatch");
    auto values = vector.overwrite(exec);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("linalg/field_to_vector", Policy(exec, 0, vector.local_size()),
        detail::FieldCopy<decltype(field), decltype(values), true>{field, values});
}

/// @brief Unpacks owned DOFs asynchronously; the field's extended tail is unchanged.
template<class Scalar, class Exec, class T, int N, class Space, class Layout>
void copy_to_field(const Vector<Scalar, Exec>& vector, fields::FieldView<T, N, Space, Layout> field, const Exec& exec = {})
{
    static_assert(!std::is_const_v<T> && Kokkos::SpaceAccessibility<Exec, Space>::accessible);
    mpi::require(vector.local_size()%static_cast<std::size_t>(N) == 0 &&
                 field.size() >= vector.local_size()/static_cast<std::size_t>(N), "Linalg: field/DOF size mismatch");
    auto values = vector.read(exec);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("linalg/vector_to_field", Policy(exec, 0, vector.local_size()),
        detail::FieldCopy<decltype(field), decltype(values), false>{field, values});
}

} // namespace krona::linalg