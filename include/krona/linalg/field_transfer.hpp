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
        const auto cell = dof/static_cast<std::size_t>(Field::components);
        const int component = static_cast<int>(dof%static_cast<std::size_t>(Field::components));
        if constexpr (ToVector) values(dof, 0) = static_cast<typename Values::non_const_value_type>(field(cell, component));
        else field(cell, component) = static_cast<typename Field::value_type>(values(dof, 0));
    }
};
} // namespace detail

/// @brief Packs owned cells into cell-major DOFs, independently of field layout.
/// The vector's local row order must be cell0/comp0,...,cell0/compN-1,cell1/comp0,... .
template<class Scalar, class Exec, class T, int N, class Space, class Layout>
void copy_from_field(Vector<Scalar, Exec>& vector, fields::FieldView<T, N, Space, Layout> field, const Exec& exec = {})
{
    static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible);
    mpi::require(vector.local_size()%static_cast<std::size_t>(N) == 0 &&
                    field.size() >= vector.local_size()/static_cast<std::size_t>(N), "Linalg: field/DOF size mismatch");
    auto values = vector.edit(exec);
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
    auto values = vector.read();
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("linalg/vector_to_field", Policy(exec, 0, vector.local_size()),
        detail::FieldCopy<decltype(field), decltype(values), false>{field, values});
}

} // namespace krona::linalg
