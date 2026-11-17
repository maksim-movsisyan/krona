#pragma once

#include <krona/linalg/vector.hpp>
#include <krona/fields/transfer.hpp>

namespace krona::linalg {

/// @brief Packs the owned prefix of each input field into the vector in cell-major order.
/// @tparam Scalar Precision of the destination vector.
/// @tparam Exec Kokkos execution space used for the copy.
/// @tparam Input A readable field view or a fields::pack(...) descriptor.
/// @details For a pack with three components, owned cell i starts at vector position 3*i.
/// Commit the vector before native use or releasing the source fields.
template<class Scalar, class Exec, class Input>
void copy_from_fields_async(Vector<Scalar, Exec>& vector, Input input, const Exec& exec = {}) {
    const auto values = vector.overwrite(exec);
    fields::copy_to_dofs_async(exec, input, Kokkos::subview(values, Kokkos::ALL(), 0));
}

/// @brief Copies owned vector values into each destination field's owned prefix.
/// @tparam Scalar Precision of the source vector.
/// @tparam Exec Kokkos execution space used for the copy.
/// @tparam Output A writable field view or a fields::pack(...) descriptor.
/// @details Destination regions must be separate from each other and the vector storage.
/// Commit the vector before using the output on another execution instance or releasing field buffers.
template<class Scalar, class Exec, class Output>
void copy_to_fields_async(const Vector<Scalar, Exec>& vector, Output output, const Exec& exec = {}) {
    const auto values = vector.read(exec);
    fields::copy_from_dofs_async(exec, Kokkos::subview(values, Kokkos::ALL(), 0), output);
}

/// @brief Convenience form: copy_from_fields_async(vector, exec, rho, momentum, energy).
template<class Scalar, class Exec, class First, class... Rest>
void copy_from_fields_async(Vector<Scalar, Exec>& vector, const Exec& exec, First first, Rest... rest)
{ copy_from_fields_async(vector, fields::pack(first, rest...), exec); }

template<class Scalar, class Exec, class First, class... Rest>
void copy_to_fields_async(const Vector<Scalar, Exec>& vector, const Exec& exec, First first, Rest... rest)
{ copy_to_fields_async(vector, fields::pack(first, rest...), exec); }

/// @brief Blocking counterpart: submits the transfer and commits its vector batch.
template<class Scalar, class Exec, class Input>
void copy_from_fields(Vector<Scalar, Exec>& vector, Input input, const Exec& exec = {})
{ copy_from_fields_async(vector, input, exec); vector.commit(); }

template<class Scalar, class Exec, class Output>
void copy_to_fields(const Vector<Scalar, Exec>& vector, Output output, const Exec& exec = {})
{ copy_to_fields_async(vector, output, exec); vector.commit(); }

template<class Scalar, class Exec, class First, class... Rest>
void copy_from_fields(Vector<Scalar, Exec>& vector, const Exec& exec, First first, Rest... rest)
{ copy_from_fields_async(vector, exec, first, rest...); vector.commit(); }

template<class Scalar, class Exec, class First, class... Rest>
void copy_to_fields(const Vector<Scalar, Exec>& vector, const Exec& exec, First first, Rest... rest)
{ copy_to_fields_async(vector, exec, first, rest...); vector.commit(); }

/// @brief Single-field forms use the same generic transfer and DOF order.
template<class Scalar, class Exec, class Field>
void copy_from_field_async(Vector<Scalar, Exec>& vector, Field field, const Exec& exec = {})
{ copy_from_fields_async(vector, field, exec); }

template<class Scalar, class Exec, class Field>
void copy_to_field_async(const Vector<Scalar, Exec>& vector, Field field, const Exec& exec = {})
{ copy_to_fields_async(vector, field, exec); }

template<class Scalar, class Exec, class Field>
void copy_from_field(Vector<Scalar, Exec>& vector, Field field, const Exec& exec = {})
{ copy_from_fields(vector, field, exec); }

template<class Scalar, class Exec, class Field>
void copy_to_field(const Vector<Scalar, Exec>& vector, Field field, const Exec& exec = {})
{ copy_to_fields(vector, field, exec); }

} // namespace krona::linalg
