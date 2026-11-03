#pragma once

#include <krona/linalg/vector.hpp>
#include <krona/fields/transfer.hpp>

namespace krona::linalg {

/// @brief Packs fields::pack(...) into owned cell-major vector DOFs without intermediate storage.
/// Opens a vector access batch; commit before native use or releasing field buffers.
template<class Scalar, class Exec, class Input>
void copy_from_fields_async(Vector<Scalar, Exec>& vector, Input input, const Exec& exec = {})
{
    const auto values = vector.overwrite(exec);
    fields::copy_to_dofs_async(exec, input, Kokkos::subview(values, Kokkos::ALL(), 0));
}

/// @brief Unpacks owned DOFs; ghost/boundary tails are untouched. Commit the vector after use.
/// Destinations must be writable, mutually non-overlapping, and separate from native values.
template<class Scalar, class Exec, class Output>
void copy_to_fields_async(const Vector<Scalar, Exec>& vector, Output output, const Exec& exec = {})
{
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