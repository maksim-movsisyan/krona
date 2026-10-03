#pragma once

#include <mpi.h>

#include <climits>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace krona::mpi {

static_assert(CHAR_BIT == 8, "KRONA MPI requires 8-bit bytes");

namespace detail {

// Canonical fixed-width handles also cover their fundamental-type aliases.
template<class T>
[[nodiscard]] inline MPI_Datatype integer_datatype(MPI_Datatype native) noexcept {
#ifdef INT8_MAX
    if constexpr (std::same_as<T, std::int8_t>) {
        return MPI_INT8_T;
    }
#endif
#ifdef UINT8_MAX
    if constexpr (std::same_as<T, std::uint8_t>) {
        return MPI_UINT8_T;
    }
#endif
#ifdef INT16_MAX
    if constexpr (std::same_as<T, std::int16_t>) {
        return MPI_INT16_T;
    }
#endif
#ifdef UINT16_MAX
    if constexpr (std::same_as<T, std::uint16_t>) {
        return MPI_UINT16_T;
    }
#endif
#ifdef INT32_MAX
    if constexpr (std::same_as<T, std::int32_t>) {
        return MPI_INT32_T;
    }
#endif
#ifdef UINT32_MAX
    if constexpr (std::same_as<T, std::uint32_t>) {
        return MPI_UINT32_T;
    }
#endif
#ifdef INT64_MAX
    if constexpr (std::same_as<T, std::int64_t>) {
        return MPI_INT64_T;
    }
#endif
#ifdef UINT64_MAX
    if constexpr (std::same_as<T, std::uint64_t>) {
        return MPI_UINT64_T;
    }
#endif
    return native;
}

template<class T>
struct datatype_map {};

// Integer specializations use fundamental types, never their aliases.
#define KRONA_MPI_DETAIL_MAP(cpp_type, mpi_expression)              \
    template<> struct datatype_map<cpp_type> {                     \
        [[nodiscard]] static MPI_Datatype get() noexcept           \
        {                                                         \
            return mpi_expression;                                \
        }                                                         \
    };

KRONA_MPI_DETAIL_MAP(char, MPI_CHAR)
KRONA_MPI_DETAIL_MAP(wchar_t, MPI_WCHAR)
KRONA_MPI_DETAIL_MAP(std::byte, MPI_BYTE)

KRONA_MPI_DETAIL_MAP(signed char, integer_datatype<signed char>(MPI_SIGNED_CHAR))
KRONA_MPI_DETAIL_MAP(unsigned char, integer_datatype<unsigned char>(MPI_UNSIGNED_CHAR))
KRONA_MPI_DETAIL_MAP(short, integer_datatype<short>(MPI_SHORT))
KRONA_MPI_DETAIL_MAP(unsigned short, integer_datatype<unsigned short>(MPI_UNSIGNED_SHORT))
KRONA_MPI_DETAIL_MAP(int, integer_datatype<int>(MPI_INT))
KRONA_MPI_DETAIL_MAP(unsigned int, integer_datatype<unsigned int>(MPI_UNSIGNED))
KRONA_MPI_DETAIL_MAP(long, integer_datatype<long>(MPI_LONG))
KRONA_MPI_DETAIL_MAP(unsigned long, integer_datatype<unsigned long>(MPI_UNSIGNED_LONG))
KRONA_MPI_DETAIL_MAP(long long, integer_datatype<long long>(MPI_LONG_LONG_INT))
KRONA_MPI_DETAIL_MAP(unsigned long long,
                     integer_datatype<unsigned long long>(MPI_UNSIGNED_LONG_LONG))

KRONA_MPI_DETAIL_MAP(float, MPI_FLOAT)
KRONA_MPI_DETAIL_MAP(double, MPI_DOUBLE)
KRONA_MPI_DETAIL_MAP(long double, MPI_LONG_DOUBLE)

#ifdef MPI_CXX_BOOL
KRONA_MPI_DETAIL_MAP(bool, MPI_CXX_BOOL)
#endif
#ifdef MPI_CXX_FLOAT_COMPLEX
KRONA_MPI_DETAIL_MAP(std::complex<float>, MPI_CXX_FLOAT_COMPLEX)
#endif
#ifdef MPI_CXX_DOUBLE_COMPLEX
KRONA_MPI_DETAIL_MAP(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX)
#endif
#ifdef MPI_CXX_LONG_DOUBLE_COMPLEX
KRONA_MPI_DETAIL_MAP(std::complex<long double>, MPI_CXX_LONG_DOUBLE_COMPLEX)
#endif

#undef KRONA_MPI_DETAIL_MAP

} // namespace detail

// Const and references are normalized; volatile and unmapped types are rejected.
template<class T>
concept MpiDatatype =
    !std::is_volatile_v<std::remove_reference_t<T>> &&
    requires {
        { detail::datatype_map<std::remove_cvref_t<T>>::get() } noexcept -> std::same_as<MPI_Datatype>;
    };

// Return a borrowed predefined handle; never commit or free it.
// Make MPI calls with it only while MPI is active; valid MPI_Op is a separate rule.
template<MpiDatatype T>
[[nodiscard]] inline MPI_Datatype datatype() noexcept {
    return detail::datatype_map<std::remove_cvref_t<T>>::get();
}
 
} // namespace krona::mpi
