#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_BlockCrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Vector.hpp>
#include <Teuchos_DefaultMpiComm.hpp>

#include <limits>
#include <cstddef>
#include <string_view>
#include <type_traits>

#ifndef HAVE_TPETRA_INST_INT_LONG_LONG
#error "KRONA linalg requires Tpetra ETI for int/long long ordinals"
#endif
#if defined(KRONA_EXECUTION_SPACE_SERIAL) && !defined(HAVE_TPETRA_INST_SERIAL)
#error "Tpetra Serial ETI is missing"
#elif defined(KRONA_EXECUTION_SPACE_OPENMP) && !defined(HAVE_TPETRA_INST_OPENMP)
#error "Tpetra OpenMP ETI is missing"
#elif defined(KRONA_EXECUTION_SPACE_CUDA) && !defined(HAVE_TPETRA_INST_CUDA)
#error "Tpetra CUDA ETI is missing"
#elif defined(KRONA_EXECUTION_SPACE_HIP) && !defined(HAVE_TPETRA_INST_HIP)
#error "Tpetra HIP ETI is missing"
#endif

namespace krona::linalg { 

using LocalOrdinal = int;
using GlobalOrdinal = long long;

namespace detail {
template<class> inline constexpr bool unsupported_node = false;
template<class Exec> struct NativeNode {
    static_assert(unsupported_node<Exec>, "Enable Tpetra ETI for the requested execution space");
};
#ifdef HAVE_TPETRA_INST_SERIAL
template<> struct NativeNode<Kokkos::Serial> { using type = Tpetra::KokkosCompat::KokkosSerialWrapperNode; };
#endif
#ifdef HAVE_TPETRA_INST_OPENMP
template<> struct NativeNode<Kokkos::OpenMP> { using type = Tpetra::KokkosCompat::KokkosOpenMPWrapperNode; };
#endif
#ifdef HAVE_TPETRA_INST_CUDA
template<> struct NativeNode<Kokkos::Cuda> { using type = Tpetra::KokkosCompat::KokkosCudaWrapperNode; };
#endif
#ifdef HAVE_TPETRA_INST_HIP
template<> struct NativeNode<Kokkos::HIP> { using type = Tpetra::KokkosCompat::KokkosHIPWrapperNode; };
#endif
} // namespace detail

/// @brief Scalar-independent distribution and sparsity types.
template<class Exec = execution::ExecutionSpace>
struct IndexTypes {
    using execution_space = Exec;
    using local_ordinal_type = LocalOrdinal;
    using global_ordinal_type = GlobalOrdinal;
    // Named nodes match Trilinos' ETI, including optional CUDA/HIP shared allocations.
    using node_type = typename detail::NativeNode<Exec>::type;
    using memory_space = typename node_type::memory_space;
    using map_type = Tpetra::Map<LocalOrdinal, GlobalOrdinal, node_type>;
    using graph_type = Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, node_type>;
};

/// @brief Exact Trilinos types; mesh ordinal widths do not change this ABI.
template<class Scalar = SolverReal, class Exec = execution::ExecutionSpace>
struct Types : IndexTypes<Exec> {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>);
#ifndef HAVE_TPETRA_INST_FLOAT
    static_assert(!std::is_same_v<Scalar, float>, "Tpetra float ETI is missing");
#endif
#ifndef HAVE_TPETRA_INST_DOUBLE
    static_assert(!std::is_same_v<Scalar, double>, "Tpetra double ETI is missing");
#endif
    using scalar_type = Scalar;
    using node_type = typename IndexTypes<Exec>::node_type;
    using matrix_type = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, node_type>;
    using block_matrix_type = Tpetra::BlockCrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, node_type>;
    using vector_type = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, node_type>;
    using multivector_type = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, node_type>;
    using operator_type = Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, node_type>;
};

namespace detail {

inline LocalOrdinal local_count(std::size_t value)
{
    mpi::require(value <= static_cast<std::size_t>(std::numeric_limits<LocalOrdinal>::max()),
                 "Linalg: local count exceeds Tpetra int ordinals");
    return static_cast<LocalOrdinal>(value);
}

} // namespace detail
} // namespace krona::linalg