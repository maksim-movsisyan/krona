#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
#include <Teuchos_DefaultMpiComm.hpp>

#include <limits>
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

/// @brief Exact Trilinos types; mesh ordinal widths do not change this ABI.
template<class Scalar = SolverReal, class Exec = execution::ExecutionSpace>
struct Types {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>);
    using scalar_type = Scalar;
    using execution_space = Exec;
    using node_type = Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Exec>;
    using map_type = Tpetra::Map<LocalOrdinal, GlobalOrdinal, node_type>;
    using graph_type = Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, node_type>;
    using matrix_type = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, node_type>;
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
