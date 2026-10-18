#pragma once

#include <krona/linalg/types.hpp>

namespace krona::linalg::detail {

// Keep native storage differences outside solver/preconditioner algorithms.
template<class S, class L, class G, class N>
auto read_coefficients(const Tpetra::CrsMatrix<S, L, G, N>& matrix)
{ return matrix.getLocalValuesDevice(Tpetra::Access::ReadOnly); }

template<class S, class L, class G, class N>
auto overwrite_coefficients(Tpetra::CrsMatrix<S, L, G, N>& matrix)
{ return matrix.getLocalValuesDevice(Tpetra::Access::OverwriteAll); }

template<class S, class L, class G, class N>
auto read_coefficients(const Tpetra::BlockCrsMatrix<S, L, G, N>& matrix)
{ return matrix.getValuesDevice(); }

template<class S, class L, class G, class N>
auto overwrite_coefficients(Tpetra::BlockCrsMatrix<S, L, G, N>& matrix)
{
    // Trilinos 17.2.1 exposes ReadWrite, not OverwriteAll, for block coefficients.
    return matrix.getValuesDeviceNonConst();
}

} // namespace krona::linalg::detail
