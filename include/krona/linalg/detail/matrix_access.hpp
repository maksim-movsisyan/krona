#pragma once

#include <krona/linalg/types.hpp>

namespace krona::linalg::detail {

// Both matrix types expose flat values in native graph order. These overloads let
// the precision conversion copy those values without depending on the native accessor names.
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
auto overwrite_coefficients(Tpetra::BlockCrsMatrix<S, L, G, N>& matrix) {
    // Trilinos 17.2.1 uses ReadWrite here, so it may sync the old host values first.
    return matrix.getValuesDeviceNonConst();
}

} // namespace krona::linalg::detail
