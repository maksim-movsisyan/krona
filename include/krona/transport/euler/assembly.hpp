#pragma once

#include <krona/linalg/bsr_matrix.hpp>
#include <krona/linalg/face_assembly.hpp>
#include <krona/transport/euler/jacobian.hpp>

namespace krona::transport::euler {

/// @brief Opens native ReadWrite access to the 5x5 blocks without clearing or committing.
/// The assembly plan must use the Euler mesh's local cell/face numbering.
template<class Scalar, class Exec>
[[nodiscard]] auto make_jacobian_view(linalg::BsrMatrix<Scalar, Exec>& matrix,
    const linalg::FaceAssemblyPlan<Exec>& assembly, const Exec& exec = {}) {
    mpi::require(matrix.pattern().native().getRawPtr() == assembly.pattern().native().getRawPtr(),
                 "Euler: matrix must share the face assembly graph");
    mpi::require(matrix.block_size() == 5, "Euler: Jacobian requires 5x5 blocks");
    return JacobianView{matrix.edit_blocks(exec), assembly.entry_map()};
}

} // namespace krona::transport::euler
