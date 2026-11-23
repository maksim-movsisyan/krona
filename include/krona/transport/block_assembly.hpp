#pragma once

#include <krona/linalg/bsr_matrix.hpp>
#include <krona/linalg/face_assembly.hpp>
#include <krona/transport/block_assembly_view.hpp>

namespace krona::transport {

/// @brief Opens native ReadWrite access to the 5x5 blocks without clearing or committing.
/// The assembly plan must use the operator mesh's local cell and face numbering.
/// @tparam Scalar Stored matrix coefficient type.
/// @tparam Exec Execution space used to access the native blocks.
/// @param matrix BSR matrix from the same graph as assembly.
/// @param assembly Plan that maps an owned cell or face side to a matrix entry.
/// @param exec Instance on which the assembly kernels will use the view.
template<class Scalar, class Exec>
[[nodiscard]] auto make_block_assembly_view(linalg::BsrMatrix<Scalar, Exec>& matrix,
                                            const linalg::FaceAssemblyPlan<Exec>& assembly, const Exec& exec = {}) {
    mpi::require(matrix.pattern().native().getRawPtr() == assembly.pattern().native().getRawPtr(),
                 "Block assembly: matrix must share the face assembly graph");
    mpi::require(matrix.block_size() == 5, "Block assembly: Jacobian requires 5x5 blocks");
    return BlockAssemblyView{matrix.edit_blocks(exec), assembly.entry_map()};
}

} // namespace krona::transport
