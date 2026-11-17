#pragma once

#include <krona/mpi/communicator.hpp>
#include <krona/linalg/csr_pattern.hpp>
#include <krona/mesh/host/mesh_part.hpp>

#include <span>


namespace krona::linalg {

/// @brief Maps original local cell indices to global algebraic IDs for owned and ghost cells.
/// Owned IDs form a contiguous range on each rank. Ghosts carry the IDs assigned by their owners.
struct CellNumbering {
    std::size_t n_owned = 0; ///< Length of the owned prefix in cell_to_global.
    GlobalOrdinal owned_begin = 0; ///< First ID of this rank's owned range.
    GlobalOrdinal global_cells = 0; ///< Total owned cells across all ranks.
    std::vector<GlobalOrdinal> cell_to_global;

    /// @brief Returns the owned ID prefix in local row order, borrowing cell_to_global.
    [[nodiscard]] std::span<const GlobalOrdinal> owned_gids() const noexcept
    { return {cell_to_global.data(), n_owned}; }

    /// @brief Expands each local cell to consecutive component DOFs without sorting cells.
    [[nodiscard]] std::vector<GlobalOrdinal> expanded_gids(int components) const;
};

/// @brief Assigns contiguous algebraic IDs to owned cells and exchanges them with ghost holders.
/// @param mesh Local partition with the host send and receive lists for the cell halo.
/// @param comm Communicator whose rank ordering matches the mesh partition.
/// @details All ranks participate. IDs are assigned from current ownership and local order, independently
/// of the original global mesh IDs. The function waits for the exchange before returning.
[[nodiscard]] CellNumbering build_cell_numbering(const mesh::host::MeshPart& mesh, mpi::CommView comm);

/// @brief Builds a host CSR graph with a row per owned cell, its diagonal, and its face neighbours.
/// @param mesh Local partition providing owner and neighbour indices for each face.
/// @param numbering Algebraic IDs for the same local cell ordering, including ghosts.
/// @details A physical boundary face adds no column. Several faces joining the same pair share one entry.
[[nodiscard]] CsrInput make_face_csr(const mesh::host::MeshPart& mesh, const CellNumbering& numbering);

} // namespace krona::linalg
