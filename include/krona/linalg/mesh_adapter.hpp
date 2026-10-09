#pragma once

#include <krona/linalg/csr_pattern.hpp>
#include <krona/mesh/localmesh.hpp>

#include <span>


namespace krona::linalg {

/// @brief Rank-contiguous cell GIDs, indexed by KRONA's original local cell IDs.
struct CellNumbering {
    std::size_t n_owned = 0;
    GlobalOrdinal owned_begin = 0;
    GlobalOrdinal global_cells = 0;
    std::vector<GlobalOrdinal> cell_to_global;

    [[nodiscard]] std::span<const GlobalOrdinal> owned_gids() const noexcept
    { return {cell_to_global.data(), n_owned}; }

    /// @brief Expands each local cell to consecutive component DOFs without sorting cells.
    [[nodiscard]] std::vector<GlobalOrdinal> expanded_gids(int components) const;
};

/// @brief Collective prefix numbering and one halo exchange on a private MPI context.
/// No Allgather table, original mesh GIDs, geometry or device memory are needed.
[[nodiscard]] CellNumbering build_cell_numbering(const mesh::MeshPart& mesh, MPI_Comm comm);

/// @brief Scalar face-neighbour graph, including each owned diagonal, excluding physical boundary slots.
/// Connectivity is a graph choice, not a discretization or boundary-condition assembly.
[[nodiscard]] CsrInput make_face_csr(const mesh::MeshPart& mesh, const CellNumbering& numbering);

} // namespace krona::linalg
