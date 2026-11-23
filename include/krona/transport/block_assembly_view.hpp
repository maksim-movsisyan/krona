#pragma once

namespace krona::transport {

/// @brief Borrows the writable matrix blocks and the locations where face contributions go.
/// @tparam Blocks Rank-three native view indexed by entry, row variable and column variable.
/// @tparam Entries Device map produced by the face assembly plan for those blocks.
template<class Blocks, class Entries>
struct BlockAssemblyView {
    Blocks blocks;   ///< Writable 5x5 blocks; assembly increments their existing coefficients.
    Entries entries; ///< Diagonal entries and off-diagonal entries for each face side.
};

} // namespace krona::transport
