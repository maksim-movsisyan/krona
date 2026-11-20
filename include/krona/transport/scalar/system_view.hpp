#pragma once

#include <krona/execution/types.hpp>

namespace krona::transport::scalar {

/// @brief Borrows a matrix coefficient slice, its RHS and the entry map for scalar assembly.
/// Initialize the writable slices before the first additive contribution.
/// @tparam Values Writable rank-one matrix view indexed by native entry ID.
/// @tparam Rhs Writable rank-one RHS view indexed by owned cell ID. The view can be strided.
/// @tparam Entries Map from local rows and faces to entries in values.
template<class Values, class Rhs, class Entries>
struct SystemView {
    Values values;   ///< Matrix coefficients for the selected scalar equation.
    Rhs rhs;         ///< Right-hand side for the same scalar equation.
    Entries entries; ///< Entry map for the layout and numbering of this matrix.
};

} // namespace krona::transport::scalar
