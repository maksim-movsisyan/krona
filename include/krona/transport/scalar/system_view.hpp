#pragma once

#include <krona/execution/types.hpp>

namespace krona::transport::scalar {

/// @brief Borrowed scalar equation slice: native-entry coefficients, owned RHS and matching entry map.
/// Arrays are writable rank-one views, including strided BSR diagonal-component slices.
/// Initialize targets once before adding contributions; make_system_view() checks graph/distribution attachment.
template<class Values, class Rhs, class Entries>
struct SystemView {
    Values values;
    Rhs rhs;
    Entries entries;
};

} // namespace krona::transport::scalar
