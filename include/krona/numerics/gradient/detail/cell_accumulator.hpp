#pragma once

#include <krona/numerics/gradient/detail/data.hpp>

namespace krona::numerics::gradient::detail {

/// @brief Holds the running gradient of each field component for one owned cell.
/// @tparam T Scalar type stored in the input and output fields.
/// @tparam N Number of input components processed together.
/// @tparam Space Kokkos memory space of the component descriptors.
/// @tparam Real Scalar type used for differences and accumulation.
template <class T, std::size_t N, class Space, class Real> 
struct CellAccumulator {
    Real gx[N]{}, gy[N]{}, gz[N]{}, self[N];

    KOKKOS_INLINE_FUNCTION CellAccumulator(const Batch<T, N, Space>& fields, std::size_t cell) {
        for (std::size_t c = 0; c < N; ++c)
            self[c] = static_cast<Real>(fields.input[c](cell));
    }

    /// @brief Adds one sample difference times a vector of scheme-specific coefficients.
    KOKKOS_INLINE_FUNCTION void add(const Batch<T, N, Space>& fields, std::size_t sample,
                                    Real x, Real y, Real z) {
        for (std::size_t c = 0; c < N; ++c) {
            const Real delta = static_cast<Real>(fields.input[c](sample)) - self[c];
            gx[c] += x * delta;
            gy[c] += y * delta;
            gz[c] += z * delta;
        }
    }

    /// @brief Converts and writes the finished gradient once per output component.
    KOKKOS_INLINE_FUNCTION void store(const Batch<T, N, Space>& fields, std::size_t cell) const {
        for (std::size_t c = 0; c < N; ++c) {
            fields.x[c](cell) = static_cast<T>(gx[c]);
            fields.y[c](cell) = static_cast<T>(gy[c]);
            fields.z[c](cell) = static_cast<T>(gz[c]);
        }
    }
};

} // namespace krona::numerics::gradient::detail
