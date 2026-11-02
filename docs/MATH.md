# Small 3D math

`krona::math` provides inline 3D value types and an SPD matrix inverse. Include `<krona/math/math.hpp>` for everything, or a specific header below.

| Header / type | Storage and operations |
| --- | --- |
| `vector3.hpp` — `Vector3<Real>` | `x, y, z`; indexed access `v[i]`; addition, subtraction, scaling and division. Member/free `dot`, `cross`, `norm2`, `norm`. |
| `matrix3.hpp` — `Matrix3<Real>` | Row-major `data[row][column]`, also accessed as `m(row,column)`; zero by default. `identity()`, matrix-vector and matrix-matrix products, arithmetic, `transpose`, `trace`, `determinant`, `outer(a,b)`. |
| `symmetric_tensor3.hpp` — `SymmetricTensor3<Real>` | Six values in order `xx, yy, zz, xy, xz, yz`; symmetric indexed access, arithmetic, tensor-vector product, `to_matrix`, `trace`, `symmetric_outer`, `double_contract`, `quadratic_form`. |
| `inverse_spd.hpp` — `inverse_spd` | Checked Cholesky inverse of a symmetric positive-definite 3-by-3 matrix; returns `bool`. |

All types store their values directly, default to zero, and are trivially copyable aggregates. Scalar types are floating-point and explicit, for example `Vector3<double>`; no project-wide precision alias is imposed. Use `float` or `double` for portable host/device arithmetic. Vector and matrix indices are zero-based and must be below 3. Index assertions are debug checks, not a release-mode validation API.

All three types support `+`, `-`, unary `-`, multiplication by a scalar on either side, division by a scalar, and compound assignments `+=`, `-=`, `*=`, `/=`. Compound assignments retain the destination precision. Division requires a nonzero divisor. Other arithmetic follows ordinary floating-point rules and may produce nonfinite results; `isfinite(value)` is available for each type.

## Use and precision

Inside a host function or device kernel:

```cpp
#include <krona/math/math.hpp>
namespace math = krona::math;

math::Vector3<double> a{1, 2, 3}, b{4, -1, 2};
const auto product = math::dot(a, b);       // 8; a.dot(b) is also available.
const auto normal = math::cross(a, b);      // {7, 10, -9}.
const auto length = math::norm(a);          // sqrt(14).
const auto tensor = math::symmetric_outer(a); // a * a^T, six stored entries.
const auto full = math::to_matrix(tensor);
const auto projected = full*b;

math::Matrix3<double> matrix{{{4, 1, 0}, {1, 3, 0}, {0, 0, 2}}};
math::Matrix3<double> inverse;
const bool ok = math::inverse_spd(matrix, inverse);
if (ok) {
    const auto solution = inverse*a;
    // Use solution.
}
```

Mixed binary operations use `std::common_type_t` of their operand scalar types. For example, `Vector3<float> * 2.0` produces `Vector3<double>`, while multiplication by `2` retains `float`. `cast<Real>(value)` explicitly converts any of the three value types. To choose dot-product arithmetic independently of storage, use `dot_as<Real>(a,b)`; conversion happens before multiplication and accumulation.

`norm()` uses nested two-argument `hypot` to avoid unnecessary overflow/underflow from squaring components. `norm2()` is an ordinary dot product and can overflow or underflow.

Symmetric off-diagonal entries are stored **once, without a factor of two**. `s(0,1)` and `s(1,0)` access the same `xy` member. `double_contract(s,t)` computes the full sum over matrix entries, including twice each off-diagonal product. `quadratic_form(s,v)` computes `v.dot(s*v)`. Symmetric storage does not imply positive definiteness.

## SPD inverse contract

The same `inverse_spd(input, output, tolerance)` interface accepts:

- `Matrix3<Real>` input/output;
- `SymmetricTensor3<Real>` input/output; the output stores the computed upper triangle once;
- raw row-major arrays or pointers to nine `Real` entries, indexed as `3*row+column`.

Input and output use the same scalar type. In-place inversion is supported. Raw pointers must address nine accessible entries; null pointers return `false`.

The default tolerance is `64 * epsilon<Real>`. Tolerance is converted to the matrix scalar type and must then be finite and strictly between zero and one. Every Cholesky diagonal residual must exceed `tolerance * trace(input)`, before taking its square root. This is a pivot rejection threshold, not a requested inverse-error bound.

Full matrices must be exactly symmetric and every input entry must be finite. There is no implicit symmetrization, rescaling or regularization. The original gradient module's Cholesky arithmetic is retained inside the checked wrapper.

`false` means invalid input/tolerance, a rejected Cholesky pivot or a nonfinite inverse. Output is unchanged on failure. Extreme numerical scales can cause failure through intermediate overflow. A successful result is finite; the threshold does not guarantee a particular condition number or forward error. The caller decides whether failure needs a fallback or a fatal application error.

## Execution and dependencies

Operations are host/device callable and compute directly in the caller's context. They allocate no dynamic storage, launch no kernels, copy no external buffers, perform no MPI calls and introduce no synchronization. The module depends on Kokkos headers/math support, with no dependency on KRONA's configuration, execution-space selection, fields, mesh or linear algebra backend. Host arithmetic requires neither MPI nor Kokkos initialization.

Arithmetic accessors/operations are `constexpr` where applicable. Norms, finiteness checks and inversion run at runtime. Public operations are `noexcept`; the inverse reports failure through its return value. Existing modules are not automatically migrated by including these headers.
