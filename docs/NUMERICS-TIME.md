# Time derivatives

`numerics/time/derivative.hpp` provides coefficients for implicit time derivatives. It does not advance a solution, store time layers, or assemble a matrix.

The supplied formulas are first-order Backward Euler and second-order BDF2, including unequal adjacent steps.

```cpp
#include <krona/numerics/time/derivative.hpp>
namespace nt = krona::numerics::time;
```

## Coefficients and ordering

`Derivative<Real, History>` owns `History + 1` scalar weights:

$$
\left.\frac{dq}{dt}\right|_{n+1}\approx
w_0q^{n+1}+w_1q^n+w_2q^{n-1}+\cdots.
$$

- `weights[0]` multiplies the current unknown, `q^(n+1)`.
- Remaining weights multiply history values, **newest first**.
- `history_size` counts old layers only, excluding the current unknown.
- Weights already contain the inverse time step and have units `1/time`. Do not divide the result by `dt` again.

| Call | Required old layers | Weights |
|---|---|---|
| `BackwardEuler{}.coefficients(dt)` | `q^n` | `{1/dt, -1/dt}` |
| `Bdf2{}.coefficients(dt)` | `q^n`, `q^(n-1)` | `{3/(2*dt), -2/dt, 1/(2*dt)}`; assumes equal adjacent steps |
| `Bdf2{}.coefficients(dt, previous_dt)` | `q^n`, `q^(n-1)` | BDF2 weights for unequal adjacent steps |

`dt = t^(n+1) - t^n`; `previous_dt = t^n - t^(n-1)`. The returned scalar type is deduced from the floating-point step arguments; both arguments of variable-step BDF2 must have the same type.

## Example

The application supplies the current value and history; this example applies variable-step BDF2 to `q(t) = t*t` at times `-0.2`, `0`, and `0.1`:

```cpp
const double dt = 0.1, previous_dt = 0.2;
const auto derivative = nt::Bdf2{}.coefficients(dt, previous_dt);

const double q_np1 = 0.01;
const double q_n = 0.0;
const double q_nm1 = 0.04;

const double dq_dt = derivative.weights[0]*q_np1
                   + derivative.weights[1]*q_n
                   + derivative.weights[2]*q_nm1;
// dq_dt ~= 0.2.
```

For field evaluation, compute the coefficients once on the host and capture the returned `Derivative` by value in your kernel. Recompute them when the relevant step sizes change. The scalar transient transport module can also consume this coefficient object when assembling its time term.

BDF2 requires two valid history layers. Startup and history rotation are the application's responsibility; Backward Euler can be used for the first step. No automatic method selection, error estimate, step acceptance, or step-ratio restriction is implemented.

## Custom coefficients and validation

A custom formula can directly construct `Derivative<Real, History>` with explicit weights. `Real` must be floating-point and `History > 0`. For the current implicit-consumer convention, supply finite weights with `weights[0] > 0`. Preserve constants by ensuring `sum(weights) = 0` up to floating-point error, and ensure that the coefficients have the intended derivative accuracy on the actual time levels.

Direct aggregate construction performs no runtime validation. The supplied factories check finite positive steps, finite coefficients and a positive current coefficient; variable-step BDF2 also checks that `dt/previous_dt` is finite and positive. They do not explicitly check the weight sum.

## Execution and ownership

Coefficient factories are synchronous host calculations. They allocate no dynamic storage, launch no kernels, perform no MPI communication and issue no fence. The returned object owns its small `Kokkos::Array` of weights and borrows no field buffers. Any asynchronous execution and buffer-lifetime requirements belong to the kernel/operator that consumes it.
