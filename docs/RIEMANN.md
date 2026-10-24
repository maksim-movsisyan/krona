# Euler numerical fluxes

`krona::numerics::riemann` implements local approximate Riemann solvers for the
three-dimensional, single-phase compressible Euler equations. All solvers have
the same pointwise interface. They return five flux components and a signal
speed, without allocating arrays, launching kernels, communicating, or fencing.

The module belongs in `include/krona/numerics/riemann`: it is numerical machinery
used by a spatial operator, independently of mesh storage and equation assembly.
It does not assemble a residual, reconstruct fields, or implement boundary conditions.

## Available solvers

| Type | Wave model | Intended role | Model restriction |
| --- | --- | --- | --- |
| `Rusanov` | One maximum acoustic speed | Simple, dissipative baseline | Prepared acoustic Euler states |
| `Hll<Speeds>` | Two bounding waves | Baseline and fallback | Prepared acoustic Euler states |
| `Hllc<Speeds>` | Two acoustic waves and a contact | General default for resolving contacts and shear | Prepared acoustic Euler states |
| `Roe<Parameter>` | Five-wave Roe linearization, including two shear waves | Low-dissipation alternative | One calorically perfect gas, common constant gamma and zero internal-energy offset |

`Speeds` defaults to `DavisSpeeds`, using endpoint acoustic bounds. This is HLL,
not a separately implemented Einfeldt HLLE solver. AUSM is not implemented.

The types are compile-time policies; there is no virtual dispatch. Different
solvers can be passed to the same kernel template. `Parameter` is the type used
to store Roe's gamma; the arithmetic type comes from the input `State<Real>`.

## State and flux contract

```cpp
#include <krona/numerics/riemann/flux.hpp>
namespace riemann = krona::numerics::riemann;

using Real = krona::ComputeReal;
riemann::State<Real> left, right;
riemann::Vector<Real> n{1, 0, 0};
```

A `State<Real>` contains:

| Member | Meaning |
| --- | --- |
| `density` | Positive density rho |
| `velocity` | Global Cartesian velocity `(u,v,w)` |
| `pressure` | Positive absolute pressure |
| `internal_energy` | Specific internal energy e, not energy density |
| `sound_speed` | Positive acoustic speed a from the EOS |

Both sides must describe the same physical fluid and use consistent units and
energy conventions. HLL/HLLC/Rusanov do not infer or verify an EOS from these
numbers. They accept finite internal energy even when an EOS uses a negative
energy reference. Positive pressure and acoustic speed are still required.
Vacuum and negative-pressure models are outside this interface.

`conservative(state)` returns `Conservative<Real>`:

- `mass = rho`;
- `momentum = rho * velocity`;
- `energy = rho * (e + |velocity|^2/2)`.

The same small value type stores a flux result, where its members instead mean
mass, momentum and total-energy **flux densities**. No field arrays are owned.

For a unit normal n directed from the left state to the right state,

\[
F_n(U)=\left(\rho u_n,\;\rho u_n\mathbf u+p\mathbf n,\;(\rho E+p)u_n\right),
\qquad u_n=\mathbf u\cdot\mathbf n.
\]

`physical_flux(state, n)` implements this expression. It is a low-level helper
without input validation. The numerical solvers validate their inputs.

For mesh faces, left normally means owner and right means neighbour. Use the
same computed face flux with opposite signs in the two cell balances. There is
no need to construct local tangent vectors: momentum remains Cartesian.

## Preparing states with the existing thermodynamics module

The adapter header is `eos.hpp`; core solver headers do not depend on boundary
conditions or thermodynamics. The umbrella `flux.hpp` includes both adapters
and all solvers.

```cpp
krona::thermodynamics::IdealGas<Real> gas;
gas.gamma = static_cast<Real>(1.4);
gas.gas_constant = static_cast<Real>(287.05);
gas.validate(); // Existing host-side model validation.

const auto left = riemann::from_pT(
    gas, Real{101325}, Real{300}, riemann::Vector<Real>{20, 0, 0});
const auto right = riemann::from_pT(
    gas, Real{100000}, Real{300}, riemann::Vector<Real>{18, 0, 0});

// Alternative when density and pressure are already available:
const auto q = riemann::ideal_gas_from_rhop(
    gas, Real{1}, riemann::Vector<Real>{0, 0, 0}, Real{1});

const auto U = riemann::conservative(q);
const auto recovered = riemann::from_conservative(gas, U);
```

Ideal-gas adapters calculate only the needed quantities; they do not calculate
entropy or logarithms. Invalid ideal-gas inputs produce an invalid prepared
state, which a solver reports through `Result::status`.

For another acoustic EOS:

```cpp
const auto thermo = eos.from_pT(pressure, temperature);
const auto q = riemann::from_thermodynamics(thermo, velocity);
// Or directly:
const auto q2 = riemann::from_pT(eos, pressure, temperature, velocity);
```

`from_thermodynamics` reads `density`, `pressure`, `internal_energy` and
`sound_speed`. Generic `from_conservative` calls `eos.from_rhoe(rho,e)`.
Generic adapters retain the supplied EOS's error policy: an EOS may abort on a
state outside its domain. They are not a replacement for EOS domain handling.

HLLC algebra is not tied to an ideal-gas gamma, but this does not establish
admissibility or accuracy for every nonideal EOS. Supply appropriate wave
estimates and validate the combination for the intended model. In particular,
the implementation checks star density, pressure estimates and finite energy;
it does not call the EOS on intermediate states.

The existing constant-density incompressible models are not acoustic Euler
models and are not supported by these fluxes.

## Calling a solver

```cpp
const riemann::Hllc<> solver{};
const auto result = solver(left, right, riemann::Vector<Real>{1, 0, 0});
if (!result.valid()) {
    // Record a failure in the application's device status buffer.
} else {
    const auto mass_flux = result.flux.mass;
    const auto momentum_flux = result.flux.momentum;
    const auto energy_flux = result.flux.energy;
    const auto signal_speed = result.max_signal_speed;
}
```

The normal must be finite and unit length. A small tolerance accounts for
floating-point geometry storage; a general nonunit vector is rejected rather
than silently normalized. Swapping states and reversing the normal reverses
the flux, provided a custom wave-speed policy obeys the same symmetry.

Primary solver calls return flux **per unit area**. Multiply by the face area
once when forming an integrated finite-volume residual. Alternatively:

```cpp
const riemann::Vector<Real> area_vector{face_area, 0, 0};
const auto integrated = riemann::integrated_flux(solver, left, right, area_vector);
```

This helper normalizes the area vector and multiplies the resulting flux by its
magnitude. Zero or nonfinite area is invalid. `max_signal_speed` remains in
velocity units; it is never multiplied by area. Time-step selection remains the
caller's responsibility, including the geometric factors and multidimensional CFL.

## Formulas and wave-speed policies

Rusanov uses

\[
\widehat F=\tfrac12(F_L+F_R)-\tfrac12 s(U_R-U_L),\qquad
s=\max(|u_{n,L}|+a_L,|u_{n,R}|+a_R).
\]

The default Davis bounds are

\[
S_L=\min(u_{n,L}-a_L,u_{n,R}-a_R),\qquad
S_R=\max(u_{n,L}+a_L,u_{n,R}+a_R).
\]

HLL selects the physical upwind flux outside the fan and uses

\[
\widehat F_{HLL}=
\frac{S_RF_L-S_LF_R+S_LS_R(U_R-U_L)}{S_R-S_L}
\]

inside it. HLLC additionally estimates the contact speed and the two star
states from the Rankine–Hugoniot relations. This restores contact and tangential
velocity transport lost in the two-wave HLL approximation.

A custom policy has this device-callable interface:

```cpp
struct MySpeeds {
    template<class R>
    KOKKOS_INLINE_FUNCTION
    riemann::WaveSpeeds<R> operator()(riemann::State<R> left,
                                    riemann::State<R> right,
                                    riemann::Vector<R> normal) const;
};
// MySpeeds must be implemented before these objects are used:
// riemann::Hll<MySpeeds> hll{policy};
// riemann::Hllc<MySpeeds> hllc{policy, true};
```

Bounds must be finite, ordered and physically appropriate. Finite ordered values
alone do not prove they bound the true fan. HLL/HLLC report the maximum absolute
bound as their signal speed. Rusanov uses its endpoint bound; Roe also includes
its corrected acoustic eigenvalue magnitudes and endpoint bound.

## Roe: model and entropy correction

```cpp
riemann::Roe<Real> roe{gas.gamma};
// Constructor arguments: gamma, enable entropy fix, enable HLL fallback.
riemann::Roe<Real> strict_roe{gas.gamma, true, false};
```

Construct Roe on the host and copy it into a kernel. Its constructor rejects
nonfinite gamma or gamma <= 1. Each call also checks compatibility of both
prepared states with `p=(gamma-1)*rho*e` and `a^2=gamma*p/rho`, within a relative
floating-point tolerance. An EOS mismatch produces `IncompatibleModel`; it does
not silently fall back to a solver for a different physical model.

Roe averages velocity and total enthalpy using square-root-density weights:

\[
\widetilde{\mathbf u}=
\frac{\sqrt{\rho_L}\mathbf u_L+\sqrt{\rho_R}\mathbf u_R}
{\sqrt{\rho_L}+\sqrt{\rho_R}},\qquad
\widetilde a^2=(\gamma-1)(\widetilde H-|\widetilde{\mathbf u}|^2/2).
\]

The flux subtracts the characteristic dissipation of the two acoustic waves,
the contact and the two shear waves. The latter are evaluated as a Cartesian
vector contribution, avoiding tangent-basis construction.

By default, acoustic absolute eigenvalues use adaptive Harten-type smoothing.
For each acoustic family, with Roe eigenvalue lambda and endpoint eigenvalues
lambda_L and lambda_R:

\[
\delta=\max(0,\lambda-\lambda_L,\lambda_R-\lambda),\qquad
|\lambda|_\delta=
\begin{cases}
\frac12(|\lambda|^2/\delta+\delta),&|\lambda|<\delta,\\
|\lambda|,&\text{otherwise}.
\end{cases}
\]

Contact and shear eigenvalues are not smoothed. This avoids applying a fixed
acoustic smoothing width to stationary contacts and compressive shocks. The
option is an entropy correction, not a proof of a global discrete entropy
inequality for the full reconstructed scheme.

## Fallbacks and errors

HLLC and Roe enable an HLL fallback by default. HLLC falls back when its star fan
has invalid ordering, nonpositive estimated pressure/density, or nonfinite
conserved quantities. Roe checks positive density and internal-energy density
in its two acoustic intermediate states, and falls back if these or its average
are invalid. Both also fall back on a nonfinite computed flux.

HLLC uses its configured speed policy for fallback. Roe uses Davis bounds.
Fallback does not repair invalid input states. Invalid custom bounds are
reported directly; retrying HLL with the same invalid bounds would not help.

```cpp
riemann::Hllc<> strict_hllc{riemann::DavisSpeeds{}, false};
if (result.used_fallback()) {
    // Optionally count this face for diagnostics.
}
```

| Status | Meaning |
| --- | --- |
| `Success` | Requested solver produced a valid result |
| `HllFallback` | Valid result produced by the fallback HLL flux |
| `InvalidState` | Invalid/nonfinite input or unrepresentable conservative state |
| `InvalidNormal` | Invalid unit normal or area vector |
| `IncompatibleModel` | Roe gamma/model compatibility check failed |
| `InvalidFan` | Invalid wave bounds, or invalid fan with fallback disabled |
| `NonfiniteFlux` | Final flux or signal speed is nonfinite/unrepresentable |

Only `Success` and `HllFallback` satisfy `valid()`. Invalid results contain NaNs
in the flux and signal speed. The solver does not abort, synchronize, or write a
shared error flag: the calling kernel chooses how to collect and handle errors.
Checking only `used_fallback()` is insufficient; always check `valid()`.

No solver here guarantees positive updated density/pressure for arbitrary
reconstruction, arbitrary EOS, arbitrary wave-speed estimates, or arbitrary
CFL. A fallback is local protection, not a positivity-preserving time integrator.

## Kokkos integration and batching

The solver operates on one prepared pair of states. A batch is a caller-owned
array of faces processed by a normal Kokkos kernel; no special batch allocation
or intermediate flux array is required by the module.

```cpp
template<class Solver, class States, class Normals, class Results>
struct FaceFluxes {
    Solver solver;
    States left, right;
    Normals normals;
    Results results;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t f) const
    {
        results(f) = solver(left(f), right(f), normals(f));
    }
};

using Exec = krona::execution::ExecutionSpace;
using Space = typename Exec::memory_space;
using States = Kokkos::View<riemann::State<Real>*, Space>;
using Normals = Kokkos::View<riemann::Vector<Real>*, Space>;
using Results = Kokkos::View<riemann::Result<Real>*, Space>;
// Allocate and populate left/right/normals before submitting the kernel.
States left_states("left", nfaces), right_states("right", nfaces);
Normals normals("normals", nfaces);
Results results("results", nfaces);
Exec exec;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
Kokkos::parallel_for("euler/face_flux", Policy(exec, 0, nfaces),
    FaceFluxes<riemann::Hllc<>, States, Normals, Results>{
        {}, left_states, right_states, normals, results});
```

In a production operator, gather reconstructed primitive components from your
existing field views, prepare the two states locally, call the solver and
immediately accumulate/store the flux as appropriate. This avoids allocating
`State` arrays solely to use this interface. Momentum and energy must be
constructed consistently from the reconstructed primitive state and EOS.

A boundary operator supplies the appropriate exterior state; the Riemann solver
does not interpret patch types. A prescribed face trace and an exterior state
are different inputs: use the exterior/mirror state intended for inviscid flux
calculation by the boundary implementation.

The pointwise function executes synchronously within its calling thread. The
surrounding `parallel_for` may be asynchronous. Submit dependent work on the
same execution-space instance, keep all referenced fields alive, and complete
halo exchange before reading reconstructed ghost inputs. Synchronize explicitly
before host inspection or buffer reuse across unordered execution contexts.
No synchronization is inserted by this module.

## Installing and running tests

Copy `include/` into the existing KRONA include tree. All headers are new; no
existing transport, boundary or EOS files are replaced.

Copy `tests/numerics/riemann/` into the matching test directory and add
`add_subdirectory(riemann)` in `tests/numerics/CMakeLists.txt`. If numerics has no
intermediate test directory, add `add_subdirectory(numerics/riemann)` from the
existing tests root instead. Use one route only.

The supplied CMake file uses the existing `krona_test_support` and
`krona::thermodynamics` targets. It registers six named tests and passes their
exact names to the existing test runner.

```sh
ctest --test-dir build/your-profile -R '^numerics\.riemann\.' --output-on-failure
```

Tests cover float/double consistency, normal reversal, rotational covariance,
area scaling, contact/shear transport, a stationary shock, strong-rarefaction
fallbacks, model mismatch, EOS conversion, Kokkos kernel execution, and a small
first-order finite-volume Sod calculation with analytic-density and conservation
checks. The Sod loop is test infrastructure, not a production time integrator.

This delivery was compiled with the available Trilinos/Kokkos headers and tested
on Kokkos Serial. CUDA/HIP compilation and execution were not available here;
the kernel test is included to exercise those backends in the project build.

## Limits and future work

- Fixed faces only: no ALE/grid velocity, moving-face energy flux or geometric
  conservation law is implemented.
- Five Euler variables only: passive-scalar/species fluxes, reacting mixtures,
  and multiphase models require an explicit extension.
- No automatic positivity limiter, low-Mach correction or carbuncle treatment.
- No flux Jacobians or implicit linearization API yet.
- HLLC wave-speed policies are replaceable. More specialized bounds can be added
  without changing the calling operator.
- General-EOS Roe requires a different thermodynamic linearization; changing
  gamma alone is not such an extension.
- AUSM-family methods can implement the same pointwise interface later.

## References

- [Clawpack Riemann book: approximate Euler solvers](https://www.clawpack.org/riemann_book/html/Euler_approximate.html)
  — Roe linearization, entropy and positivity issues.
- [E. F. Toro: HLLC Riemann solver lecture](https://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf)
  — HLL/HLLC wave structure and star-state construction.
- [Batten et al., On the Choice of Wavespeeds for the HLLC Riemann Solver](https://doi.org/10.1137/S1064827593260140)
  — the importance of wave-speed estimates.
