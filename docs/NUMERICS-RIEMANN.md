# Riemann solvers

`krona::numerics::riemann` computes approximate inviscid Euler fluxes from two prepared states. It does not reconstruct fields, evaluate boundary conditions, select a time step, or update a residual.

```cpp
#include <krona/numerics/riemann/flux.hpp>
namespace ri = krona::numerics::riemann;
```

`flux.hpp` includes the solvers and thermodynamic adapters. Individual headers are `types.hpp`, `eos.hpp`, `hll.hpp`, `hllc.hpp`, and `roe.hpp`.

## States, orientation and units

`State<R>` contains:

| Member | Meaning |
|---|---|
| `density` | Mass density `rho`. |
| `velocity` | Three velocity components in the global coordinate system. |
| `pressure` | Static thermodynamic pressure. |
| `internal_energy` | Specific internal energy `e`, excluding kinetic energy. |
| `sound_speed` | Adiabatic acoustic speed. |

`R` must be floating-point. Both states and the normal use the same scalar type. Solver calls deduce `R` from them; a solver's parameter type, such as `Roe<Parameter>`, may differ and is converted to `R`.

Call a solver as `solver(left, right, normal)`. **The normal is unit length and points from left to right.** On a mesh face, use owner as left, neighbour as right, and the owner-to-neighbour normal. The implementation checks unit length with a tolerance based on `R` and `GeometryReal`. No coordinate rotation is needed.

`Conservative<R>` is used for both conservative states and fluxes:

| Member | From `conservative(state)` | In a solver's `result.flux` |
|---|---|---|
| `mass` | `rho` | Mass flux per unit face area. |
| `momentum` | `rho * velocity` | Three momentum flux components per unit area, in global coordinates. |
| `energy` | `rho * (e + velocity.norm2()/2)` | Total-energy flux per unit area. |

For identical states, the numerical flux agrees with `physical_flux(state, normal)`:

$$
F_n=\left(\rho u_n,\ \rho u_n\mathbf u+p\mathbf n,
\ (\rho E+p)u_n\right),\qquad
E=e+\tfrac12|\mathbf u|^2.
$$

`conservative()` and `physical_flux()` are direct algebraic helpers; they do not validate inputs.

## Available solvers

| Solver | Behaviour / model restrictions |
|---|---|
| `Rusanov{}` | Local Lax–Friedrichs flux: averages physical fluxes and adds dissipation using `max(abs(u_n) + sound_speed)` over the two states. |
| `Hll<>{}` | Two-wave HLL flux. Uses the left or right physical flux if the entire estimated fan travels in one direction. |
| `Hllc<>{}` | Three-wave HLLC flux with a middle contact wave. Checks its intermediate fan and can fall back to HLL. |
| `Roe{gamma}` | Five-wave ideal-gas Roe flux. Requires one calorically perfect gas with the specified `gamma` and the unshifted energy relation `p = (gamma-1)*rho*e`. Checks that `sound_speed² = gamma*p/rho` within tolerance. |

`Rusanov`, `Hll` and `Hllc` consume the supplied `rho, u, p, e, a` without accessing an EOS. The caller is responsible for their thermodynamic consistency and for selecting an algorithm appropriate to the material; finite inputs alone do not establish physical EOS compatibility.

HLL/HLLC default to `DavisSpeeds`:

$$
S_L=\min(u_{n,L}-a_L,u_{n,R}-a_R),\qquad
S_R=\max(u_{n,L}+a_L,u_{n,R}+a_R).
$$

Roe enables acoustic entropy smoothing and HLL fallback by default. HLLC enables HLL fallback by default. To disable fallback:

```cpp
auto strict_hllc = ri::Hllc<>{ri::DavisSpeeds{}, false};
auto strict_roe = ri::Roe{1.4, true, false}; // gamma, entropy_fix, fallback.
```

Fallback handles selected intermediate-fan/flux failures. Invalid input states/normals, invalid HLL/HLLC outer-wave bounds, and Roe model incompatibility are rejected directly. Roe's fallback uses HLL with Davis speeds; HLLC's uses its configured speed policy. A fallback attempt can itself fail.

## Thermodynamic adapters

The solver state omits temperature and entropy. Prepare it from an EOS or reuse an already evaluated thermodynamic state:

| Adapter | Operation |
|---|---|
| `from_thermodynamics(q, velocity)` | Copies `density`, `pressure`, `internal_energy` and `sound_speed` from `q`, adds velocity, and returns a `State<R>`. `R` is the velocity scalar type. No EOS call, allocation or retained reference. |
| `from_pT(eos, p, T, velocity)` | Calls the generic EOS's `from_pT(p,T)`, then converts its result to a solver state. |
| `from_conservative(eos, U)` | Extracts `rho`, computes velocity and **specific internal energy** `e = U.energy/rho - velocity.norm2()/2`, then uses `from_rhoe(rho,e)`. |
| `ideal_gas_from_rhop(gas, rho, velocity, p)` | Specialized ideal-gas path using density/pressure directly: `e = p/((gamma-1)*rho)`, `a = sqrt(gamma*p/rho)`. |

The supplied `thermodynamics::IdealGas` also has specialized `from_pT` and `from_conservative` overloads. They avoid evaluating unused entropy/logarithms. Invalid inputs produce invalid-state markers (zeros and/or NaNs), which a subsequent solver call rejects. They do not return `Result`/`Status` themselves.

Generic EOS adapters preserve the EOS's own domain/error policy: it may abort on an inadmissible state before a solver can return a status. `from_thermodynamics()` performs conversions, not validation. Keep energy references consistent between the EOS and conservative energy. Negative specific internal energy from a shifted reference is not generally rejected by the state checks, but is incompatible with the supplied Roe solver's energy relation.

## One-face example

```cpp
krona::thermodynamics::IdealGas<double> gas;
gas.validate();

const auto left = ri::from_pT(
    gas, 101325.0, 300.0, ri::Vector<double>{50.0, 0.0, 0.0});
const auto right = ri::from_pT(
    gas, 95000.0, 290.0, ri::Vector<double>{20.0, 0.0, 0.0});
const ri::Vector<double> normal{1.0, 0.0, 0.0};
const auto result = ri::integrated_flux(ri::Hllc<>{}, left, right, normal, 2.0);
krona::mpi::require(result.valid(), "Invalid Riemann flux");
// result.flux already includes face area 2.0.
// result.max_signal_speed remains a velocity.
```

Three call forms are available:

| Call | Geometry / returned flux |
|---|---|
| `solver(left, right, normal)` | Unit normal; flux density, **no area included**. |
| `integrated_flux(solver, left, right, area_vector)` | `area_vector = area * normal`; normalizes it and multiplies the flux by its magnitude. |
| `integrated_flux(solver, left, right, normal, area)` | Unit normal and finite positive area supplied separately. |

Area is applied exactly once. `max_signal_speed` is never area-weighted. When forming `dU/dt = -sum(face_flux)/volume`, an owner-to-neighbour integrated flux is subtracted from the owner and added to an owned neighbour, with the respective cell-volume division. If assembling a residual with another sign convention, apply that convention explicitly.

For higher order, reconstruct left/right states before EOS preparation. At physical boundaries, supply the appropriate interior/exterior inviscid states from the boundary algorithm; the solver does not interpret patches or `CellExtnd` slots.

## Results and failures

`Result<R>` contains `flux`, `max_signal_speed`, and `status`. Always inspect `valid()` before consuming a flux. `used_fallback()` identifies an accepted HLL replacement.

| Status | `valid()` | Meaning |
|---|---|---|
| `Success` | `true` | Selected solver succeeded. |
| `HllFallback` | `true` | HLL fallback succeeded. |
| `InvalidState` | `false` | Rejected state: requires finite positive density, pressure and sound speed, finite velocity/internal energy, and finite conservative variables. |
| `InvalidNormal` | `false` | Rejected normal or integrated-flux area geometry. |
| `IncompatibleModel` | `false` | Roe's perfect-gas relations do not match the state/parameter. |
| `InvalidFan` | `false` | Invalid wave bounds or an intermediate-fan failure without a successful enabled fallback. |
| `NonfiniteFlux` | `false` | Nonfinite flux or invalid/nonfinite reported speed; also covers overflow during area multiplication. |

Failure results returned by the supplied solvers/helpers contain NaNs. `valid()` checks status only; it does not validate a manually constructed result. Invalid solver inputs return statuses instead of throwing.

## Kernel use and runtime selection

Assume current, accessible arrays `left_states`, `right_states` of `State<R>`, `normals` of `Vector<R>`, and positive `areas`, each covering `count` faces. The application owns them. Prepare states, reconstruction ghosts and boundary data before this stage.

```cpp
using Exec = krona::execution::ExecutionSpace;
using R = krona::ComputeReal;
using Space = typename Exec::memory_space;
Exec exec{};
Kokkos::View<ri::Result<R>*, Space> results("riemann/results", count);
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;

auto launch = [&](auto solver) {
    Kokkos::parallel_for("riemann/faces", Policy(exec, 0, count),
        KOKKOS_LAMBDA(std::size_t f) {
            results(f) = ri::integrated_flux(solver,
                left_states(f), right_states(f), normals(f), static_cast<R>(areas(f)));
        });
};

enum class Method { Rusanov, Hll, Hllc, Roe };
const Method method = Method::Hllc; // Select from application configuration.
switch (method) {
case Method::Rusanov: launch(ri::Rusanov{}); break;
case Method::Hll:     launch(ri::Hll<>{}); break;
case Method::Hllc:    launch(ri::Hllc<>{}); break;
case Method::Roe:     launch(ri::Roe{gas.gamma}); break;
}
exec.fence("Riemann results ready");

const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
for (std::size_t f = 0; f < count; ++f)
    krona::mpi::require(host(f).valid(), "Invalid Riemann flux");
```

Here runtime selection happens once on the host; each kernel uses a concrete solver type. Instead of storing all `Result` objects, a flux kernel can consume valid results immediately and record error/fallback flags for later checking. Choose race-free cell accumulation or appropriate atomics when updating shared residuals.

`Hll<Speeds>` and `Hllc<Speeds>` accept a custom device-copyable speed policy with `const operator()(State<R>, State<R>, Vector<R>) -> WaveSpeeds<R>`. Supply finite ordered bounds `left < right` that are physically appropriate; the interface checks finiteness/order, not whether the bounds enclose the true fan. A custom solver can use the same `const operator()(left,right,normal) -> Result<R>` contract and work with `integrated_flux()`.

## Ownership and asynchrony

Built-in states, results and solver policies contain scalar values or small value-owned policies. They retain no field pointers and allocate no arrays. All flux calculations and state adapters execute inline in the caller's host/device context: no independent kernel launch, fence, MPI exchange or mesh traversal. A custom EOS/policy may introduce its own storage or failure behaviour.

The enclosing Kokkos launch can be asynchronous. Keep its captured buffers valid until completion; order producers/consumers on the same execution instance or establish completion explicitly across instances, host access and external MPI. The example fences before reading statuses on the host. This module does not exchange ghosts or synchronize ranks.
