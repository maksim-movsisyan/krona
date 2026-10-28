# Scalar transient operator

The module adds an implicit time derivative to a scalar transport equation.
It supplies backward Euler and BDF2 coefficients, additive system assembly,
and the matching matrix-free residual contribution. It does not own a matrix,
solve a system, choose time steps, or maintain solution history.

```cpp
#include <krona/numerics/time/derivative.hpp>
#include <krona/transport/scalar/transient/operator.hpp>

namespace nt = krona::numerics::time;
namespace ts = krona::transport::scalar;
namespace tt = krona::transport::scalar::transient;
using Real = krona::ComputeReal;
```

## Equation and signs

The differential storage term is `d(rho*phi)/dt`. After integration over a
stationary control volume, its discrete representation is

\[
\frac{dQ_i}{dt},\qquad Q_i=M_i\phi_i,\qquad M_i=\rho_iV_i.
\]

Here `phi` is the transported scalar; multiplying by volume does not change
its interpretation. For a specific quantity, `M*phi` represents the cell's
total transported quantity. This relation is exact if `phi` is the corresponding
mass-weighted cell average; multiplying independently volume-averaged density
and scalar is a discrete approximation to the integral of their product.

The module uses the same integrated convention as the spatial operators:
`A*phi=b`, with residual `r=b-A*phi`. No contribution is divided by volume.
For coefficients `w[j]`, it adds

\[
A_{ii}\mathrel{+}=w_0M_i^{n+1},\qquad
b_i\mathrel{+}=-\sum_{j=1}^{N}w_jM_i^{n+1-j}\phi_i^{n+1-j}.
\]

The residual operation adds the historical RHS above minus
`w[0]*M_new*phi_new`. Contributions preserve existing target values.

## Schemes

`Derivative<Real, N>` contains `N+1` dimensional coefficients in `weights`;
`history_size=N` is the number of required past layers. Index 0 always refers
to the new unknown layer. Weights already include division by time step.

| Scheme | Factory | Required past layers | Weights for constant step |
| --- | --- | --- | --- |
| Backward Euler | `nt::BackwardEuler{}.coefficients(dt)` | 1 | `(1, -1)/dt` |
| BDF2 | `nt::Bdf2{}.coefficients(dt)` | 2 | `(3/2, -2, 1/2)/dt` |
| Variable-step BDF2 | `nt::Bdf2{}.coefficients(dt, previous_dt)` | 2 | Depends on both steps |

For variable-step BDF2, let `r=dt/previous_dt`. The coefficients are

\[
(w_0,w_1,w_2)=\frac{1}{\Delta t}
\left(\frac{1+2r}{1+r},\;-(1+r),\;\frac{r^2}{1+r}\right).
\]

`dt` spans the new step; `previous_dt` spans the last accepted step. Use Euler
for startup when only one accepted layer exists, then BDF2 when two are available.
Startup selection is explicit; the module does not switch schemes automatically.

Factories execute on host and throw `std::invalid_argument` for nonpositive or
nonfinite steps and unrepresentable coefficients/ratios. Custom `Derivative`
objects must have finite weights, a positive current coefficient, and a
consistent derivative formula. In particular, their weights must sum to zero
to preserve constants; that consistency condition is the caller's responsibility.

## Mass and accepted history

Every mass provider implements `mass(i)` and returns **integrated cell mass**,
not density:

```cpp
auto stored = tt::CellMass{integrated_mass_view};
auto computed = tt::DensityVolumeMass{cell_volumes, density_view};
auto constant_density = tt::DensityVolumeMass{cell_volumes, Real{1}};
auto constant_mass = tt::UniformMass{Real{2}};
```

`DensityVolumeMass` accepts either a scalar density or a rank-one density view
and computes the product on demand, without allocating a mass array. A custom
device-copyable provider may compute mass in another way. For an equation
without density in its storage term, use `M=V`, for example `CellMass{cell_volumes}`.
`UniformMass{1}` means unit mass per cell, not unit density on arbitrary volumes.

`TimeLevel{mass, field}` describes one accepted layer; `field` is a rank-one
Kokkos view of a scalar component. Each layer supplies its own mass, allowing
density to change between steps:

```cpp
auto previous = tt::TimeLevel{mass_previous, phi_previous};
auto older = tt::TimeLevel{mass_older, phi_older};
auto euler_history = tt::time_history(previous);
auto bdf_history = tt::time_history(previous, older); // Newest accepted first.
```

History is a nested value structure whose length is known at compile time;
it allocates no arrays. The length must match the derivative's history size,
otherwise compilation fails. Descriptors copy view handles, **not array data**.
They do not snapshot, rotate, or accept layers. Constant mass data may be
shared across layers; varying density must retain its correct past values.

## Usage

The following CSR example assumes that the existing mesh/algebra setup has
provided `matrix`, `rhs`, `assembly`, `comm`, `exec`, and the indicated input
views. All views use the same local owned-cell ordering.

```cpp
#include <krona/transport/scalar/assembly.hpp>

using Exec = krona::execution::ExecutionSpace;
tt::TransientOperator<Exec, Real> temporal(owned_count, exec);
ts::AssemblyStatus<Exec> status(exec);

auto mass_new = tt::DensityVolumeMass{cell_volumes, density_new};
auto previous = tt::TimeLevel{
    tt::DensityVolumeMass{cell_volumes, density_previous}, phi_previous};
auto history = tt::time_history(previous);
auto derivative = nt::BackwardEuler{}.coefficients(dt);

auto target = ts::make_system_view(matrix, rhs, assembly, exec);
Kokkos::deep_copy(exec, target.values, 0);
Kokkos::deep_copy(exec, target.rhs, 0);

status.begin_async();
temporal.add_system_async(target, status, derivative, mass_new, history);
// Add convection, diffusion and sources to the same initialized system.
static_cast<void>(status.check(comm));
matrix.commit();
rhs.commit();
// Solve using the existing linear algebra interface.
```

For subsequent BDF2 steps, pass two layers and the matching coefficients:

```cpp
auto older = tt::TimeLevel{mass_older, phi_older};
auto history_bdf = tt::time_history(previous, older);
auto derivative_bdf = nt::Bdf2{}.coefficients(dt, previous_dt);

// Inside a new, initialized assembly batch:
temporal.add_system_async(target, status, derivative_bdf, mass_new, history_bdf);
```

For BSR, obtain the target with
`make_system_view(matrix, rhs, assembly, exec, component)`. The temporal operator
adds only to the selected equation/variable diagonal slice and its AoS RHS
component. Other components remain unchanged. Initialize the complete coupled
system separately as required.

For a matrix-free residual, initialize the output once and add temporal and
spatial contributions to it:

```cpp
Kokkos::deep_copy(exec, residual, 0);
status.begin_async();
temporal.add_residual_async(
    phi_new, residual, status, derivative_bdf, mass_new, history_bdf);
// Add spatial/source residual contributions before checking the batch.
static_cast<void>(status.check(comm));
```

## Contracts and asynchronous execution

- Kokkos must be initialized. The operator and status must use the same
  execution-space instance. The operator itself stores only an owned-cell
  count and an execution instance.
- Inputs/outputs are accessible rank-one views, including strided component
  views. Inputs need at least `owned_count` entries; ghosts and boundary slots
  are not read. System RHS and diagonal map must have exactly that count.
- Each diagonal map entry must identify the correct native coefficient and
  be distinct from the other rows' entries. Handcrafted targets retain this
  responsibility; duplicate entries are not detected by this operator.
- Masses must be finite and positive, layer values finite, and accumulated
  contributions representable in the target precision. Built-in view providers
  check extents/accessibility on host; custom providers own those contracts.
- Inputs must not overlap written targets, including strided aliases. The
  residual method rejects identical starting pointers, but does not perform
  comprehensive overlap detection.

Contribution calls add no allocations, fences, MPI operations, halo exchanges,
or algebra commits. They submit work on the stored execution instance; Serial
may naturally execute it synchronously. Obtaining native algebra views retains
the algebra wrapper's own synchronization behavior.

Call `status.begin_async()` once before a batch. Device checks record errors
without a host round trip per cell. `status.check(comm)` copies diagnostics,
fences its execution instance and performs a blocking MPI all-reduce; every
rank in the communicator must participate, including empty ranks. It does not
commit algebra wrappers. Failed assembly is not transactional: discard/rebuild
the targets rather than using partially updated data. A pending status also
fences its instance on destruction.

Keep all input storage alive and unchanged until its readers complete. Order
dependent copies/kernels on the same execution instance; establish completion
explicitly when crossing instances. Keep accepted history fixed through all
nonlinear/deferred-correction iterations of a physical step. Only after step
acceptance, copy/rotate the history buffers and update `previous_dt`; rejected
steps must leave history intact. Queue reuse after the buffers' last readers.

Rebuilding a system requires reinitializing its targets; repeatedly appending
the same temporal term double-counts it. Changes to step size or current mass
change the matrix diagonal and must follow the existing matrix revision and
preconditioner preparation rules.
The temporal contribution does not set `AssemblyReport::has_diffusion_anchor`;
that boundary flag alone does not determine solvability of a transient system.

## Limits and future extensions

The module provides a frozen-mass implicit contribution, not a full nonlinear
Jacobian: derivatives of a state-dependent mass are not assembled. Density
updates, discrete continuity, nonlinear convergence and step acceptance belong
to the calling solver. Preserving a uniform scalar with varying density also
requires compatible mass fluxes and a consistent discrete continuity equation.

Variable-step coefficients do not guarantee stability for arbitrary step
sequences; no step-ratio policy, adaptive error control, CFL selection or BDF2
positivity/TVD guarantee is implemented. RK/IMEX stage execution is not present.
Cell ordering must remain fixed during history use. Moving-mesh GCL and history
remapping after adaptation/repartitioning require additional work.

Possible extensions are a separate time-loop/history owner with explicit step
acceptance, error-controlled step selection, and RK/IMEX drivers. These can be
added without making the current additive operator own the algebra system.
