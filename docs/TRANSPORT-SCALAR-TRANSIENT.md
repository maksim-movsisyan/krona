# Scalar transient term

`krona::transport::scalar::transient` adds an implicit time derivative to a scalar equation or evaluates its residual. It works on owned cells and takes the time coefficients, current mass and accepted history from the application. It owns no mesh, matrix, field or history storage and does not advance time or solve a system itself.

The `krona::transport_scalar_transient` target depends only on execution/Kokkos and MPI support. Native linear algebra is optional and enters through the separate assembly adapter.

## Discretized quantity and signs

The operator discretizes the **integrated conservative quantity**

$$
\frac{d(M_i\phi_i)}{dt},\qquad M_i=\rho_iV_i,
$$

not `M_current * d(phi)/dt` when mass changes. For a unit-density scalar, supply the cell volume as its mass.

With derivative weights `w[0 ... N]`:

$$
\frac{d(M\phi)}{dt}\approx
w_0M^{n+1}\phi^{n+1}+\sum_{j=1}^{N}w_jM^{n+1-j}\phi^{n+1-j}.
$$

Assembly adds

$$
A_{ii}\mathrel{+}=w_0M_i^{n+1},\qquad
b_i\mathrel{+}=-\sum_{j=1}^{N}w_jM_i^{n+1-j}\phi_i^{n+1-j}.
$$

The residual uses the shared convention `r = b - A*phi`, so this module adds **minus the integrated time derivative**. It does not multiply by volume or divide by the time step again: mass already includes volume, and the weights already include inverse time.

Current mass is frozen during assembly. No derivatives of a mass/density model with respect to the unknown are included.

## Inputs: mass, history and coefficients

### Mass providers

Every provider returns a finite, strictly positive **integrated cell mass** for each accessed owned cell.

| Provider | Meaning |
| --- | --- |
| `UniformMass<Real>{value}` | Same integrated mass for every cell; default `1`. This is not a uniform density unless volumes are also accounted for. |
| `CellMass{view}` | Reads an existing rank-one mass array; at least `owned` entries. `CellMass{cell_volumes}` also represents unit density. |
| `DensityVolumeMass{volumes, density}` | Evaluates `rho * V` without an intermediate mass array. Density is an arithmetic scalar or a rank-one view; all arrays cover at least `owned` cells. |
| Custom callable | Device-copyable `mass(cell)` returning the integrated coefficient. Custom storage extents/accessibility and parameter validation are caller contracts. |

`DensityVolumeMass` converts **both density and volume to the transient operator's `Real` before multiplying**, for current and historical masses. Float inputs can therefore be used with a double operator without first overflowing, underflowing or rounding their product in float. Each factor and the resulting mass must be finite and strictly positive in that computation type. A standalone `mass(cell)` call uses `ComputeReal`; `mass.template operator()<R>(cell)` explicitly selects another floating-point type. Construction and ordinary application call signatures are unchanged.

`UniformMass` and `CellMass` values are converted to the operator's `Real` when read. Custom `mass(cell)` providers keep their existing interface and control the precision of their internal calculations.

For an equation with a different positive storage coefficient, a custom provider can supply its cell integral; spatial/source units must agree. For example, define at namespace scope:

```cpp
template<class Volumes>
struct CapacityMass {
    Volumes volumes;
    ComputeReal capacity;
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell) const
    { return capacity * volumes(cell); }
};
```

### Accepted history

`TimeLevel{mass, field}` pairs an accepted scalar field with the mass from that same time. `field` can be a strided rank-one component. The following indices describe a step while it is being solved.

| Time | Meaning | How it reaches the operator |
| --- | --- | --- |
| `t_(n+1)` | New step being solved | Pass its mass separately. A residual evaluation also takes a candidate scalar at this time. |
| `t_n` | Last accepted step | First argument to `time_history()`. |
| `t_(n-1)` | Accepted step before `t_n` | Second argument to `time_history()` for BDF2. |

```cpp
const auto history = tt::time_history(
    tt::TimeLevel{mass_n, q_n},       // Accepted state at t_n.
    tt::TimeLevel{mass_nm1, q_nm1});  // Accepted state at t_(n-1).
```

With changing density, use independent density or mass snapshots, for example `DensityVolumeMass{volumes, rho_n}` and `DensityVolumeMass{volumes, rho_nm1}`. Binding both layers to an array later overwritten with the new density loses the old masses.

**`time_history()` makes no snapshots or array copies.** It stores the supplied descriptors by value. Keep accepted layers unchanged throughout assembly, the solve and any nonlinear/reconstruction outer iterations. Rotate them only when the full time step is accepted. All layers must use the same owned-cell ordering; changed numbering requires external history remapping.

### `numerics/time` coefficients

| Factory | History required | Weights |
| --- | --- | --- |
| `BackwardEuler{}.coefficients(dt)` | One layer: `n` | `[1, -1] / dt` |
| `Bdf2{}.coefficients(dt)` | Two layers: `n`, `n-1` | `[3/2, -2, 1/2] / dt`; equal adjacent steps only. |
| `Bdf2{}.coefficients(dt, previous_dt)` | Two layers: `n`, `n-1` | Variable-step BDF2; `previous_dt = t_n - t_(n-1)`. |

`weights[0]` multiplies `(M*phi)_(n+1)` at the time being solved. `weights[1]` multiplies the first accepted history entry `(M*phi)_n`, and so on. Here `dt = t_(n+1) - t_n`. Steps must be finite and positive, with representable coefficients and step ratio.

BDF2 has no automatic startup: use Backward Euler for the first step, then BDF2 once two accepted layers are available. Step selection, admissible step ratios, error control and acceptance are application decisions.

A custom `numerics::time::Derivative<C, N>` requires floating-point `C` and `N > 0`. It contains `N+1` dimensional weights and requires **exactly `N` history layers**, checked at compile time. The operator converts weights to its computation `Real` and requires all to be finite with `w[0] > 0`. It does not verify order, stability or `sum(weights) == 0`; custom coefficients must preserve a constant integrated quantity. All supplied history masses/values must be valid even when a particular history weight is zero.

## Assemble, solve and accept steps

The following example starts from `phi = 2` with unit density and source density `1`, so the equation is `d(V*phi)/dt = V`. It demonstrates the complete history lifecycle. It assumes active MPI/Kokkos/Tpetra and a prepared host mesh `host`, its device view `m`, matching `comm` and execution instance `exec`. Run host setup/submission under the project's `mpi::guarded` error boundary.

```cpp
#include <krona/transport/scalar/transient/operator.hpp>
#include <krona/transport/scalar/assembly.hpp>
#include <krona/transport/scalar/source/operator.hpp>
#include <krona/linalg/solver.hpp>

using namespace krona;
namespace ts = transport::scalar;
namespace tt = transport::scalar::transient;
namespace nt = numerics::time;
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Space = typename Exec::memory_space;

const auto owned = static_cast<std::size_t>(m.sizes.n_own);
tt::TransientOperator<Exec, Real> temporal(owned, exec);
ts::SourceOperator<Exec, Real> sources(m, exec);
ts::AssemblyStatus<Exec> status(exec);
const auto mass = tt::CellMass{m.geometry.cell_volume}; // rho = 1; static volumes.

Kokkos::View<StorageReal*, Space> q_new("candidate", owned);
Kokkos::View<StorageReal*, Space> q_n("accepted/n", owned);
Kokkos::View<StorageReal*, Space> q_nm1("accepted/n-1", owned);
Kokkos::deep_copy(exec, q_new, StorageReal{2});
Kokkos::deep_copy(exec, q_n, StorageReal{2});
// q_nm1 is not used at startup; it receives q_n when the first step is accepted.

linalg::FaceAssemblyPlan<Exec> assembly(host, comm, exec);
linalg::CsrMatrix<SolverReal, Exec> matrix(assembly.pattern());
linalg::Vector<SolverReal, Exec> rhs(matrix.distribution()), x(matrix.distribution());
linalg::Solver<decltype(matrix)> linear(matrix);
Real time = 0, previous_dt = 0;

for (int step = 0; step < 10; ++step) {
    const Real dt = Real{0.01}; // Replace with the chosen step size.
    const auto target = ts::make_system_view(matrix, rhs, assembly, exec);
    Kokkos::deep_copy(exec, target.values, SolverReal{0});
    Kokkos::deep_copy(exec, target.rhs, SolverReal{0});
    status.begin_async();

    const auto accepted_n = tt::TimeLevel{mass, q_n};
    if (step == 0)
        temporal.add_system_async(target, status, nt::BackwardEuler{}.coefficients(dt),
            mass, tt::time_history(accepted_n));
    else
        temporal.add_system_async(target, status, nt::Bdf2{}.coefficients(dt, previous_dt),
            mass, tt::time_history(accepted_n, tt::TimeLevel{mass, q_nm1}));
    sources.add_async(target.rhs, status, ts::UniformSource<Real>{1});
    // Add any spatial contributions for this equation here.
    (void)status.check();
    matrix.commit();
    rhs.commit();

    x.copy_from_component(q_new, exec);
    const auto result = linear.solve(rhs, x);
    if (!result.converged) break; // No field publication or history rotation.
    x.copy_to_component(q_new, exec);

    // Accept the completed step. Copy older history before overwriting q_n.
    Kokkos::deep_copy(exec, q_nm1, q_n);
    Kokkos::deep_copy(exec, q_n, q_new);
    exec.fence("accepted time layers ready");
    previous_dt = dt;
    time += dt;
}
```

The operator itself is independent of `FaceAssemblyPlan`: this example uses the standard shared graph so spatial contributions can be added to the same system. A purely temporal equation can use a diagonal graph or custom diagonal storage instead.

For changing mass, maintain/rotate its accepted snapshots alongside the scalar snapshots. Do not rotate history between outer iterations of the same step. If a step is rejected, retain the old layers and `previous_dt`; changing `dt` requires new weights and reassembly. Changing current mass also changes the temporal diagonal.

The example uses owned-only fields because this module reads no ghosts. Spatial reconstruction/residuals may require larger current-field storage, halo publication, gradients and boundary data. Those operations remain external. No boundary manager is needed by the temporal term itself.

## Matrix-free temporal residual

Given a candidate `q_new` and **unchanged accepted input layers for that step**, evaluate the time residual before history rotation:

```cpp
const Real dt = Real{0.01};
const auto derivative = nt::BackwardEuler{}.coefficients(dt);
const auto history = tt::time_history(tt::TimeLevel{mass, q_n});
Kokkos::View<Real*, Space> residual("time residual", owned);
Kokkos::deep_copy(exec, residual, Real{0});

status.begin_async();
temporal.add_residual_async(q_new, residual, status, derivative, mass, history);
sources.add_async(residual, status, ts::UniformSource<Real>{1});
(void)status.check();
```

Without the separate source call this adds only `-d(M*phi)/dt`. For Backward Euler:

$$
r_i^{\mathrm{time}}=\frac{M_i^n\phi_i^n-M_i^{n+1}\phi_i^{n+1}}{\Delta t}.
$$

Add diffusion, convection and source residuals into the same initialized output to assess the full equation. Use identical weights, mass snapshots and accepted history when comparing against `b - A*phi` from assembly. This path needs no matrix or algebra graph.

## Assemble a correction equation

For `phi^(n+1) = phi_iterate + delta_phi`, the time term adds `w[0]*M^(n+1)` to the correction matrix diagonal. Its RHS contribution is the time residual at `phi_iterate`. With an initialized system and an open assembly batch, use:

```cpp
temporal.add_system_async(target, status, derivative, mass, history, /*add_rhs=*/false);
temporal.add_residual_async(phi_iterate, target.rhs, status, derivative, mass, history);
```

With `add_rhs=false`, the matrix pass uses only the leading weight and the new-time mass. The history type still matches the derivative, and its data are read by the residual pass. Omitting the flag assembles both the diagonal and the history term for a solve in the field itself.

## Storage contracts and custom integration

`TransientOperator<Exec, Real>(owned, exec)` retains only the owned count and execution instance. `local_size()` returns that count; `execution_space_instance()` returns the instance. There is no plan or communicator attachment.

| Input/output | Contract |
| --- | --- |
| Current/historical scalar views | Accessible rank-one Kokkos views, each with at least `owned` entries; strided components work. Accessed values must remain finite after conversion to computation `Real`. |
| Mass arrays | Accessible rank-one views with at least `owned` entries; current and historical returned masses must remain finite and strictly positive in `Real`. |
| Residual output | Writable floating-point rank-one view with at least `owned` entries. Only the owned prefix is modified. |
| `target.values` | Writable floating-point rank-one native coefficient slice, including strided BSR component slices. |
| `target.rhs` | Writable floating-point rank-one view with **exactly** `owned` entries. |
| `target.entries.diagonal` | Accessible rank-one index view with **exactly** `owned` entries. Each entry must identify the true diagonal for that owned row, be in bounds and be distinct from other rows' entries. |

Only `entries.diagonal` is used; owner/neighbour entry maps are unnecessary. The kernel checks entry bounds but does not establish uniqueness, row identity or matching cell ordering. Different rows writing the same diagonal position would race.

Current/history inputs and mass-provider storage must not overlap written coefficients/RHS/residuals. In-place residual evaluation is unsupported; equal starting pointers are rejected, while partial/strided aliases are a caller contract. All arrays must remain stable until queued work completes.

The standard `make_system_view(matrix, rhs, assembly, exec, component)` supports CSR and a selected diagonal BSR equation/variable slice. For BSR it targets `(block, component, component)` and RHS DOFs `cell * block_size + component`. It requires the matrix's exact shared assembly graph and the RHS's exact matrix distribution. Other block components are untouched. Initialize the whole intended equation/system and assemble other components before solving a full block system.

For another backend, construct `SystemView{values, rhs, entries}` directly; the operator imposes no native matrix type. Assembly is additive and only changes diagonals/RHS, preserving off-diagonal coefficients. It does not clear targets or check solvability of the completed system. Positive temporal diagonals do not set `AssemblyReport::has_local_diffusion_anchor`; that flag describes diffusion boundary contributions only.

Coefficients are calculated in computation `Real`, then converted to the output scalar. The added temporal diagonal must remain positive/finite in coefficient storage. Nonfinite or unrepresentable contributions and accumulated targets are rejected through assembly diagnostics.

## Ownership, errors and asynchrony

| Operation | Behaviour |
| --- | --- |
| `coefficients()` | Local synchronous host calculation; no array allocation, kernel launch or MPI. |
| `TimeLevel`, `time_history()`, mass-provider construction | Small descriptor/value construction; no snapshots, array copies, launches or history rotation. |
| `TransientOperator` construction | Local setup requiring active Kokkos; stores no arrays and launches no work. |
| `AssemblyStatus::begin_async()` | Queues diagnostic flag reset; does not clear equation outputs. |
| `add_system_async()`, `add_residual_async()` | Queue owned-cell kernels on the operator's instance; no allocation, fence, halo exchange, MPI, commit or history update. |
| `AssemblyStatus::check()` | Copies the local flags to host, fences the execution instance and checks the flags. |
| Matrix/vector `commit()` | Completes pending native access and closes its batch; no MPI collective. |
| History copies with `deep_copy(exec, ...)` | May remain queued; order on the same instance or fence before other consumers. |

Use one `begin_async()` → contributions → `check()` batch on the same execution instance. Each rank checks its own flags. An error detected by any rank aborts the MPI job through `mpi::require`. All ranks, including empty ranks, call the collective solve in the same order. Host validation catches incompatible views or invalid coefficient conversion. Kernel flags report invalid masses, nonfinite time-layer values, invalid diagonal positions and nonfinite or unrepresentable additions. Flags persist across contributions, and some entries may already have been updated when an error is detected. Call `check()` explicitly to validate a batch. The destructor only waits for pending work.

`AssemblyStatus` owns a small device flag array and host mirror. The transient operator owns only two values: count and execution instance. History/mass providers retain their supplied members by value; managed views retain allocations, unmanaged views borrow. Neither form preserves old data against overwriting. `DensityVolumeMass` creates no mass array. Application code owns the field/history allocations and performs all snapshots, rotations and algebra transfers.

Keep producers and consumers ordered on the operator's execution instance; synchronize other execution instances/external MPI before use. Reuse the operator across steps while the owned count and ordering stay compatible. Any mesh/history remapping, time-step acceptance, nonlinear iterations and field publication belong to the application.
