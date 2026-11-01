# Scalar transient term

`krona::transport::scalar::transient` adds an implicit time derivative to a scalar equation or evaluates its residual. It works on owned cells and takes the time coefficients, current mass and accepted history from the application. It owns no mesh, matrix, field or history storage and does not advance time or solve a system itself.

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

`TimeLevel{mass, field}` describes one accepted layer. `field` is a rank-one scalar view, including a strided field component. `time_history()` combines these descriptors **newest first**:

```cpp
const auto history = tt::time_history(
    tt::TimeLevel{mass_n, q_n},       // Accepted state at t_n.
    tt::TimeLevel{mass_nm1, q_nm1});  // Accepted state at t_(n-1).
```

Current mass and the candidate field are passed separately. Each historical mass must belong to its own historical field. With changing density, use independent density/mass snapshots, for example `DensityVolumeMass{volumes, rho_n}` and `DensityVolumeMass{volumes, rho_nm1}`. Binding both layers to an array later overwritten with current density loses the old masses.

**`time_history()` makes no snapshots or array copies.** It stores the supplied descriptors by value. Keep accepted layers unchanged throughout assembly, the solve and any nonlinear/reconstruction outer iterations. Rotate them only when the full time step is accepted. All layers must use the same owned-cell ordering; changed numbering requires external history remapping.

### `numerics/time` coefficients

| Factory | History required | Weights |
| --- | --- | --- |
| `BackwardEuler{}.coefficients(dt)` | One layer: `n` | `[1, -1] / dt` |
| `Bdf2{}.coefficients(dt)` | Two layers: `n`, `n-1` | `[3/2, -2, 1/2] / dt`; equal adjacent steps only. |
| `Bdf2{}.coefficients(dt, previous_dt)` | Two layers: `n`, `n-1` | Variable-step BDF2; `previous_dt = t_n - t_(n-1)`. |

`weights[0]` multiplies the proposed current state at `t_(n+1)`, `weights[1]` the newest accepted state at `t_n`, and so on. Here `dt = t_(n+1) - t_n`. Steps must be finite and positive, with representable coefficients/step ratio.

BDF2 has no automatic startup: use Backward Euler for the first step, then BDF2 once two accepted layers are available. Step selection, admissible step ratios, error control and acceptance are application decisions.

A custom `numerics::time::Derivative<C, N>` requires floating-point `C` and `N > 0`. It contains `N+1` dimensional weights and requires **exactly `N` history layers**, checked at compile time. The operator converts weights to its computation `Real` and requires all to be finite with `w[0] > 0`. It does not verify order, stability or `sum(weights) == 0`; custom coefficients must preserve a constant integrated quantity. All supplied history masses/values must be valid even when a particular history weight is zero.

## Assemble, solve and accept steps

The following example starts from `phi = 2` with unit density and source density `1`, so the equation is `d(V*phi)/dt = V`. It demonstrates the complete history lifecycle. It assumes active MPI/Kokkos/Tpetra and a prepared host mesh `host`, its device view `m`, matching `comm` and execution instance `exec`. Run host setup/submission under the project's `mpi::guarded` error boundary.

```cpp
#include <krona/transport/scalar/transient/operator.hpp>
#include <krona/transport/scalar/assembly.hpp>
#include <krona/transport/scalar/source.hpp>
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

    const auto newest = tt::TimeLevel{mass, q_n};
    if (step == 0)
        temporal.add_system_async(target, status, nt::BackwardEuler{}.coefficients(dt),
            mass, tt::time_history(newest));
    else
        temporal.add_system_async(target, status, nt::Bdf2{}.coefficients(dt, previous_dt),
            mass, tt::time_history(newest, tt::TimeLevel{mass, q_nm1}));
    sources.add_async(target.rhs, status, ts::UniformSource<Real>{1});
    // Add any spatial contributions for this equation here.
    (void)status.check(comm);
    matrix.commit();
    rhs.commit();

    x.copy_from_component(q_new, exec);
    x.commit();
    const auto result = linear.solve(rhs, x);
    if (!result.converged) break; // No field publication or history rotation.
    x.copy_to_component(q_new, exec);
    x.commit();

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
(void)status.check(comm);
```

Without the separate source call this adds only `-d(M*phi)/dt`. For Backward Euler:

$$
r_i^{\mathrm{time}}=\frac{M_i^n\phi_i^n-M_i^{n+1}\phi_i^{n+1}}{\Delta t}.
$$

Add diffusion, convection and source residuals into the same initialized output to assess the full equation. Use identical weights, mass snapshots and accepted history when comparing against `b - A*phi` from assembly. This path needs no matrix or algebra graph.

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

For another backend, construct `SystemView{values, rhs, entries}` directly; the operator imposes no native matrix type. Assembly is additive and only changes diagonals/RHS, preserving off-diagonal coefficients. It does not clear targets or check solvability of the completed system. Positive temporal diagonals do not set `AssemblyReport::has_diffusion_anchor`; that flag describes diffusion boundary contributions only.

Coefficients are calculated in computation `Real`, then converted to the output scalar. The added temporal diagonal must remain positive/finite in coefficient storage. Nonfinite or unrepresentable contributions and accumulated targets are rejected through assembly diagnostics.

## Ownership, errors and asynchrony

| Operation | Behaviour |
| --- | --- |
| `coefficients()` | Local synchronous host calculation; no array allocation, kernel launch or MPI. |
| `TimeLevel`, `time_history()`, mass-provider construction | Small descriptor/value construction; no snapshots, array copies, launches or history rotation. |
| `TransientOperator` construction | Local setup requiring active Kokkos; stores no arrays and launches no work. |
| `AssemblyStatus::begin_async()` | Queues diagnostic flag reset; does not clear equation outputs. |
| `add_system_async()`, `add_residual_async()` | Queue owned-cell kernels on the operator's instance; no allocation, fence, halo exchange, MPI, commit or history update. |
| `AssemblyStatus::check(comm)` | Copies flags to host, fences its instance and performs an MPI all-reduce. Does not commit algebra wrappers. |
| Matrix/vector `commit()` | Completes pending native access and closes its batch; no MPI collective. |
| History copies with `deep_copy(exec, ...)` | May remain queued; order on the same instance or fence before other consumers. |

Use one `begin_async()` → contributions → `check(comm)` batch on the same execution instance. All ranks, including empty ranks, call collective checks/solves in the same order. A valid later contribution does not erase earlier errors. Host validation rejects incompatible views/coefficient conversion; kernel flags report invalid masses, nonfinite time-layer values, invalid diagonal positions and nonfinite/unrepresentable additions. A failed batch is not rolled back. A pending status destructor fences its instance but does not perform the collective error check.

`AssemblyStatus` owns a small device flag array and host mirror. The transient operator owns only two values: count and execution instance. History/mass providers retain their supplied members by value; managed views retain allocations, unmanaged views borrow. Neither form preserves old data against overwriting. `DensityVolumeMass` creates no mass array. Application code owns the field/history allocations and performs all snapshots, rotations and algebra transfers.

Keep producers and consumers ordered on the operator's execution instance; synchronize other execution instances/external MPI before use. Reuse the operator across steps while the owned count and ordering stay compatible. Any mesh/history remapping, time-step acceptance, nonlinear iterations and field publication belong to the application.
