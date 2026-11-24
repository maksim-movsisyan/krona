# Scalar finite-volume solver

`krona::solvers::scalar::Solver` assembles and solves one scalar equation using
the convection, diffusion, source and transient operators in `transport/scalar`.
Include `<krona/solvers/scalar/solver.hpp>` and link `krona::solver_scalar` in a
build with Trilinos enabled.

## Ownership and inputs

The solver owns the CSR matrix, RHS, linear solver, halo buffers, gradient
and limiter workspace, and accepted time levels. It updates a borrowed,
writable scalar `FieldView`. The caller can allocate this field with
`FieldsManager` or a standalone `FieldStorage`. Keep the field allocation,
mesh data and partition communicator alive for the life of the solver.
Initialize the owned values before the first solve.

The field contains owned cells and MPI ghosts. Append one slot per physical
boundary face if the selected reconstruction or nonorthogonal diffusion
correction needs gradients. The solver exchanges the relevant ghosts and
fills physical boundary samples when computing gradients. After a solve, the
owned values are current. The next assembly that needs ghosts or boundary
samples refreshes them.

Construct the solver with the host partition, matching device mesh and
cell-to-face incidence, auxiliary geometry and halo plan. Supply the
auxiliary geometry needed by the selected gradient and diffusion options.
The finalized scalar boundary manager must use the same mesh and execution
instance. Use matching halo tags across neighbour ranks and distinct tags
for exchange groups that can be active simultaneously.

## Providers and time integration

The caller supplies these policies. They run on the selected execution space.

| Input | Contract |
| --- | --- |
| Face flux | Returns the signed, area-integrated mass flow rate at the face. Use a prescribed velocity policy or `FaceMassFlux` for a precomputed face field. At MPI interfaces the copies must represent the same physical flux in their respective owner orientations. |
| Diffusion law | An `IsotropicDiffusion` law with a callable face coefficient provider. |
| Source | Returns source density per unit cell volume for each owned cell. The default is zero. |
| Mass | For transient runs, returns integrated cell mass \(\rho V\) for each owned cell at the requested time level. |

Supply a flux and diffusion policy to the solve methods. Options::convection
and Options::diffusion determine which terms enter the equation.

For a steady solve, select `TimeScheme::Steady` and call `solve_steady`.
For a transient solve, initialize the history with the initial field and its
mass at time zero. `next_time()` exposes the next evaluation time so that
the caller can bind time-dependent providers before calling `advance`.
The implicit time step is fixed. BDF2 starts with backward Euler and then
uses two accepted levels. Each accepted layer stores its own mass values,
so replacing the current density buffer after a step preserves the history.

```cpp
namespace ss = krona::solvers::scalar;
namespace ts = krona::transport::scalar;

ss::Options options;
options.time_scheme = ts::TimeScheme::Bdf2;
options.time_step = 0.01;

ss::Solver solver(host, mesh, connectivity, geometry, halo_plan,
                  phi, options, comm, exec);
solver.initialize_history(initial_mass);

for (int step = 0; step < steps; ++step) {
    auto source = make_source(solver.next_time());
    const auto result = solver.advance(boundary, flux, diffusion, mass, source);
    // Owned values in phi are ready. result.time is the accepted time.
}
```

The example assumes `phi` is a writable one-component `FieldView` and the
caller supplies the providers and the prepared mesh and boundary objects.
`solve_steady` and `advance` are collective and blocking. Call them in the
same order on all ranks. Use the same execution instance for dependent work
or finish external producers before the solver reads their inputs.

The base matrix uses upwind convection and two-point diffusion. Selected
higher-order convection and nonorthogonal diffusion contributions enter as
explicit RHS corrections in the outer iterations. The result reports the
linear iteration count for each outer update.
