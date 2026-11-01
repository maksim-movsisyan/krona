# Scalar convection-diffusion: assembly pipeline

This outline combines the scalar operators into a solver for

$$
\frac{\partial(\rho\phi)}{\partial t}
+\nabla\cdot(\rho\mathbf u\phi)
-\nabla\cdot(\Gamma\nabla\phi)=s.
$$

The main example is steady: omit the time derivative. It uses limited linear convection, an explicit nonorthogonal diffusion correction and one shared CSR matrix. Mesh loading, boundary configuration and application convergence helpers are omitted.

## 1. Prepare the shared inputs

Use one prepared mesh and local numbering for every plan. Assume active MPI/Kokkos/Tpetra, host mesh `host`, device storage `device_mesh`, its view `m`, auxiliary storages `conn`/`geom`, communicator `comm` and execution instance `exec`.

Required data:

- Mesh: `MeshData::FiniteVolume`, including halo maps.
- Connectivity: `CellFaces`.
- Auxiliary geometry: `FaceNormalDistance`, `FaceAreaOverNormalDistance`, `FaceInterpolationWeights`, `FaceNonOrthogonalityVector`.
- `boundary`: a fully assigned scalar manager on the same mesh.
- `mass_flux`: ready rank-one signed face mass fluxes, already including density and area, with consistent interface signs across ranks.
- `inflow`: `FixedValueInflow{}` when all possible inflows have `FixedValue`, or a custom policy/profile registry otherwise. Diffusive gradient/flux data alone do not supply convective inflow values.

The snippets use these aliases:

```cpp
using namespace krona;
namespace ts = transport::scalar;
namespace cv = transport::scalar::convection;
namespace df = transport::scalar::diffusion;
namespace rec = numerics::reconstruction;
namespace grad = numerics::gradient;
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Space = typename Exec::memory_space;
```

Include the operator headers for scalar convection/diffusion/source, `scalar/assembly.hpp`, `linalg/solver.hpp`, `fields/field_storage.hpp`, `halo/exchange.hpp` and `numerics/gradient/gradient_operator.hpp`. Run host setup/submission under the project's `mpi::guarded` error boundary.

## 2. Create fields, operators and reusable exchanges

```cpp
const auto owned = static_cast<std::size_t>(m.sizes.n_own);
const auto cells = static_cast<std::size_t>(m.sizes.n_cells);
fields::FieldStorage<StorageReal, 1, Exec> phi(
    "phi", boundary.required_field_size(), fields::FieldLocation::Custom, exec);
fields::FieldStorage<StorageReal, 3, Exec> g(
    "gradient/phi", cells, fields::FieldLocation::Custom, exec);
fields::FieldStorage<StorageReal, 1, Exec> theta(
    "limiter/phi", cells, fields::FieldLocation::Custom, exec);
phi.fill(exec, StorageReal{0}); // Replace with the chosen finite initial guess.
g.fill(exec, StorageReal{0});   // Initial boundary correction uses zero gradients.
theta.fill(exec, StorageReal{1});
const auto q = phi.view().component(0);

cv::ConvectionPlan<Exec, Real> cplan(host, m, conn.view(), comm, exec);
df::DiffusionPlan<Exec, Real> dplan(host, m, conn.view(), geom.view(),
    comm, exec, df::DiffusionOptions{false, 1e-6}); // Permit nonorthogonal geometry.
cv::ConvectionOperator<Exec, Real> convection(cplan);
df::DiffusionOperator<Exec, Real> diffusion(dplan);
ts::SourceOperator<Exec, Real> sources(m, exec);
ts::AssemblyStatus<Exec> status(exec);
grad::GradientOperator<Exec> gradient(m, conn.view(), geom.view(), {}, exec);
rec::MeshStencil<Space, Real> stencil(m, conn.view());

const auto flux = cv::FaceMassFlux{mass_flux};
const auto law = transport::IsotropicDiffusion{transport::UniformCoefficient<Real>{2}};
const ts::UniformSource<Real> source{0}; // Replace with the volume-source density.
const Real time = 0; // Evaluation time, fixed throughout these outer iterations.

const auto gradients = rec::gradient_pack(g.view());
const auto unlimited = rec::data(rec::field_pack(phi.view()), gradients);
const auto limited = rec::data(rec::field_pack(phi.view()), gradients,
    rec::field_pack(theta.view()));
const auto reconstruction = cv::make_reconstruction(cplan, rec::LimitedLinear{}, limited);
const auto correction = df::NonOrthogonalCorrection{df::CellGradients{
    g.view().component(0), g.view().component(1), g.view().component(2)}};

halo::Exchange<StorageReal, Exec> phi_halo(m, device_mesh.halo_plan(), comm, 71, exec);
phi_halo.add_field(phi.view());
phi_halo.prepare();
halo::Exchange<StorageReal, Exec> reconstruction_halo(
    m, device_mesh.halo_plan(), comm, 72, exec);
reconstruction_halo.add_field(g.view());
reconstruction_halo.add_field(theta.view());
reconstruction_halo.prepare();
```

`phi` contains real cells and boundary-face samples. Gradients and factors contain real cells, including ghosts. `LimitedLinear` limits only the convective increment; diffusion uses the original gradients. The default gradient method is face-neighbour least squares and requires a valid nondegenerate 3-D stencil.

## 3. Assemble the base system once

```cpp
linalg::FaceAssemblyPlan<Exec> assembly(host, comm, exec);
linalg::CsrMatrix<SolverReal, Exec> matrix(assembly.pattern());
linalg::Vector<SolverReal, Exec> rhs(matrix.distribution()), x(matrix.distribution());
const auto target = ts::make_system_view(matrix, rhs, assembly, exec);
Kokkos::deep_copy(exec, target.values, SolverReal{0});
Kokkos::deep_copy(exec, target.rhs, SolverReal{0});

status.begin_async();
convection.add_system_async(target, boundary, status, flux,
    cv::NoReconstruction{}, inflow, time);
diffusion.add_system_async(target, boundary, status, law, df::NoCorrection{}, time);
sources.add_async(target.rhs, status, source);
(void)status.check(comm);
matrix.commit();
rhs.commit();

Kokkos::View<SolverReal*, Space> base_rhs("transport/base_rhs", owned);
Kokkos::deep_copy(exec, base_rhs, Kokkos::subview(rhs.read(exec), Kokkos::ALL(), 0));
rhs.commit();
linalg::SolverOptions options;
options.method = "GMRES"; // Convection generally makes the system nonsymmetric.
options.preconditioner.type = "RELAXATION";
linalg::Solver<decltype(matrix)> linear(matrix, options);
Kokkos::View<Real*, Space> residual("transport/residual", owned);
```

The matrix contains upwind convection plus two-point diffusion. **`base_rhs` contains neither convective reconstruction nor nonorthogonal diffusion correction**, even from a first iteration. It already includes prescribed inflow, zero-correction diffusive boundary terms and the source.

Boundary conditions enter this assembly directly through `boundary`/`inflow`. Filling boundary slots is a separate operation needed below for gradient/limiter sampling. The steady problem still needs appropriate physical constraints; `status.check()` is not a solvability test.

## 4. Prepare reconstruction data for the current iterate

```cpp
auto prepare_iterate = [&] {
    phi_halo.exchange();
    // Use initialized/previous gradients for the first boundary sampling pass.
    boundary.apply_async(phi.view(), law, correction, time);
    gradient.apply(phi.view(), g.view()); // Writes owned gradients only.
    // Refresh gradient-dependent boundary samples before limiter preparation.
    boundary.apply_async(phi.view(), law, correction, time);
    rec::compute_limiters_async<Real>(exec, stencil, unlimited,
        rec::field_pack(theta.view()), rec::BarthJespersen{}, rec::SecondOrderUpwind{});
    reconstruction_halo.exchange(); // Publish both gradients and factors.
};
```

All operations use `exec`, so boundary filling precedes gradient reads and gradient writes precede limiter reads. The two boundary passes serve different input states: previous gradients first, newly computed gradients second. For fixed-value conditions the face value itself is unchanged. Gradient/flux/Robin boundary sampling can depend on the gradient correction and participates in the outer iteration.

Use the unlimited version of the consumed scheme when preparing factors: here `SecondOrderUpwind` feeds `LimitedLinear`. Changing the convective scheme may change factor/stencil requirements; see `NUMERICS-RECONSTRUCTION.md`.

## 5. Update RHS, solve and check the full equation

```cpp
prepare_iterate();
bool outer_converged = false;
for (int outer = 0; outer < max_outer_iterations; ++outer) {
    const auto b = Kokkos::subview(rhs.overwrite(exec), Kokkos::ALL(), 0);
    Kokkos::deep_copy(exec, b, base_rhs); // Discard both previous corrections.
    status.begin_async();
    convection.add_correction_async(b, boundary, status, flux, reconstruction, time);
    diffusion.add_correction_async(b, boundary, status, law, correction, time);
    (void)status.check(comm);
    rhs.commit();

    x.copy_from_component(q, exec);
    x.commit();
    const auto solved = linear.solve(rhs, x);
    mpi::require(solved.converged, "Scalar transport: linear solve failed");
    x.copy_to_component(q, exec); // Owned cells only.
    x.commit();

    prepare_iterate(); // Refresh ghosts, boundary samples, gradients and factors.
    Kokkos::deep_copy(exec, residual, Real{0});
    status.begin_async();
    convection.add_residual_async(q, residual, boundary, status, flux,
        reconstruction, inflow, time);
    diffusion.add_residual_async(q, residual, boundary, status, law, correction, time);
    sources.add_async(residual, status, source);
    (void)status.check(comm);

    // Application helper: global owned-cell norm, including an MPI reduction.
    if (global_owned_l2(residual, comm) <= outer_tolerance) {
        outer_converged = true;
        break;
    }
}
mpi::require(outer_converged, "Scalar transport: outer iteration did not converge");
```

`max_outer_iterations`, `outer_tolerance` and `global_owned_l2()` are application choices/helpers, not KRONA APIs. Every rank must make the same convergence decision. The residual is integrated and equals **source minus outward convective/diffusive fluxes**; do not divide by volume unless your convergence definition explicitly requires it.

A converged linear solve only satisfies the system with frozen corrections. The full residual above uses the updated field and freshly prepared reconstruction data. Applications may additionally monitor field/gradient changes or apply relaxation to owned updates before `prepare_iterate()`.

For fixed mass flux, diffusion coefficient and boundary matrix slopes, only the RHS changes, so the matrix/preconditioner can be reused. Rebuild the base system when those coefficients change; refresh its RHS when inflow/source/base boundary data change. Custom diffusion boundary slopes must be independent of the correction increment. No correction may accumulate on an already corrected RHS.

## Optional: add time dependence

At each physical step, prepare derivative weights, current integrated mass `rho_new * V` and accepted history, newest first. Add the temporal term to the **base assembly** before its `check()`:

```cpp
temporal.add_system_async(target, status, derivative, current_mass, history);
```

Add its residual inside the same full-residual batch:

```cpp
temporal.add_residual_async(q, residual, status, derivative, current_mass, history);
```

Here `temporal` is `ts::transient::TransientOperator<Exec, Real>(owned, exec)` from `<krona/transport/scalar/transient/operator.hpp>`. Use Backward Euler at startup and BDF2 once two accepted layers exist; see `TRANSPORT-SCALAR-TRANSIENT.md`. Evaluate spatial boundary/provider data at the chosen step time. Keep derivative, historical masses/fields and current frozen coefficients fixed throughout the outer solve. Update history/time only after the step is accepted. Refresh the base RHS for new history, and the matrix when temporal diagonal or spatial coefficients change.

All shown contributions are additive. Exchanges use the blocking convenience path; gradient/boundary/limiter/assembly kernels enqueue work. `check()` fences and performs collective diagnostics, while native `commit()` closes algebra access. Keep storages, providers, managers, plans and communicator alive for their users. Complete the outer solve on all ranks before consuming or exporting the final field.
