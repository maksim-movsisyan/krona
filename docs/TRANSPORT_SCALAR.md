# Scalar transport

The current `krona::transport::scalar` module assembles and solves steady scalar isotropic diffusion:

$$
-\nabla\cdot(D\nabla\phi)=s,\qquad q_n=-D\,\partial_n\phi.
$$

There is one unknown per owned cell. Sources are densities per unit volume; boundary fluxes are outward flux densities per unit area. Assembly integrates both over the corresponding volumes and areas. Advection and time derivatives are not implemented in this module yet.

## Objects and ownership

| Object / header | Responsibility |
| --- | --- |
| `linalg::FaceAssemblyPlan<Exec>` / `linalg/face_assembly.hpp` | Shared cell-neighbour graph and face-to-native-entry map, usable by CSR and BSR matrices. |
| `DiffusionPlan<Exec, Real>` / `transport/scalar/plan.hpp` | Shared assembly plan plus borrowed mesh, cell-face connectivity and geometry views; validates diffusion geometry. |
| `DiffusionOperator<Scalar, Exec, Real>` / `transport/scalar/diffusion.hpp` | Owns one equation's scalar CSR coefficients, RHS and validation flags; assembles and checks the system. |
| `SteadyDiffusion<Storage, Scalar, Exec, Real, PrecScalar>` / `transport/scalar/steady_diffusion.hpp` | Owns a diffusion operator, solution vector and linear solver; borrows one writable field component. |
| `IsotropicDiffusion<Coefficient>` / `transport/isotropic_diffusion.hpp` | Converts between outward flux and normal gradient using a supplied coefficient provider. |
| `UniformSource`, `CellSource`, `DiffusionOptions`, `BalanceResult` / `transport/scalar/source.hpp` | Source providers, geometry checks and balance results. |

Plan copies share allocations. Different equations have independent matrix coefficients, RHS and solver workspaces, while sharing topology and geometry. `Storage`, `Scalar`, `Real` and `PrecScalar` default to `StorageReal`, `SolverReal`, `ComputeReal` and `PreconditionerReal`. Linear-solver precision conversion follows the contracts in `LINALG.md`.

Keep the field allocation, mesh/auxiliary buffer owners and borrowed MPI communicator alive for all users and submitted work. Host mesh/auxiliary containers must also remain alive when compute views borrow their buffers. Initialize MPI, Kokkos and Tpetra before constructing these objects, and destroy them before finalization.

## Mesh and geometry contracts

The host and compute mesh must describe the same indexed topology on the same communicator. Faces are grouped into owned/owned, owned/ghost and physical-boundary intervals. The operator requires:

- Attached face-cell connectivity, face geometry, cell centres for owned and MPI ghost cells, and positive owned-cell volumes.
- Complete `AuxConnType::CellFaces` incidence for owned cells.
- `AuxGeomType::FaceNormalDistance` and `AuxGeomType::FaceAreaOverNormalDistance` for all faces.
- `AuxGeomType::FaceInterpolationWeights` when interpolating cell coefficients across inner faces.

Normals are unit vectors pointing outward from each face's owner. Normal distances must be positive and consistent with coordinates; cached area/distance factors and any attached weights are checked against the geometry.

`DiffusionOptions` defaults to `require_orthogonal = true` and `orthogonality_tolerance = 1e-6`. The check compares the tangential part of the centre separation with its normal projection; on a physical boundary, the separation is owner-centre to face-centre.

The scalar boundary manager must borrow the **same compute mesh buffers** as the plan. Configure every labelled patch, including patches empty on a particular rank, and explicitly handle unlabelled faces when present. Extended fields contain owned cells, MPI ghosts, then physical-boundary face values; their required size is `n_cells + n_faces - n_inner_faces`.

Topology and geometry are static for the lifetime of a diffusion plan. Rebuild affected plans after changes to connectivity, numbering or geometry. `FaceAssemblyPlan` itself depends only on topology/numbering and can survive a geometry-only update.

## Diffusion coefficients and discretization

The same `IsotropicDiffusion` law is used for interior diffusion and scalar boundary conditions. A separate array of face coefficients is optional:

| Provider | Required data / behaviour |
| --- | --- |
| `UniformCoefficient<Real>{D}` | Constant coefficient; no coefficient array or interpolation weights. |
| `FaceCoefficient{view}` | Rank-one face-indexed View covering all local faces; no interpolation weights. |
| `InterpolatedCellCoefficient{view}` | Rank-one View covering owned and MPI ghost cells; harmonic interpolation by default. Refresh coefficient ghosts before assembly; inner faces require attached weights. Physical boundaries use the owner value. |
| `InterpolatedCellCoefficient<View, ArithmeticCoefficient>` | Same storage and weight requirements, with arithmetic interpolation. |
| `OwnerCellCoefficient{view}` | Exterior-face extrapolation for boundary laws. The diffusion operator rejects this provider when any inner faces exist. |
| Custom provider | Device-callable `coefficient(context)`; may evaluate a model or read existing fields. The caller supplies valid, accessible backing storage. |

Every assembled coefficient must be finite and strictly positive, including both cell values used by an interpolating provider. Zero diffusion is not supported. Providers are sampled during assembly; field-dependent coefficients do not introduce an automatic nonlinear iteration.

The default arguments of `assemble_async()` and `solve_owned()` mean `D = 1`, `s = 0` and `time = 0`. Explicit arguments are ordered as `(boundary, law, source, time)`.

For an interior face, let `d_P` and `d_N` be the projected owner-to-face and face-to-neighbour distances. The existing mesh weight is `w = d_N / (d_P + d_N)`:

$$
D_f^{\mathrm{arith}}=wD_P+(1-w)D_N,\qquad
D_f^{\mathrm{harm}}=\frac{1}{(1-w)/D_P+w/D_N}.
$$

Arithmetic interpolation estimates the face value. The harmonic formula represents the effective coefficient of two segments in series, with resistances `d_P/D_P` and `d_N/D_N`; this is why its distance factors appear in the opposite positions. Missing weights are not replaced by `0.5`.

With `d = x_N - x_P` and `d_n = d dot n_f`, the integrated outward owner flux is

$$
Q_P=-D_f\frac{A_f}{d_n}(\phi_N-\phi_P).
$$

Each owned row receives `+a` on its diagonal and `-a` for its neighbour, where `a = D_f A_f/d_n > 0`. Multiple connecting faces accumulate into the same graph entry. No interior atomic additions are needed because each kernel iteration owns one row; boundary-face contributions sharing an owner use atomic additions.

**There is no nonorthogonal correction.** The stored `face_non_ortho_*` vectors are not used, and boundary relations are evaluated with zero reconstruction increment. Setting `require_orthogonal = false` only disables the geometry check; it does not add the missing gradient contribution. Distances and area/distance factors currently come from attached arrays rather than being recomputed on demand.

Built-in boundary conditions are `FixedValue`, `NormalGradient` and `FixedFlux`; custom registered conditions return the scalar module's `Relation`. `FixedFlux` uses the supplied diffusion law, with the outward sign convention above. Assembly accepts finite affine outward-flux relations with a nonnegative owner coefficient.

## Shared setup and steady solve

The example assumes an orthogonal mesh with exactly two labelled patches, IDs `0` and `1`, and no unlabelled boundary faces. `host_mesh`, compute storage `mesh`, auxiliary storage `connectivity` and `geometry`, `comm`, and `exec` are already prepared and remain alive. Select mesh halo data as well if publishing ghost values as below.

```cpp
#include <krona/transport/scalar/steady_diffusion.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>

using namespace krona;
namespace ts = transport::scalar;
namespace bc = boundary::scalar;
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;

// Build this once; other scalar CSR or coupled BSR systems can share its graph/map.
linalg::FaceAssemblyPlan<Exec> shared(host_mesh, comm, exec);
ts::DiffusionPlan<Exec, Real> plan(
    host_mesh, mesh.view(), connectivity.view(), geometry.view(), shared, exec);

bc::Manager<Exec, Real> boundary(host_mesh, mesh.view(), exec);
boundary.set(0, bc::FixedValue<Real>{0});
boundary.set(1, bc::FixedValue<Real>{1});

fields::FieldStorage<StorageReal, 1, Exec> phi(
    "phi", boundary.required_field_size(), fields::FieldLocation::Custom, exec);
phi.fill(exec, StorageReal{0}); // Also supplies the initial owned-cell guess.

ts::SteadyDiffusionOptions options;
options.linear.method = "GMRES";
options.linear.preconditioner.type = "AMG"; // These are already the defaults.
ts::SteadyDiffusion<> solver(plan, phi.view().component(0), options);

halo::Exchange<StorageReal, Exec> exchange(
    mesh.view(), mesh.halo_plan(), comm, 17, exec, halo::Transfer::HostStaged);
exchange.add_field(phi.view()); // Register other scalars here before prepare().
exchange.prepare();           // Reuse this exchange for subsequent solves.

const auto law = transport::IsotropicDiffusion{
    transport::UniformCoefficient<Real>{2}};
const ts::UniformSource<Real> source{3}; // Density: RHS receives 3 * cell_volume.

const auto result = solver.solve_owned(boundary, law, source);
if (result.converged) {
    exchange.exchange(); // Blocking: MPI ghosts are now current.
    boundary.apply_async(phi.view(), law); // Queues boundary-face values on exec.
    exec.fence("Scalar field ready for host access or another execution instance");
}
```

All ranks participate in plan construction and `solve_owned()`, including empty ranks. `solve_owned()` uses the current owned field as the initial guess. On convergence, it completes the copy back to owned cells before returning. On nonconvergence, the entire application field stays unchanged; `solver.solution()` retains the latest algebraic iterate.

Ghost and boundary slots are never updated by `solve_owned()`. Publish them explicitly when a downstream operation needs them. Tpetra handles its own remote imports during the linear solve; it does not refresh KRONA's application-field ghosts. Several scalar solvers can share `plan` and publish through one grouped exchange after their owned solutions are ready.

The borrowed component may be a strided rank-one subview of a multi-component field. It must be writable, have the selected `Storage` value type, be accessible from `Exec`, and cover the full extended size.

## Cell-dependent sources and coefficients

A source has the device-callable interface `source(std::size_t cell)` and returns a **density**, not a volume-integrated value. Negative finite densities represent sinks. Only owned source entries are read, so source ghosts are unnecessary. For example, replace the uniform source with:

```cpp
using Space = typename Exec::memory_space;
Kokkos::View<Real*, Space> density("source", host_mesh.n_own);
const auto x = mesh.view().geometry.cell_centroid_x;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<LocalIndex>>;
Kokkos::parallel_for("prepare_source", Policy(exec, 0, host_mesh.n_own),
    KOKKOS_LAMBDA(LocalIndex cell) {
        density(cell) = Real{3} * (Real{1} + static_cast<Real>(x(cell)));
    });
const ts::CellSource source{density};
const auto result = solver.solve_owned(boundary, law, source);
```

Both kernels use the same `exec`, so preparation is ordered before assembly without an intermediate fence. A custom source functor can compute the density directly instead of allocating `density`. Sources receive only the cell index; carry coordinates, model parameters or time explicitly in the functor. The assembly `time` argument is supplied to face contexts, not as a separate source argument.

For cell-dependent diffusion, attach `FaceInterpolationWeights`, prepare a rank-one coefficient View covering `n_cells`, and refresh its MPI ghosts before assembly:

```cpp
// D is an existing cell coefficient View with current owned and ghost values.
const auto law = transport::IsotropicDiffusion{
    transport::InterpolatedCellCoefficient{D}}; // Harmonic by default.

// Alternative numerical interpolation policy:
using CellD = std::remove_cvref_t<decltype(D)>;
const auto arithmetic_law = transport::IsotropicDiffusion{
    transport::InterpolatedCellCoefficient<CellD, transport::ArithmeticCoefficient>{D}};
```

Custom coefficient providers must support both interior and boundary contexts. Common members include `face`, `owner`, face position, unit normal, `distance`, `area` and `time`. Interior contexts additionally expose `neighbour` and `owner_weight`; the latter is meaningful only when interpolation weights are attached. Providers must produce consistent physical coefficients on both sides of an MPI interface.

## Assembly, asynchronous work and ordering

`assemble_async(boundary, law, source, time)` starts a **full overwrite** of the operator's own system: it clears coefficients, RHS and flags, queues owned-row assembly, adds boundary contributions, then queues numerical validation. All this work uses the plan's stored execution-space instance. Boundary assembly uses host patch dispatch and runs on that instance even if the boundary manager stores another instance.

| Operation | Completion contract |
| --- | --- |
| `FaceAssemblyPlan` construction | Collective, blocking topology setup and upload. |
| `DiffusionPlan` construction | Blocking geometry validation; the communicator overload also constructs a new assembly plan collectively. |
| `assemble_async()` | Local kernel submission; no halo exchange or MPI status reduction. Native Tpetra view access may synchronize its own storage. |
| `DiffusionOperator::commit()` | Completes assembly, reads device validation flags, performs a collective MPI reduction, and marks the system ready. |
| `solve_owned()` | Collective and blocking: assembly, commits, linear solve, and successful owned-field publication. |
| `balance()` | Blocking local reductions followed by a collective MPI sum. |
| Boundary `apply_async()` | Queues boundary-slot updates on the boundary manager's instance. |

Keep coefficient/source backing buffers alive and unchanged until `commit()` completes, or until `solve_owned()` returns. Data produced on the same execution instance is consumed in submission order. For data produced on a different instance, complete that producer or establish an explicit dependency before assembly. Use the same ordering rule when consuming asynchronously filled boundary values.

The low-level path allows application work between submission and commit:

```cpp
ts::DiffusionOperator<> op(plan);
linalg::Vector<SolverReal, Exec> solution(op.matrix().distribution());
linalg::Solver<typename decltype(op)::matrix_type> linear(op.matrix());

op.assemble_async(boundary, law, source);
solution.copy_from_component(phi.view().component(0), exec);
// Independent work may be submitted here.
op.commit();       // Required collectively; also checks the assembled system.
solution.commit();
const auto result = linear.solve(op.rhs(), solution);
if (result.converged) {
    solution.copy_to_component(phi.view().component(0), exec);
    solution.commit(); // Completes the copy, including on ranks without neighbours.
}
```

Call `commit()` before another assembly or native solve. An execution-space fence alone does not perform collective validation or clear the operator/wrapper state. Destruction drains pending local work but does not replace the explicit validating commit. Repeated assembly starts from zero; it does not retain a previously assembled convection or diffusion contribution.

## Balance and current limitations

After a successful solve, an optional collective diagnostic is:

```cpp
const auto balance = solver.balance(boundary, law, source);
// source_integral, outward_boundary_flux, defect = flux - source.
```

It evaluates the supplied source and boundary laws using the application field's owned values; it does not read stored boundary slots or reuse cached assembly integrals. Supply the same law/source/time and coefficient/source data used for assembly. The diagnostic is valid for an already committed operator and introduces synchronization; it is not part of the asynchronous assembly path.

Validation checks geometry/incidence during plan construction, known provider shapes on the host, and finite numerical data, positive diffusion, acceptable boundary flux relations and a positive finite diagonal during assembly/commit. Opaque custom functors remain responsible for their own backing-storage shapes and accesses. Validation does not prove discretization accuracy or general matrix solvability.

For a nonempty global system, commit requires at least one value/Robin anchor. Pure Neumann problems are rejected: there is no reference-cell, mean-value constraint or nullspace treatment. The global anchor check does not detect separately unanchored disconnected components. Empty ranks and globally empty systems are supported.

The current operator assembles scalar CSR systems only. Its shared `FaceAssemblyPlan` can also serve BSR matrices, but that does not make this diffusion assembler a block-equation assembler. Each operator allocates independent coefficients and RHS. Advection, transient terms, nonorthogonal correction, anisotropic diffusion and automatic nonlinear iteration are outside the current implementation.

## Future extensions

- **Separate contribution assembly from system ownership.** Move matrix/RHS ownership to the equation/system or steady-solve layer. A diffusion assembler could accept external buffers on a compatible shared graph and accumulate its contribution alongside convection or time derivatives. This needs an explicit additive mode: clear the complete system once, add source/diagonal/off-diagonal contributions, then commit after all producers. Solvability checks such as the pure-Neumann restriction belong to the complete system; an individual diffusion contribution may be singular. No external-matrix constructor or additive assembly method exists yet.
- **Nonorthogonal correction.** Use a reconstructed face gradient with the existing nonorthogonality vectors, and define the correction's sign, boundary treatment and iteration/update contract explicitly.
- **Optional computed geometry.** Select cached metrics or compute distances, area/distance factors and actual interpolation weights inside kernels. This could reduce GPU memory usage; measure the additional coordinate traffic and arithmetic. A missing geometric weight must not silently become `0.5`.
- **Broader equations and mesh lifecycle.** Add advection/transient terms, coupled block contributions and explicit nullspace/reference handling. Mesh movement/adaptation will need geometry refresh and topology/numbering invalidation rules for the shared plans.
