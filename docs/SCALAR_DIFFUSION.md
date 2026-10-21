# Scalar diffusion

KRONA provides a cell-centred finite-volume discretization of isotropic scalar diffusion:

$$
-\nabla\cdot(D\nabla\phi)=s,\qquad q_n=-D\,\partial_n\phi.
$$

The spatial operator can add its contribution to an existing algebraic system or evaluate its residual directly from a field. The convenience `SteadyDiffusion` solver builds and solves a complete steady diffusion equation. Convection and time derivatives are not implemented here yet.

The shared convention is **cell-integrated equations and residuals**:

$$
A\phi=b,\qquad r=b-A\phi,\qquad b_i^{\rm source}=s_iV_i.
$$

Sources are densities per unit volume; prescribed outward fluxes are densities per unit area. Neither RHS entries nor residuals are divided by cell volume. A positive source increases both `b` and `r`; an outward diffusive flux decreases `r`.

## Objects, namespaces and ownership

The examples use these aliases:

```cpp
namespace ts = krona::transport::scalar;
namespace td = krona::transport::scalar::diffusion;
namespace bc = krona::boundary::scalar;
```

Header paths below are relative to `include/krona/`.

| Object / header | Responsibility |
| --- | --- |
| `linalg::FaceAssemblyPlan<Exec>` / `linalg/face_assembly.hpp` | Shared cell-neighbour graph and face-to-native-entry map for CSR and BSR assembly. Independent of the diffusion geometry plan. |
| `td::DiffusionPlan<Exec, Real>` / `transport/scalar/diffusion/plan.hpp` | Validated borrowed mesh, cell-face connectivity, geometry and communicator. Allocates no algebra graph, matrix or vector. |
| `td::DiffusionOperator<Exec, Real>` / `transport/scalar/diffusion/operator.hpp` | Adds integrated diffusion coefficients or a matrix-free residual. Stores a copy of the borrowed plan; allocates no coefficient or work arrays. |
| `ts::SystemView<Values, Rhs, Entries>` / `transport/scalar/system_view.hpp` | A descriptor containing writable rank-one coefficient/RHS views and a face-entry map. |
| `ts::make_system_view(...)` / `transport/scalar/assembly.hpp` | Opens compatible native CSR or selected BSR-component views without allocating another coefficient array. |
| `ts::AssemblyStatus<Exec>` / `transport/scalar/status.hpp` | Reusable device flags and host buffer for a complete batch of contributions. `check(comm)` completes work and performs collective diagnostics. |
| `ts::SourceOperator<Exec, Real>`, `UniformSource`, `CellSource` / `transport/scalar/source.hpp` | Independent additive volume source and its providers. The operator borrows owned-cell volumes. |
| `ts::SteadyDiffusion<Storage, Scalar, Exec, Real, PrecScalar>` / `transport/scalar/solvers/steady_diffusion.hpp` | Owns a scalar CSR matrix, RHS, solution, linear solver and assembly status. Shares the algebra plan and borrows one application-field component. |
| `transport::IsotropicDiffusion<Coefficient>` / `transport/diffusion/isotropic.hpp` | Shared physical law used by the spatial operator and scalar boundary conditions. Its namespace remains `krona::transport`. |
| `td::DiffusionOptions`, `td::BalanceResult` / `transport/scalar/diffusion/types.hpp` | Geometry-validation options and integrated balance diagnostics. |

Common assembly objects and sources remain in `transport::scalar`; diffusion-specific numerical types live in `transport::scalar::diffusion`. The ready-made steady solver remains in `transport::scalar`.

The low-level diffusion/residual and source headers do not require Tpetra. Include the algebra adapter or steady solver when using KRONA's linear-algebra wrappers. MPI and Kokkos must be active; initialize Tpetra as well for the algebra/solver path. Destroy all dependent objects before finalizing their runtimes.

Plan copies copy descriptors, not mesh arrays. A shared `FaceAssemblyPlan` allows several matrices to reuse one graph and entry map while retaining independent coefficients. Keep mesh/auxiliary owners, source/coefficient backing buffers, borrowed field allocations and communicators alive until their users and submitted work finish. Host containers must remain alive whenever compute views borrow their buffers.

## Mesh and geometry contracts

The host and compute mesh must use the same topology, local numbering and communicator. Faces are grouped into owned/owned, owned/ghost and physical-boundary intervals. Required data are:

- Face-cell connectivity and face geometry; cell centres for owned and real MPI ghost cells; positive owned-cell volumes.
- Complete `AuxConnType::CellFaces` incidence for owned cells.
- `AuxGeomType::FaceNormalDistance` and `AuxGeomType::FaceAreaOverNormalDistance` for all faces.
- `AuxGeomType::FaceInterpolationWeights` when using interpolated cell coefficients on inner faces.

Normals are unit vectors directed outward from the canonical face owner. Cached normal distances must be positive and consistent with coordinates; area/distance factors and any attached interpolation weights are checked against geometry.

`td::DiffusionOptions` defaults to `require_orthogonal = true` and `orthogonality_tolerance = 1e-6`. The check compares tangential centre separation with its normal projection. At a physical boundary, the separation is from the owner centre to the face centre.

The scalar boundary manager must borrow the **same compute mesh buffers** as the diffusion plan. Configure every labelled patch, including patches empty on a particular rank, and handle unlabelled boundary faces explicitly when present.

The matrix-free input needs the real-cell prefix of size `n_cells`: owned cells followed by MPI ghosts. It does not read boundary slots. A field used with boundary `apply_async()` needs the extended size

```text
n_cells + n_faces - n_inner_faces
```

where the tail stores one face-centroid value per physical boundary face. The steady solver itself only requires a writable component covering owned cells; allocate the extended field when downstream halo/boundary operations need it.

Topology and geometry are static while a diffusion plan is in use. After changing geometry, complete pending work, refresh the underlying geometry/metrics and revalidate by constructing a new diffusion plan. Connectivity or numbering changes also invalidate algebra graphs and face-entry maps. A geometry-only update does not require rebuilding an unchanged algebra graph/map.

## Constitutive law and spatial discretization

Use the same `transport::IsotropicDiffusion` law for interior diffusion and boundary flux conversion. A face-coefficient array is optional:

| Coefficient provider | Data and behaviour |
| --- | --- |
| `UniformCoefficient<Real>{D}` | Constant coefficient; no coefficient array or weights. |
| `FaceCoefficient{view}` | Rank-one face-indexed Kokkos View covering all local faces; no weights. |
| `InterpolatedCellCoefficient{view}` | Rank-one View covering owned and MPI ghost cells; harmonic interpolation by default. Refresh coefficient ghosts first. Inner faces require attached weights; physical boundaries use the owner value. |
| `InterpolatedCellCoefficient<View, ArithmeticCoefficient>` | The same storage/weight contract, with arithmetic interpolation. |
| `OwnerCellCoefficient{view}` | Owner extrapolation for exterior faces. The diffusion operator rejects this provider when inner faces exist. |
| Custom provider | Device-callable `coefficient(context)`; may compute a model directly or read existing fields. The caller guarantees accessible, correctly sized backing storage. |

These provider names are in `krona::transport`. Every sampled diffusion coefficient must be finite and strictly positive, including both cell values used by an interpolating provider. **Zero diffusion is currently rejected.** Coefficients are frozen at provider values during assembly: no derivatives of nonlinear coefficient models are inserted into the matrix.

For projected owner-to-face and face-to-neighbour distances `d_P` and `d_N`, the existing owner interpolation weight is `w = d_N/(d_P+d_N)`:

$$
D_f^{\rm arith}=wD_P+(1-w)D_N,\qquad
D_f^{\rm harm}=\frac{1}{(1-w)/D_P+w/D_N}.
$$

The harmonic expression follows from series resistances `d_P/D_P + d_N/D_N`; its distance fractions therefore appear in the opposite positions to arithmetic interpolation. Missing geometric weights are not replaced by `0.5`.

With `d = x_N-x_P` and `d_n = d\cdot n_f > 0`, the integrated owner-outward flux is

$$
Q_P=-D_f\frac{A_f}{d_n}(\phi_N-\phi_P),\qquad a_f=D_fA_f/d_n>0.
$$

An owned row receives `+a_f` on its diagonal and `-a_f` at the neighbour entry. Multiple connecting faces accumulate into the same graph entry. One kernel iteration owns one row, so interior assembly needs no atomics. Boundary faces sharing an owner use atomic additions.

**There is no nonorthogonal correction.** Stored `face_non_ortho_*` vectors are not used, and boundary relations receive a zero reconstruction increment. Disabling `require_orthogonal` only disables the geometry check. Cached normal distances and area/distance factors are mandatory; computing them on demand is not implemented.

Built-in scalar conditions are `FixedValue`, `NormalGradient` and `FixedFlux`; registered custom conditions return a scalar `Relation`, allowing affine closures such as Robin conditions. Assembly converts the relation into integrated outward flux `Q = a*phi_owner + c`, requiring finite coefficients and `a >= 0`. It adds `+a` to the diagonal and `-c` to the RHS. `FixedFlux` is converted using the supplied physical law.

## Shared setup and a complete steady solve

This example assumes an orthogonal mesh with exactly two labelled patches, IDs `0` and `1`, and no unlabelled boundary faces. `host_mesh`, `mesh`, `connectivity`, `geometry`, `comm` and `exec` are already prepared and remain alive. Attach mesh halo data if using the exchange below.

```cpp
#include <krona/transport/scalar/solvers/steady_diffusion.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>

using namespace krona;
namespace ts = transport::scalar;
namespace td = transport::scalar::diffusion;
namespace bc = boundary::scalar;
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;

// Separate plans: borrowed spatial data and shared algebra graph/entry map.
td::DiffusionPlan<Exec, Real> plan(
    host_mesh, mesh.view(), connectivity.view(), geometry.view(), comm, exec);
linalg::FaceAssemblyPlan<Exec> shared(host_mesh, comm, exec);

bc::Manager<Exec, Real> boundary(host_mesh, mesh.view(), exec);
boundary.set(0, bc::FixedValue<Real>{0});
boundary.set(1, bc::FixedValue<Real>{1});

fields::FieldStorage<StorageReal, 1, Exec> phi(
    "phi", boundary.required_field_size(), fields::FieldLocation::Custom, exec);
phi.fill(exec, StorageReal{0}); // Initial owned-cell guess and initialized tail.

ts::SteadyDiffusionOptions options;
options.linear.method = "GMRES";
options.linear.preconditioner.type = "AMG"; // Already the defaults.
ts::SteadyDiffusion<> solver(plan, shared, phi.view().component(0), options);

halo::Exchange<StorageReal, Exec> exchange(
    mesh.view(), mesh.halo_plan(), comm, 17, exec, halo::Transfer::HostStaged);
exchange.add_field(phi.view());
exchange.prepare(); // Reuse this exchange; register other fields before prepare().

const auto law = transport::IsotropicDiffusion{
    transport::UniformCoefficient<Real>{2}};
const ts::UniformSource<Real> source{0};
const auto result = solver.solve_owned(boundary, law, source);
if (result.converged) {
    exchange.exchange(); // Blocking publication to application MPI ghosts.
    boundary.apply_async(phi.view(), law); // Boundary-face slots on boundary's exec.
    exec.fence("Scalar field ready for host access or another execution instance");
}
```

Every rank participates in algebra-plan construction and collective solves, including empty ranks. The diffusion plan performs blocking local validation; it no longer constructs an algebra plan implicitly.

`solve_owned(boundary, law, source, time)` clears its own matrix/RHS once, adds diffusion and source, checks the complete steady system, commits the wrappers and solves. The current owned field is the initial guess. On convergence it completes the owned-cell copy back before returning. On nonconvergence the application field stays unchanged; `solution()` retains the latest algebraic iterate.

Application ghosts and boundary slots are not published by the solve. Tpetra imports its own remote vector entries, independently of KRONA's field halos. Several steady solvers can share the plans and publish their fields through one grouped exchange afterwards.

The borrowed field component may be strided. It must be writable, accessible from `Exec` and use the selected `Storage` type. Template defaults are `StorageReal`, `SolverReal`, `execution::ExecutionSpace`, `ComputeReal` and `PreconditionerReal`. Different solver/preconditioner scalar types introduce the conversion storage and operator-adapter costs described in `LINALG.md`; matching types avoid that precision adapter.

## Adding diffusion to an external system

The low-level operator owns neither a matrix nor an RHS. The following continues from the setup above but creates a separate externally owned system:

```cpp
#include <krona/transport/scalar/assembly.hpp>
#include <krona/linalg/solver.hpp>

linalg::CsrMatrix<SolverReal, Exec> matrix(shared.pattern());
linalg::Vector<SolverReal, Exec> rhs(matrix.distribution());
linalg::Vector<SolverReal, Exec> solution(matrix.distribution());
linalg::Solver<decltype(matrix)> linear(matrix);
td::DiffusionOperator<Exec, Real> diffusion(plan);
ts::SourceOperator<Exec, Real> sources(mesh.view(), exec);
ts::AssemblyStatus<Exec> status(exec); // Allocate once; reuse for later batches.

const auto target = ts::make_system_view(matrix, rhs, shared, exec);
Kokkos::deep_copy(exec, target.values, SolverReal{0});
Kokkos::deep_copy(exec, target.rhs, SolverReal{0});

status.begin_async();
diffusion.add_system_async(target, boundary, status, law);
sources.add_async(target.rhs, status, source);
// Other compatible contributions can be added here, on the same exec instance.
const auto report = status.check(comm); // Blocking collective diagnostics.
(void)report;
matrix.commit();
rhs.commit();

solution.copy_from_component(phi.view().component(0), exec);
solution.commit();
const auto result = linear.solve(rhs, solution);
if (result.converged) {
    solution.copy_to_component(phi.view().component(0), exec);
    solution.commit();
}
```

`add_system_async(target, boundary, status, law, time)` preserves existing coefficients and RHS entries. Clear the complete system **once**, outside the contributing operators, then add every term. Repeating an additive call adds its contribution again. Source terms are separate and are never implicitly included by the diffusion call.

`SystemView` contains `values`, `rhs` and `entries`. Its entry map provides `diagonal(cell)`, `owner_entries(face)` and `neighbour_entries(face)`; the last applies to faces between owned cells. These are native matrix-entry positions, so kernels do not search matrix rows.

`make_system_view` requires the exact shared native graph and the matrix's exact point-distribution object for the RHS. Matching dimensions alone are insufficient. It also checks the component index. The application must ensure that the assembly map and diffusion plan describe the same local mesh numbering. Native ReadWrite access can synchronize Tpetra's existing host/device storage, but the adapter allocates no additional coefficient or packing array.

Unlike `SteadyDiffusion`, this low-level path does not impose positive whole-system diagonals, a value anchor or general solvability. The application owns these complete-system decisions.

### Selecting a scalar equation in BSR storage

The same face-entry map works for a BSR graph with identical cell numbering. For example, assemble the third scalar equation/variable component in blocks of width three:

```cpp
#include <krona/linalg/bsr_matrix.hpp>

linalg::BsrMatrix<SolverReal, Exec> block_matrix(shared.pattern(), 3);
linalg::Vector<SolverReal, Exec> block_rhs(block_matrix.distribution());
Kokkos::deep_copy(exec, block_matrix.overwrite_blocks(exec), SolverReal{0});
Kokkos::deep_copy(exec, block_rhs.overwrite(exec), SolverReal{0});

const auto slice = ts::make_system_view(block_matrix, block_rhs, shared, exec, 2);
status.begin_async();
diffusion.add_system_async(slice, boundary, status, law);
sources.add_async(slice.rhs, status, source);
(void)status.check(comm);
block_matrix.commit();
block_rhs.commit();
```

The selected coefficient view addresses `(native_block_entry, 2, 2)` with the correct strides; the RHS slice addresses cell-major scalar DOFs `cell*3 + 2`. Other components and cross-component coefficients are preserved. Here they were initialized to zero; this snippet fills only one equation component, so the remaining equations must be assembled before solving the complete block system. This is scalar contribution assembly into BSR storage, not a general coupled diffusion law.

## Matrix-free residual

With current real-cell ghost values, compute the integrated residual without constructing a graph, matrix or Tpetra vector:

```cpp
#include <krona/transport/scalar/diffusion/operator.hpp>

using Space = typename Exec::memory_space;
Kokkos::View<Real*, Space> residual("residual", host_mesh.n_own);
td::DiffusionOperator<Exec, Real> diffusion(plan);
ts::SourceOperator<Exec, Real> sources(mesh.view(), exec);
ts::AssemblyStatus<Exec> status(exec);

// Refresh phi's MPI ghosts before submission if its owned values have changed.
Kokkos::deep_copy(exec, residual, Real{0});
status.begin_async();
diffusion.add_residual_async(phi.view().component(0), residual, boundary, status, law);
sources.add_async(residual, status, source);
(void)status.check(comm);
// residual now contains b_diffusion + b_source - A_diffusion*phi.
```

The diffusion call adds only `b_boundary - A_diffusion*phi`. Source integration is a separate contribution. Existing residual values are preserved; zero them only when starting a new complete residual.

The input must cover `n_cells`; output must cover owned cells. Both may be strided native rank-one Kokkos Views. Only the owned output prefix is modified, and extended boundary values are not read. **Input and output must not overlap**, including partially overlapping/strided aliases. The implementation rejects identical base pointers for nonempty owned ranges; detecting every possible overlapping alias remains the caller's responsibility. Pure Neumann diffusion is allowed in this path.

## Sources and variable coefficients

A source is a device-callable `source(std::size_t cell)` returning a finite density. Negative densities are sinks. Only owned entries are read, so no source halo is required. `UniformSource<Real>{3}` contributes `3*cell_volume` to the RHS/residual. `CellSource{view}` reads an existing rank-one Kokkos View.

To compute a source directly without a separate density array:

```cpp
template<class X, class R = krona::ComputeReal>
struct LinearSource {
    X x;
    R offset = 0, slope = 0;
    KOKKOS_INLINE_FUNCTION R operator()(std::size_t cell) const
    { return offset + slope*static_cast<R>(x(cell)); }
};

// In the setup/assembly code:
const auto x = mesh.view().geometry.cell_centroid_x;
const LinearSource<decltype(x), Real> spatial_source{x, Real{3}, Real{2}};
const auto result = solver.solve_owned(boundary, law, spatial_source);
```

The provider returns `3 + 2*x`; the source operator performs volume multiplication. Carry any model parameters or time explicitly in the functor: assembly's `time` argument is passed to face contexts, not as a separate source argument. `sources.add_async(output, status, provider)` can target a system RHS or an independent residual View.

For cell-dependent diffusion, attach interpolation weights and prepare an existing coefficient View `D` covering owned and MPI ghost cells:

```cpp
const auto variable_law = transport::IsotropicDiffusion{
    transport::InterpolatedCellCoefficient{D}}; // Harmonic by default.

using CellD = std::remove_cvref_t<decltype(D)>;
const auto arithmetic_law = transport::IsotropicDiffusion{
    transport::InterpolatedCellCoefficient<CellD, transport::ArithmeticCoefficient>{D}};
```

Refresh `D`'s ghosts before use. Custom providers must support both interior and boundary contexts. Common members include `face`, `owner`, position, unit normal, `distance`, `area` and `time`; interior contexts also supply `neighbour` and `owner_weight`. The latter is meaningful only with attached weights. Provider values must agree across MPI interfaces for conservative assembly.

Defaults are `D = 1` and `time = 0`; a default source provider gives zero. The diffusion methods accept `(target-or-field/output, boundary, status, law, time)` as shown above, while the ready-made solver accepts `(boundary, law, source, time)`.

## Asynchrony and completion

All contributions in a batch use the plan/source operator's stored execution instance. `AssemblyStatus::device_flags(exec)` requires it to match the status instance. Diffusion boundary kernels use host patch dispatch and run on the diffusion plan's instance; boundary `apply_async()` separately uses the boundary manager's instance.

| Operation | Completion contract |
| --- | --- |
| `DiffusionPlan` construction | Blocking local geometry/incidence validation; no implicit algebra-plan construction. |
| `FaceAssemblyPlan` construction | Collective blocking topology/graph setup and upload. |
| `make_system_view()` | Opens native ReadWrite views and marks algebra wrappers pending; native storage access may synchronize. No zeroing or commit. |
| `status.begin_async()` | Queues flag reset and opens one contribution batch. Does not clear coefficients, RHS or residual. |
| `add_system_async()`, `add_residual_async()`, source `add_async()` | Local kernel submission on the stored instance; no MPI, fence, automatic reset or algebra commit. |
| `status.check(comm)` | Copies flags to host, fences that instance, collectively reduces diagnostics and closes the batch. Does not commit algebra wrappers. |
| Matrix/vector `commit()` | Completes pending access on its recorded instance and clears wrapper pending state. Does not replace transport diagnostics. |
| `solve_owned()` | Collective blocking assembly, diagnostics, commits, solve and successful owned-field publication. |
| `balance()` | Blocking local reductions followed by a collective MPI sum. |

The normal external-system sequence is: initialize targets once, `begin_async()`, submit contributions, `check(comm)`, commit all touched algebra wrappers, then call the linear solver. A fence alone does not clear wrapper state or perform collective diagnostics. Even though `check()` fences submitted work, explicit wrapper commits are still required.

Keep every captured buffer alive and unmodified until its consumers finish. Data produced on the **same execution instance** is ordered before later submissions without intermediate fences. For a different instance, complete the producer or establish an explicit dependency before reading its data. Apply the same rule to halo unpacking and boundary-value publication. The module does not perform application halo exchanges for you.

Do not start another batch on a pending status; call `check()` first. One status can aggregate several contributions without losing an earlier error. Its destructor drains pending local work but does not replace collective checking. For matrix-free arrays, no algebra-wrapper commits exist: `check()` completes that batch, and later operations require their own ordering.

## Balance, validation and limitations

An optional collective diagnostic is available for any supplied valid field:

```cpp
const auto balance = solver.balance(boundary, law, source);
// Or: diffusion.balance(phi_component, boundary, law, source, time).
// source_integral, outward_boundary_flux, defect = flux - source.
```

It evaluates the current source and boundary law from owned values. It does not read boundary slots or reuse assembly integrals, and it has no assembled/committed-operator prerequisite. Supply the law, source, time and backing data corresponding to the state being checked. This is a synchronized physical boundary/source balance, not a residual norm or a solvability test.

Validation is deliberately split by scope:

- Plan construction checks mesh incidence, geometry, cached metrics and optional orthogonality.
- Host metadata checks known provider/View shapes and execution accessibility. Opaque custom functors are responsible for their own accesses and backing-storage sizes.
- Contribution kernels flag invalid sources, nonpositive/nonfinite diffusion, unacceptable affine boundary fluxes and nonfinite/unrepresentable updates. `check()` aggregates those flags collectively and reports whether any diffusion value/Robin anchor was encountered.
- Only `SteadyDiffusion` additionally checks a positive finite complete-system diagonal and requires a global diffusion anchor for nonempty systems. The low-level contribution/status path imposes neither requirement.

Validation does not prove discretization accuracy or general matrix solvability. The steady solver rejects pure Neumann problems because no reference, mean constraint or nullspace handling is implemented. Its global anchor check does not detect separately unanchored disconnected components. Empty ranks and globally empty systems are supported.

Current numerical limitations are TPFA without nonorthogonal correction, strictly positive isotropic diffusion, frozen coefficients/explicit source providers, and no automatic nonlinear iteration. The ready-made steady solver owns a scalar CSR system. Low-level assembly also supports selected diagonal BSR component slices, preserving other components; it does not create cross-component diffusion terms. The matrix is the frozen diffusion linearization, not a full Jacobian of nonlinear coefficient/source models.

## Future extensions

- Add convection driven by a supplied face mass flux and conservative temporal terms at the equation/system layer. External ownership and additive diffusion assembly are already implemented foundations for this composition.
- Add nonorthogonal correction with an explicit gradient, boundary and iteration/update contract.
- Allow a geometry policy to select cached metrics or compute actual distances, area/distance factors and interpolation weights inside kernels. Missing weights must not silently become `0.5`.
- Extend coefficient/assembly policies to anisotropic or coupled diffusion, and provide explicit reference/nullspace treatment where needed.
- Define geometry refresh and graph/map invalidation rules for moving or adaptive meshes before using these static-mesh plans there.
