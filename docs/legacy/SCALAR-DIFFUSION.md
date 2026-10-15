# Scalar diffusion contracts

## Equation and signs

The stationary equation is `-div(D * grad(phi)) = s`, with outward flux density
`q_n = -D * d(phi)/dn`. Cell equations are **integrated**, not divided by volume:
`sum(outward face flux * face area) = s_P * V_P`.

For an interior face, `h = dot(x_N - x_P, n_owner)` and `a_f = D_f * A_f / h`.
The owner equation receives `+a_f * phi_P - a_f * phi_N`; the neighbour equation
receives the opposite face flux. Boundary `Relation` contributes
`+A_f * flux_owner` to the diagonal and `-A_f * flux_constant` to the RHS.
Area is multiplied exactly once. Boundary values are not extra matrix unknowns.

## Mesh and coefficients

- Faces must be grouped into owned-owned, owned-ghost and exterior intervals.
  Select `MeshData::FiniteVolume` for the complete steady solver.
- Owned volumes and face areas are positive; normals are unit, outward from owner.
  Interior centre-to-centre and exterior centre-to-face normal distances are positive.
- Mesh geometry is stationary. Changing geometry, connectivity or array addresses
  requires rebuilding the operator and its dependent objects.
- The default check requires tangential displacement / normal distance no larger
  than `orthogonality_tolerance`. `require_orthogonal=false` only disables this
  check: the same uncorrected two-point formula is still used.
- `D_f` is a finite, strictly positive scalar on **every** local face. The caller
  supplies face values/interpolation, including a harmonic interface value when
  required by the material model. Copies of an MPI cut face must use consistent
  geometry and `D_f` on both ranks.
- `source(cell)` returns finite source density on owned cells.
  `diffusion(face)` uses absolute local face indices, including boundary faces.
  No source or diffusivity halo exchange is hidden in assembly.
- Custom BCs return the existing `Relation<Real>`. The implicit flux owner
  coefficient must be finite and nonnegative. Each connected component needs an
  anchoring boundary coefficient; this first module assumes a connected domain.
  The collective check detects a globally unanchored problem, not disconnected
  components. Pure-Neumann reference constraints are not implemented.

The physical model is scalar isotropic diffusion. There is no tensor diffusivity,
advection, reaction, transient term, or explicit nonorthogonal correction yet.

## Storage and assembly

`DiffusionOperator` creates mesh-independent algebraic numbering through the
existing linalg adapter. Its CSR graph, matrix allocation, RHS vector and row plan
are reused. Original mesh GIDs do not become matrix row IDs.

Native Tpetra entry IDs are resolved once during setup. Per owned row the plan
stores an incidence interval, face IDs and off-diagonal entry IDs, plus one
diagonal entry ID. An interior face also has cached `A_f/h`. There is no second
coefficient matrix or intermediate CSR value buffer. Multiple faces joining the
same pair of cells accumulate into the same coefficient.

One Kokkos thread assembles each owned row without atomics. Boundary kernels
dispatch once per nonempty patch on the host and use atomics when multiple faces
contribute to the same diagonal/RHS. Conditions are concrete types in kernels.
All assembly kernels use the operator's supplied execution-space instance.

```cpp
ts::DiffusionOperator<> op{mesh_part, mesh_view, comm, exec};
op.assemble_async(boundary, source, diffusion);
// Independent application work may be submitted here.
op.commit(); // collective, blocking; data/anchoring checks

krona::linalg::Vector<> solution{op.matrix().pattern().distribution()};
krona::linalg::Solver<> linear{op.matrix(), linear_options};
// Initialize and commit solution before the solve.
const auto result = linear.solve(op.rhs(), solution);
```

Every rank in `comm`, including empty ranks, must participate in construction,
each `commit()`, linear solve and balance diagnostic in matching order.
Assembly itself performs no MPI. A pending batch must be committed before the
next assembly or a native operation. Construction validates geometry; commit
checks current source, diffusivity and assembled values. Positive coefficients
must remain representable in `SolverReal`.

## Complete solve and lifetimes

`SteadyDiffusion` borrows one writable rank-one component with at least
`n_cells + n_faces - n_inner_faces` elements. Unit and nonunit strides are
supported. Owned values provide the initial guess. Extra components/tail entries
are not modified.

On successful convergence it publishes the owned solution, completes its halo
exchange, and fills boundary slots with face-centroid values. On failure it
returns the `SolveResult` without modifying the borrowed field; the latest native
iterate is available through `solution()`. No boundary gradients are filled.

The solver duplicates `comm` once to isolate its halo tags. Host-staged MPI is
the default; `DeviceDirect` remains an explicit caller-supported choice.
The compute arrays, registered field and runtimes must outlive the solver.
Host patch/peer metadata is copied during construction. A GPU mesh may release
its MeshPart afterwards; CPU MeshStorage still borrows its selected MeshPart
arrays, so their owner must remain alive. The low-level DiffusionOperator borrows
its communicator, while SteadyDiffusion owns its duplicate. The compute mesh
and boundary manager must refer to the same arrays. Finish unrelated asynchronous
users before modifying their inputs.

Source/BC/D data used by `assemble_async()` must remain valid until `commit()`.
Pending producers of these data or the initial guess must use the solver's
execution instance, or complete before the corresponding read is submitted.
The full `solve()` is collective and blocking. Its matrix assembly uses its own
`Exec`; final boundary filling may use the manager's instance, after halo
completion, and is fenced before return.

`Real` defaults to `ComputeReal`, coefficients and Krylov vectors to `SolverReal`,
and the published component to `StorageReal`. The boundary manager uses the same
`Real` as the operator. GMRES/Ifpack2 are the existing linalg implementation.
The true residual reported by Trilinos concerns its solver vector; rounding into
the field can change the physical balance. Optional `balance()` integrates the
supplied/published field using `AccumulationReal`, then globally sums double
diagnostic totals. It is blocking and must use the same source, D, time and BC
parameters as assembly. It does not run automatically inside `solve()`.

`Context::boundary_index` is relative to the boundary block. The field index is
`n_cells + context.boundary_index`, equivalently `geometry.extended_index(face)`.
When later supplying boundary values to the gradient module, select its
`BoundaryValues::Face` mode.

## Numerical references

The tests cover 3D affine exactness, mixed value/flux/normal-gradient conditions,
layered material conduction, second-order refinement of `phi=x*(1-x)`, source
integration, duplicate faces, strided fields, graph reuse, storage rounding,
subcommunicators, empty ranks and explicit invalid-input rejection.

The two-point discretization and its orthogonality limitation follow the usual
finite-volume treatment described in CFD Direct's
[Laplacian discretisation](https://doc.cfd.direct/notes/cfd-general-principles/laplacian-discretisation)
and [surface-normal gradient](https://doc.cfd.direct/notes/cfd-general-principles/surface-normal-gradient).
