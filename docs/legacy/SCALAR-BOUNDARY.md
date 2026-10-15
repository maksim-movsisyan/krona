# Scalar boundary conditions

`krona::boundary::scalar` supplies scalar face values and the corresponding
diffusive-flux coefficients. It knows mesh geometry and fields, but does not depend
on Trilinos, a particular equation, or a physical state vector.
`krona::boundary::PatchCatalog` in `boundary/patches.hpp` is shared infrastructure
for future physical boundary modules; it does not depend on scalar conditions.

## Storage and execution contracts

- Faces are grouped; each local patch occupies one contiguous absolute face
  interval `[begin,end)`. Setup scans host `face_patch` once. Neither `patch_faces`
  nor a device patch-ID array is retained.
- Patch metadata is replicated across ranks. IDs are indices in `MeshPart::patches`;
  names must match exactly. Empty local patches are valid and launch no work.
- A field contains at least `n_cells + n_faces - n_inner_faces` entries.
  Boundary face `f` uses slot `n_cells + f - n_inner_faces`.
  **These slots contain values at face centroids, not mirrored ghost-cell values.**
- Only those boundary slots are written. Owned cells, MPI ghosts, extra trailing
  entries and other field components are left untouched.
- When using KRONA gradients, select **`BoundaryValues::Face`**, overriding the
  gradient module's current `MirrorCell` default. Gradient boundary/ghost output
  slots are not populated by this scalar-value module.
- Setup validates geometry and completes before returning. `apply_async` submits
  work on the manager's execution-space instance and performs no fence or MPI.
  Kokkos may execute some host backends synchronously; the API does not promise
  actual overlap on every backend.
- Borrowed compute geometry and field/provider/gradient buffers must outlive all
  submitted work. Finish work before freeing/reallocating them or finalizing Kokkos.
  Use the same execution-space instance for dependent kernels, or synchronize explicitly.
- `Real` defaults to `ComputeReal`; storage values are converted at reads/writes.
  Uniform diffusivity is checked on the host. Face diffusivities and correction
  gradients must be valid before kernels consume them; apply does not reduce them
  back to the host for validation.

## Conditions and signs

Let `n` be the owner's outward unit normal, `d = x_f - x_P`,
`h = d dot n > 0`, and `tau = grad(phi)_P dot (d - h*n)`.
Without `GradientCorrection`, `tau = 0` is the orthogonal approximation.
Diffusivity `D` is the scalar coefficient used by the equation at the face.

`Relation` returns four coefficients:

```text
phi_f = value_owner * phi_P + value_constant
q_n   = flux_owner  * phi_P + flux_constant
q_n   = -D * d(phi)/dn
```

`q_n` is **outward diffusive flux per unit area**, not an integrated flux.
Multiply it by face area exactly once. The supported conditions are:

| Condition | Face value | Outward flux density |
| --- | --- | --- |
| `FixedValue{v}` | `v` | `D/h * (phi_P - v + tau)` |
| `NormalGradient{g}` | `phi_P + h*g + tau` | `-D*g` |
| `FixedFlux{q}` | `phi_P - h*q/D + tau` | `q` |

`NormalGradient{0}` is zero normal derivative; on a skew face, its corrected
face-centroid value can differ from `phi_P`. `D >= 0` is valid for fixed value
and normal derivative; fixed flux requires **`D > 0`**, even when `q = 0`.
The default `UniformDiffusivity{1}` is convenient for value filling, but physical
assembly and flux boundaries must use the equation's actual diffusivity.
Spatially/time-varying conditions can use `Context` through custom functors.

## TOML

Keep the partitioner's `[[boundary_condition]]` entries and metadata. Its top-level
physical `type` does not select a scalar condition. Add one table per scalar:

```toml
[[boundary_condition]]
patch_id = 0
name = "inlet"
family = "fluid"
source_type = "BCInflow"
type = "INLET_TOTAL_TP" # Reserved for the physical boundary module.

[boundary_condition.scalar.temperature]
type = "FIXED_VALUE"
value = 300.0

[boundary_condition.scalar.tracer]
type = "NORMAL_GRADIENT"
gradient = 0.0

[[boundary_condition]]
patch_id = 1
name = "wall"
type = "USER_DEFINED"

[boundary_condition.scalar.temperature]
type = "FIXED_FLUX"
flux = 100.0 # Positive means diffusion out of the domain, per unit area.

[boundary_condition.scalar.tracer]
type = "ZERO_GRADIENT" # No parameter.

[unlabelled.scalar.temperature]
type = "ZERO_GRADIENT"

[unlabelled.scalar.tracer]
type = "ZERO_GRADIENT"
```

`read_config(path, "temperature")` extracts only that scalar's tables.
Each labelled mesh patch needs an explicit condition, including patches empty
on the current rank. If global metadata or local faces show unlabelled boundaries,
an explicit `unlabelled` condition is mandatory. There are no inferred conditions
from import labels. Scalar keys are strict; unrelated metadata and other physics
tables are ignored. The parser accepts integer or floating-point finite parameters.

Reading/parsing is host-only and local; no implicit broadcast or collective occurs.
All participating ranks must receive the same configuration, for example by reading
the same shared file. Parse/binding errors terminate through `krona::mpi::fatal`.
The parser supports toml++ exception and parse-result modes; C++ exceptions remain
enabled in KRONA. toml++ headers stay private to `config.cpp`.

## Application

```cpp
namespace scalar = krona::boundary::scalar;
using Exec = krona::execution::ExecutionSpace;
using Real = krona::ComputeReal;

Exec exec;
auto config = scalar::read_config("bc.toml", "temperature");
scalar::Manager<Exec> boundary{mesh_part, mesh_storage.view(), config, exec};

boundary.apply_async(temperature, scalar::UniformDiffusivity<Real>{diffusivity});
// Subsequent dependent kernels use exec. Fence before host access or external consumers.
```

`temperature` is a mutable scalar `FieldView`. For a vector component, use
`apply_component_async(state.component(c), diffusion)`. Both SoA and strided AoS
component views work. Variable diffusivity uses `FaceDiffusivity{face_view}` indexed
by **absolute mesh face ID**. Nonorthogonal correction uses:

```cpp
auto correction = scalar::GradientCorrection{gx, gy, gz}; // Rank-one owned-cell gradients.
boundary.apply_async(temperature, diffusion, correction, time);
```

For a corrected solve, first obtain valid gradients using initialized boundary values;
then refresh boundary values and explicit corrections as the nonlinear/deferred
iteration requires. Passing uninitialized gradients is not a bootstrap procedure.

## Implicit assembly

The manager does not allocate CSR slots or assemble a matrix itself. Pass a
small device-callable sink to consume the same relations as field filling:

```cpp
struct AddDiffusionBoundary {
    Values diagonal, rhs; // Equation-owned accessible Views; this example uses Real values.
    KOKKOS_INLINE_FUNCTION void operator()(const scalar::Context<Real>& c,
                                          const scalar::Relation<Real>& r) const
    {
        auto owner = static_cast<std::size_t>(c.owner);
        Kokkos::atomic_add(&diagonal(owner), c.area * r.flux_owner);
        Kokkos::atomic_add(&rhs(owner), -c.area * r.flux_constant);
    }
};

boundary.evaluate_async(AddDiffusionBoundary{diagonal, rhs}, diffusion, correction, time);
```

`evaluate_async` submits one typed kernel per nonempty patch, with no temporary
relation array or fence. The sink owns the indexing, precision conversions and
write-conflict handling of its output. No hidden field fill is performed.
For custom launch policies, `for_each_patch(visitor)` exposes host dispatch and
`geometry_view().context<Real>(...)` exposes device geometry. Use named kernel
functors when launching from a generic visitor: NVCC does not accept an extended
CUDA lambda nested inside a generic host lambda.

This sign convention corresponds to `div(-D*grad(phi)) = source`, or equivalently
`-div(D*grad(phi)) = source`, with integrated cell balances. A volume-normalized
equation must divide contributions by the owner volume. Accumulate shared cell slots
with atomics, or use a cell-based assembly to avoid concurrent writes.
The example's `diagonal` is the equation's diagonal coefficient array; its CSR slot
mapping remains the assembler's responsibility. Filling boundary field slots alone
does not impose conditions on an implicit matrix.

The advection scheme selects how to use `phi_f`: for first-order upwind, outward
flow uses the owner value, while inward flow needs appropriate incoming data.
This module does not implement advection switching, limiters, periodic interfaces,
or physical coupled boundary states. An all-Neumann diffusion problem still needs
compatible sources/fluxes and a nullspace/reference treatment in the equation/solver.

## Extension

Custom conditions are compile-time kernel types, returning `Relation<Real>` from
`operator()(const Context<Real>&)`. Register their types in the manager:

```cpp
struct MovingValue {
    KOKKOS_INLINE_FUNCTION scalar::Relation<Real> operator()(const scalar::Context<Real>& c) const
    { return scalar::FixedValue<Real>{Real{300} + c.x + c.time}(c); }
};

scalar::Manager<Exec, Real, MovingValue> boundary{mesh_part, mesh_storage.view(), exec};
boundary.set(0, MovingValue{});
// Assign every remaining labelled patch, and unlabelled faces when present.
```

No variant, virtual call, string lookup or allocation enters a face kernel.
Built-ins are parsed by the current TOML reader; custom types are assigned in C++.
Future physical modules can share the patch metadata/file while using separate
field-vector kernels and their own parameter validation.
