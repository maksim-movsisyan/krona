# Scalar convection

KRONA's scalar convection module provides the cell-integrated finite-volume contribution of

$$
\nabla\cdot(\rho\mathbf u\phi).
$$

The application supplies the **signed, face-integrated mass flow rate**. The operator uses donor-cell upwind coefficients for the implicit matrix and puts any higher-order reconstruction increment into the RHS. It can also evaluate the same convective residual directly from a field.

The module is a spatial building block. Diffusion, volume sources, time discretization, gradient/limiter preparation, halo exchange and the linear/nonlinear solution procedure are composed by the caller. It has no equation-of-state dependency and uses the same interface for compressible and incompressible applications.

The shared algebraic convention is

$$
A\phi=b,\qquad r=b-A\phi.
$$

Equations, RHS entries and residuals are integrated over cells. The convection contribution is never divided by cell volume.

## Objects and ownership

Examples use:

```cpp
using namespace krona;
namespace cv = transport::scalar::convection;
namespace ts = transport::scalar;
namespace bc = boundary::scalar;
namespace rec = numerics::reconstruction;
using Exec = execution::ExecutionSpace;
using Space = typename Exec::memory_space;
using Real = ComputeReal;
using Scalar = SolverReal;
```

Header paths in the table are relative to `include/krona/`.

| Object / header | Responsibility |
| --- | --- |
| `cv::ConvectionPlan<Exec, Real>` / `transport/scalar/convection/plan.hpp` | Validates and borrows the static mesh, owned-cell face incidence, execution instance and communicator. No geometry cache, graph or work-array allocation. |
| `cv::ConvectionOperator<Exec, Real>` / `transport/scalar/convection/operator.hpp` | Adds convection coefficients/RHS, evaluates a residual, or adds only the reconstruction correction. Stores a copy of the borrowed plan. |
| `cv::FaceMassFlux<View>`, `cv::UniformMassFlux<Real>` / `transport/scalar/convection/types.hpp` | Supply already integrated, signed mass flow rates. |
| `cv::FixedValueInflow` / the same `types.hpp` | Default incoming scalar value, obtained from the patch's scalar `FixedValue` condition. |
| `cv::InflowProfiles<Exec, Fallback>`, `cv::make_inflow_profiles(...)` / `transport/scalar/convection/inflow_profiles.hpp` | Host-side bindings of compact profile views to patches, with a fallback for unbound patches. |
| `cv::PatchInflowValues<Value>` / `transport/scalar/convection/types.hpp` | Small device descriptor selected by the registry: a borrowed pointer, stride and local patch start. |
| `cv::Reconstruction`, `cv::make_reconstruction(...)`, `cv::NoReconstruction` / `transport/scalar/convection/reconstruction.hpp` and `types.hpp` | Adapt a numerical reconstruction to the increment of one transported scalar component. |
| `linalg::FaceAssemblyPlan<Exec>` / `linalg/face_assembly.hpp` | Shared mesh-neighbour graph and face-to-native-entry maps for scalar CSR and diagonal-component BSR assembly. |
| `ts::make_system_view(...)` / `transport/scalar/assembly.hpp` | Opens writable native coefficient/RHS slices with the matching entry map. |
| `ts::AssemblyStatus<Exec>` / `transport/scalar/status.hpp` | Reusable device diagnostics for an entire assembly/residual batch; `check(comm)` is blocking and collective. |

Include `convection/operator.hpp` for the spatial API, reconstruction adapters and profile registry. The spatial headers do not require Tpetra. Include `scalar/assembly.hpp` and the appropriate linear-algebra headers for algebraic assembly.

Mesh/connectivity storage, borrowed field allocations and communicators must outlive their users and submitted work. Plan copies copy descriptors rather than mesh arrays. A profile registry allocates only small host metadata: its boundary catalog must remain alive, and every registered allocation must remain alive until all kernels using it finish. **Registration does not retain a managed View or increment its allocation's reference count.** Custom fallback functors are stored by value and retain their own members' ownership semantics.

MPI and Kokkos must be active. Initialize Tpetra as well when using KRONA's matrix/vector/solver wrappers; destroy dependent objects before finalizing their runtimes.

## Mesh, indices and geometry

The plan requires the same host/compute local numbering and communicator, grouped faces, and:

- `MeshData::FaceCells`, `CellGeometry` and `FaceGeometry`.
- Complete `AuxConnType::CellFaces` incidence for owned cells.
- Finite cell centres and positive cell volumes, including real MPI ghosts, as required by the shared mesh validator.
- Valid face geometry with positive areas and outward unit normals.

Plan construction performs blocking local validation. Convection requires no cached diffusion distances, area/distance factors, interpolation weights or nonorthogonality vectors. Reconstruction geometry is obtained from the mesh coordinates by the selected stencil provider.

Normals point from the canonical owner to the neighbour. At physical boundaries they point out of the owned cell. The local face intervals are:

| Interval | Faces |
| --- | --- |
| `[0, n_internal_faces)` | Owned/owned |
| `[n_internal_faces, n_inner_faces)` | Owned/MPI-ghost |
| `[n_inner_faces, n_faces)` | Physical boundary |

All face indices used by this module are **local to the rank**, not global distributed face IDs. Physical faces of each patch form one contiguous local interval `[range.begin, range.end)` in `boundary.catalog()`.

The scalar boundary manager must borrow the same compute mesh buffers as the convection plan. Configure every labelled patch, including locally empty patches, and explicitly configure unlabelled boundaries when present. A profile binding supplies convection data; it does not replace this manager configuration.

## Supplied mass flow rates

For a stationary mesh, the usual physical interpretation is

$$
\dot m_f=(\rho\mathbf u\cdot\mathbf n)_f A_f.
$$

Supply `dot_m`, with units of mass/time, rather than normal velocity, mass-flux density or an unsigned magnitude. **Face area and density are already included.** The operator multiplies it by the transported scalar; it does not multiply by area or density again.

For example, `dot_m = -0.02 kg/s` and incoming `phi = 300` give a convective inflow RHS contribution of `+6` in the corresponding integrated equation units. Temperature examples below illustrate the scalar interface; the application remains responsible for the thermophysical formulation of its temperature/energy equation.

| Signed `dot_m` | Inner-face donor | Physical-boundary behaviour |
| --- | --- | --- |
| `> 0` | Owner | Outflow: reconstruct from the owned cell |
| `< 0` | Neighbour, possibly an MPI ghost | Inflow: obtain the scalar from the selected inflow policy |
| `== 0` | No donor is read | No scalar, inflow value or reconstruction increment is evaluated |

These are exact sign/zero tests; there is no built-in small-flow cutoff. Nonfinite sampled mass flow rates are diagnosed. In algebraic assembly, a nonzero rate in the operation's `Real` must also remain finite and nonzero after conversion to the matrix coefficient type.

### Array provider

```cpp
const auto mass_flux = cv::FaceMassFlux{face_mass_flow};
```

`face_mass_flow` is an accessible rank-one Kokkos View with extent at least the local `n_faces`. Strided and const views are supported. Entry `f` corresponds to local face `f`, including inner and physical boundary faces. The provider reads existing storage and allocates nothing.

At an MPI interface, both ranks must describe the same physical flux with their respective canonical normal signs. With owner-outward interface normals on both partitions, the two stored signed rates are opposite. This consistency is supplied by the application; the operator performs no flux exchange or reconciliation.

### On-demand provider

A custom device-callable `flux(context)` can read application fields or evaluate an already chosen mass-flow law on demand. Both context types expose local `face`, `owner`, face-centroid `x,y,z`, unit-normal `nx,ny,nz`, `area` and the operation's `time`. The inner context additionally has `neighbour`; the physical-boundary context has boundary indices and boundary geometry.

For example, define this provider at namespace scope for a prescribed uniform velocity/density test:

```cpp
struct PrescribedMassFlow {
    Real density = 1, ux = 1, uy = 0, uz = 0;

    template<class Context>
    KOKKOS_INLINE_FUNCTION Real operator()(const Context& c) const
    {
        return density*(ux*c.nx + uy*c.ny + uz*c.nz)*c.area;
    }
};
```

The application chooses this law; the convection operator still receives a mass-flow provider. Providers must be pure and give consistent answers when evaluated from multiple rows or ranks.

`cv::UniformMassFlux<Real>{m}` returns the same signed number for every canonical face. It is useful for controlled tests; it does not represent a uniform Cartesian velocity on arbitrary face normals and does not enforce mass conservation.

Density variations enter through the supplied mass rates and the rest of the equation assembled by the application. The operator does not assume `sum(dot_m_outward) == 0`. For a constant scalar it returns

$$
r_i^{conv}=-\phi\sum_{f\in\partial i}\dot m_{if}.
$$

In a compressible transient formulation, the time and continuity terms must be discretized consistently by the caller.

## Boundary conditions and inflow selection

The sign of the physical boundary mass flow determines whether a face is an inlet or outlet **on each call and each face**. Patch names and scalar condition types do not override this sign.

For nonzero physical-boundary flow, the scalar trace is

$$
\phi_f=
\begin{cases}
\phi_{in}(c,\mathrm{condition}), & \dot m_f<0,\\
\phi_P+\delta\phi_{P,f}, & \dot m_f>0.
\end{cases}
$$

The outflow increment is zero with `NoReconstruction`. An incoming physical face never evaluates the interior reconstruction. An outgoing face never evaluates the inflow policy.

Incoming data are selected as follows:

| Configuration | Value used on an incoming face |
| --- | --- |
| No explicit inflow argument | `FixedValueInflow`: the scalar `FixedValue::value` |
| Custom inflow functor passed directly | `inflow(context, typed_condition)` |
| Profile registry with this patch bound | The compact patch profile entry |
| Profile registry with this patch unbound | Its fallback, which defaults to `FixedValueInflow` |

`FixedValueInflow` accepts scalar `FixedValue` conditions. For other condition types it produces invalid input, reported as `missing or nonfinite inflow value` by `AssemblyStatus::check()`, if the face actually has incoming flow.

Consequences for the built-in scalar conditions:

| Scalar condition | Incoming convection without a profile/custom policy | Outgoing convection |
| --- | --- | --- |
| `FixedValue` | Uses its prescribed value | Uses the interior trace; this prescribed value is not imposed on the convective outflow |
| `NormalGradient` | Requires separate incoming scalar data | Uses the interior trace |
| `FixedFlux` | Requires separate incoming scalar data | Uses the interior trace |

`FixedFlux` is a physical scalar-flux condition interpreted by a supplied flux law elsewhere, for example in diffusion. It does not prescribe the fluid mass rate or directly replace `dot_m*phi` in this convection module.

An outlet with local backflow becomes an incoming face and needs incoming scalar data. Provide a profile or fallback for such faces; the operator does not clamp the mass flow, reverse it, or silently extrapolate an unspecified external scalar.

**Profiles affect convection only.** They do not modify scalar conditions, fill the extended field, change diffusive boundary fluxes, or update boundary data used when computing gradients/limiters. Configure consistent data separately for those operations. Convection itself does not need a preceding `boundary.apply_*()` call because its standard inflow path reads conditions/profiles directly.

## Basic setup and additive assembly

The following function-body example assumes ready `host_mesh`, compute `mesh` storage, `connectivity` storage, `comm` and `exec`. They use the same local numbering and remain alive. `face_mass_flow` is already populated in accessible memory, with writes ordered before convection on `exec`.

For this example there are two labelled patches, IDs `0` and `1`, and no unlabelled faces. Patch `1` uses an outflow-style scalar condition; add incoming data if backflow is possible.

```cpp
#include <krona/transport/scalar/convection/operator.hpp>
#include <krona/transport/scalar/assembly.hpp>
#include <krona/linalg/solver.hpp>

cv::ConvectionPlan<Exec, Real> plan(
    host_mesh, mesh.view(), connectivity.view(), comm, exec);
cv::ConvectionOperator<Exec, Real> convection(plan);

bc::Manager<Exec, Real> boundary(host_mesh, mesh.view(), exec);
boundary.set(0, bc::FixedValue<Real>{300});
boundary.set(1, bc::NormalGradient<Real>{0});

linalg::FaceAssemblyPlan<Exec> assembly(host_mesh, comm, exec);
linalg::CsrMatrix<Scalar, Exec> matrix(assembly.pattern());
linalg::Vector<Scalar, Exec> rhs(matrix.distribution());
ts::AssemblyStatus<Exec> status(exec);
const auto mass_flux = cv::FaceMassFlux{face_mass_flow};

auto target = ts::make_system_view(matrix, rhs, assembly, exec);
Kokkos::deep_copy(exec, target.values, Scalar{0});
Kokkos::deep_copy(exec, target.rhs, Scalar{0});

status.begin_async();
convection.add_system_async(target, boundary, status, mass_flux);
// Other compatible contributions may be added here, before the shared check.
(void)status.check(comm);
matrix.commit();
rhs.commit();
```

First-order assembly needs no scalar-field or gradient input. `make_system_view()` opens native read/write access; it does not clear the system. `add_system_async()` preserves all existing coefficients/RHS entries and adds its own contribution. Initialize once per complete assembly batch, rather than once per operator.

The graph and face-entry maps must describe the same local cells and faces as the spatial plan. Multiple faces connecting the same pair of cells accumulate into the shared graph entry. Inner rows are assembled by one thread per owned row; boundary contributions use atomics when faces share an owner.

Matrix/vector constructors and graph setup have their own blocking/collective contracts. Native access opened by `make_system_view()` may synchronize host changes. The no-fence contribution contract starts after a compatible system view has been obtained.

### BSR component slices

The same operator can add one scalar equation to a selected diagonal component of a block matrix:

```cpp
#include <krona/linalg/bsr_matrix.hpp>

linalg::BsrMatrix<Scalar, Exec> block_matrix(assembly.pattern(), 5);
linalg::Vector<Scalar, Exec> block_rhs(block_matrix.distribution());

Kokkos::deep_copy(exec, block_matrix.overwrite_blocks(exec), Scalar{0});
Kokkos::deep_copy(exec, block_rhs.overwrite(exec), Scalar{0});
auto component_target = ts::make_system_view(
    block_matrix, block_rhs, assembly, exec, 4);

status.begin_async();
convection.add_system_async(component_target, boundary, status, mass_flux);
(void)status.check(comm);
block_matrix.commit();
block_rhs.commit();
```

The matrix retains the compact cell/block graph. Its `distribution()` already describes expanded scalar DOFs and is the distribution to use for its vectors. The selected target addresses `block(entry,4,4)` and RHS slot `cell*5+4`; other block components are preserved. This adds no cross-variable derivatives or coupling blocks.

## Compact profiles on inlet patches

### Allocate and populate one local patch profile

A profile contains **exactly one scalar per local face of its patch**, in the same order as the patch interval. Its size is not `n_faces`, the total number of boundary faces, or the global patch size.

For patch-local index `k`, the corresponding local mesh face is

$$
f=\mathrm{range.begin}+k,\qquad
\phi_{in}(f)=\mathrm{profile}(k).
$$

For example, in a function using the objects above:

```cpp
const mesh::PatchId inlet_patch_id = 0;
const auto inlet_range = boundary.catalog().patches()[inlet_patch_id].range;
const auto inlet_count =
    static_cast<std::size_t>(inlet_range.end-inlet_range.begin);

Kokkos::View<Real*, Space> inlet_profile(
    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "inlet/profile"),
    inlet_count);

const auto face_y = mesh.view().geometry.face_centroid_y;
const auto first_face = static_cast<std::size_t>(inlet_range.begin);
using ProfilePolicy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
Kokkos::parallel_for("inlet/fill_profile", ProfilePolicy(exec, 0, inlet_count),
    KOKKOS_LAMBDA(std::size_t k) {
        const Real y = static_cast<Real>(face_y(first_face+k));
        inlet_profile(k) = Real{300} + Real{20}*Kokkos::sin(y);
    });

auto inflow = cv::make_inflow_profiles(boundary);
inflow.set(inlet_patch_id, inlet_profile);
```

This illustrative profile uses a nondimensional coordinate in `sin(y)`. The profile kernel and subsequent contributions use the same `exec`, so no intervening fence is required. `set()` reads only view metadata; it does not read or copy the device values.

Alternatively, if an application has already loaded `host_values` with exactly `inlet_count` entries, populate the device profile through an independent host mirror before submission:

```cpp
auto host_profile = Kokkos::create_mirror(Kokkos::WithoutInitializing, inlet_profile);
for (std::size_t k = 0; k < inlet_count; ++k)
    host_profile(k) = static_cast<Real>(host_values[k]);
Kokkos::deep_copy(exec, inlet_profile, host_profile);
```

Keep `host_profile` alive until the copy completes, for example until the later status check on the same instance. If releasing the host buffer earlier, first complete that copy explicitly. The registry does not perform this host-to-device transfer for you.

Use the registry in a fresh initialized assembly batch:

```cpp
auto profiled_target = ts::make_system_view(matrix, rhs, assembly, exec);
Kokkos::deep_copy(exec, profiled_target.values, Scalar{0});
Kokkos::deep_copy(exec, profiled_target.rhs, Scalar{0});
status.begin_async();
convection.add_system_async(profiled_target, boundary, status, mass_flux,
    cv::NoReconstruction{}, inflow);
(void)status.check(comm);
matrix.commit();
rhs.commit();
```

The profile overrides the scalar condition's incoming convection value on this patch, including when that condition is `NormalGradient` or `FixedFlux`. It is ignored on outgoing or zero-flow faces of the same patch. Every face still has a profile slot even if it is currently outgoing, so a later flow reversal can use the same storage.

### Several patches, storage types and strides

Register any number of labelled patch profiles with the same registry:

```cpp
inflow.set(inlet_patch_id, inlet_profile);
inflow.set(other_inlet_patch_id, other_inlet_profile);
```

Each view is checked against its own local patch length. Views may independently store `float` or `double`; values are converted to the operation's `Real` when sampled. Views must be rank one, have valid storage/stride for nonempty patches, and be accessible from `Exec`.

A component of an existing patch-sized field is also accepted:

```cpp
inflow.set(inlet_patch_id, patch_fields.view().component(temperature_component));
```

The field must have the local patch size. Its component may be strided, for example with `LayoutRight`; the registry preserves the stride without copying the component into a contiguous array. Do not pass the whole `FieldView` wrapper where a rank-one Kokkos View is expected.

On a rank where a labelled patch has no local faces, register a zero-length view. Patch IDs refer to the manager's catalog, while lengths and ordering are local to each rank. The registry does not distribute a global profile or exchange profile data between ranks.

For the unlabelled boundary interval, use:

```cpp
inflow.set_unlabelled(unlabelled_profile);
```

Its size must equal `catalog().unlabelled().end - catalog().unlabelled().begin`. This registration requires the catalog to report unlabelled boundaries; locally empty intervals are accepted when the patch exists globally. The boundary manager still needs an explicit unlabelled scalar condition.

### Fallback, updates and lifetime

Unbound patches use `FixedValueInflow` by default. A custom fallback is described below. Restore fallback handling with:

```cpp
inflow.clear(inlet_patch_id);
inflow.clear_unlabelled();
```

Modifying the contents of a registered allocation takes effect on subsequent operations without calling `set()` again. Replacing/reallocating the buffer requires another `set()` with its new view. `set()` can also replace a binding with a different supported scalar type or stride.

Creation allocates small host metadata once. Binding, clearing and selecting a profile do not allocate device buffers, copy values or synchronize. Selection occurs in the existing host `for_each_patch` dispatch; only a small pointer/stride/start descriptor is copied into the selected kernel. There is no per-face patch search or profile-kind branch on the device.

The registry is movable and noncopyable and is passed to contributions by reference. It is bound to the **exact boundary catalog** used to create it; passing it with a different manager/catalog is rejected, even if that manager happens to describe equal ranges. Construct another registry when replacing the manager.

The registry does not own profile allocations. Retain the original View/field owner until all kernels reading it complete. In particular, registering a temporary owning allocation does not keep that allocation alive. Clearing or rebinding metadata does not wait for older kernels: their old backing buffers must remain alive until those kernels finish.

### Local index reference

| Index | Meaning |
| --- | --- |
| `c.face` | Local face index in the complete mesh-face array |
| `c.boundary_index` | `c.face - n_inner_faces`, indexing all local physical boundary faces |
| `c.extended_index` | `n_cells + c.boundary_index`, indexing an extended cell field |
| Profile index | `c.face - range.begin`, indexing only this patch |

`FaceMassFlux` intentionally uses the first indexing scheme because rates are needed on inner and boundary faces. `InflowProfiles` uses the last scheme for compact boundary-only data.

## Analytic inflow and custom fallback

To avoid a profile array entirely, supply a pure device-callable functor. Define functors with member templates at namespace scope:

```cpp
struct AnalyticInflow {
    template<class R, class Condition>
    KOKKOS_INLINE_FUNCTION R operator()(
        const boundary::FaceContext<R>& c, const Condition&) const
    {
        return R{300} + R{20}*Kokkos::sin(c.y)*Kokkos::cos(c.time);
    }
};
```

Here coordinates/time are illustrative nondimensional inputs. A real model can carry its own physical scales and parameters. The condition argument is already a concrete condition type selected by the boundary manager, rather than its host-side `std::variant`.

Pass the functor directly for all incoming faces:

```cpp
// Inside an initialized assembly batch:
convection.add_system_async(target, boundary, status, mass_flux,
    cv::NoReconstruction{}, AnalyticInflow{}, time);
```

Or use it only where there is no registered patch profile:

```cpp
auto mixed_inflow = cv::make_inflow_profiles(boundary, AnalyticInflow{});
mixed_inflow.set(inlet_patch_id, inlet_profile);
```

Registered patches use their arrays; other incoming patches use the analytic function. Both are ignored on outflow. Pass the operation's `time` after the reconstruction and inflow arguments; array profiles themselves are not automatically recomputed as time changes.

A custom policy can instead read `field(c.extended_index)` from a prepared extended field. The caller then guarantees its size, memory accessibility, current values and lifetime. The operator treats every returned incoming value as prescribed data for the current assembly; it does not differentiate owner/state dependence inside a custom inflow function.

## Discretization and reconstruction

For an owned cell `i`, let `F_if` be the face mass rate directed out of that cell. It equals the stored canonical rate for the owner and its negative for the neighbour. The interior upwind matrix contribution is

$$
\begin{cases}
A_{ii}\mathrel{+}=F_{if}, & F_{if}>0,\\
A_{ij}\mathrel{+}=F_{if}, & F_{if}<0.
\end{cases}
$$

Thus incoming off-diagonal coefficients are negative. With reconstructed donor trace `phi_d + delta_phi_d,f`, the explicit increment contributes

$$
b_i\mathrel{-}=F_{if}\,\delta\phi_{d,f}.
$$

At a physical boundary:

| Flow | Matrix/RHS addition |
| --- | --- |
| Incoming | `b_owner += -dot_m * phi_in` |
| Outgoing | `A_owner,owner += dot_m`; `b_owner += -dot_m * delta_phi_out` |
| Zero | No flux contribution |

The matrix remains compact donor-cell upwind for every reconstruction. Mass rates and reconstruction increments are frozen in this linearization; no derivatives of their possible dependence on unknowns are assembled.

The operator's `Real` defaults to `ComputeReal`. Arithmetic uses that type before conversion to the target coefficient/RHS/output types, normally `SolverReal` for an algebraic system. Profile conversion happens when sampling; the operator builds no converted profile or coefficient arrays. Finite precision loss and reduction order can produce rounding differences between algebraic and matrix-free evaluation or between backends.

### Connect the existing reconstruction module

The adapter separates numerical `Scheme`, borrowed `Data` and geometric/sample `Stencil`. `make_reconstruction(plan, scheme, data, component)` supplies the existing allocation-free `rec::MeshStencil` using the plan's mesh/connectivity.

For example, assume `phi`, `gradients` and `factors` are already populated field owners. For one scalar, their component counts are `1`, `3` and `1`; the gradient order is `x,y,z`:

```cpp
auto reconstruction_data = rec::data(
    rec::field_pack(phi.view()),
    rec::gradient_pack(gradients.view()),
    rec::field_pack(factors.view()));

auto reconstruction = cv::make_reconstruction(
    plan, rec::LimitedLinear{}, reconstruction_data, 0);
```

The zero selects a component from the reconstruction data batch. For multiple scalars, gradients use `3*component + axis`; a factor pack may have one shared component or one per scalar. One convection call still adds one selected scalar equation. Selecting a BSR target component is a separate operation in `make_system_view()`.

Typical policies are:

| Reconstruction | Required data beyond scalar values |
| --- | --- |
| `cv::NoReconstruction` | None; default first-order path, no reconstruction data object |
| `rec::FirstOrder` through the adapter | No derivatives; zero increment |
| `rec::SecondOrderUpwind` | Cell gradients |
| `rec::LimitedLinear` | Cell gradients and factors; omitted factors default to one |
| `rec::Tvd` | Gradients and the opposite real-cell value; ratio limiter defaults to Van Leer |
| `rec::MusclKappa` | Gradients and the opposite real-cell value; this policy is unlimited |
| `rec::Limited<Scheme>` | The wrapped scheme's inputs plus cell factors |
| `rec::Quadratic` / `LimitedQuadratic` | Gradients, Hessians and central cell moments; limited form also uses factors |
| `rec::Quick` | Explicit three-sample stencils and weights |

Convection does not construct gradients, factors, Hessians or moments. Factor computation belongs to the reconstruction module/application. Required owned and MPI-ghost inputs must be current before submission. Factors for limited schemes must satisfy the numerical reconstruction contract, including their valid range.

Scalar values used by the reconstruction data must correspond to the field/component supplied to residual evaluation and to the current deferred-correction iterate. The adapter returns an **increment**, not an already reconstructed full value.

### Outflow reconstruction

There is no physical boundary ghost cell supplying an opposite exterior value. With the mesh-based factory:

- Schemes without `OtherValues` use the same polynomial at outflow.
- TVD/MUSCL and other schemes declaring `OtherValues` default to one-sided `rec::LimitedLinear` at outflow. It uses the data's cell factor, or factor one when omitted.
- An explicit outflow scheme may be supplied, but it must not require `OtherValues`.

For example, use TVD internally and owner-value extrapolation at outflow:

```cpp
auto tvd_with_first_order_outflow = cv::make_reconstruction(
    plan, rec::Tvd{}, reconstruction_data, rec::FirstOrder{}, 0);
```

An outflow `FixedValue` or extended boundary slot is not used to fabricate the missing opposite sample. A line-based TVD limiter alone does not prove multidimensional boundedness, especially with transverse skew correction or an unlimited one-sided outflow trace.

For QUICK, provide an explicit stencil object:

```cpp
auto quick_reconstruction = cv::make_reconstruction(
    rec::Quick{}, quick_data, quick_stencil, component);
```

It supplies `face(f)` with left/right `ThreePointStencil` sides and `boundary_side(f)` with an explicit outflow stencil. The mesh-based factory does not find distant upstream samples automatically. The caller validates stencil geometry, weights, sample indices and any extra halo depth.

A custom convection reconstruction can also supply `interior(context, from_owner)` and `boundary(context)` directly, both returning increments convertible to `Real`. Only the selected donor side or an actual outflow side is evaluated.

## Matrix-free residual

The same mass-flow, reconstruction and inflow policies can be used without allocating an algebra graph or matrix:

```cpp
Kokkos::View<Real*, Space> residual(
    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "scalar/residual"),
    static_cast<std::size_t>(mesh.view().sizes.n_own));
Kokkos::deep_copy(exec, residual, Real{0});

status.begin_async();
convection.add_residual_async(phi.view().component(0), residual,
    boundary, status, mass_flux, reconstruction, inflow, time);
(void)status.check(comm);
```

The convection contribution is `-sum(F_if * phi_face)`, consistent with `b-A*phi` for the assembled operator and the same frozen data. Positive outgoing transport decreases the residual. Other residual contributions may be added in the same batch before checking status.

The input is an accessible rank-one View covering at least `n_cells`, including current real MPI ghosts. The writable output covers at least `n_own`; only that prefix is updated. Both can be strided. Input and output storage must not overlap, including partial/strided aliases; reconstruction/profile/flux inputs must also remain disjoint from written targets. The operator detects identical base pointers for direct in-place residual use, not every possible alias.

Default reconstruction/inflow processing does not read the extended physical-boundary tail. External gradient or limiter preparation may need those boundary samples; allocate and populate them according to those tools' contracts.

## Deferred correction with a frozen matrix

`add_correction_async()` adds only the difference between reconstructed and first-order convection RHS contributions. It updates inner faces and physical outflows. It does not add the upwind matrix, prescribed inflow terms, sources or any other base term again, and it takes no inflow argument.

For a scalar CSR system, first assemble and commit the complete **first-order base system**, using `NoReconstruction` and the intended inflow policy. Save its RHS:

```cpp
Kokkos::View<Scalar*, Space> base_rhs(
    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "scalar/base_rhs"),
    matrix.distribution().local_size());
{
    const auto b = rhs.read(exec);
    Kokkos::deep_copy(exec, base_rhs, Kokkos::subview(b, Kokkos::ALL(), 0));
    rhs.commit();
}
```

After updating the iterate, its gradients/factors and all required halos, form each corrected RHS from that saved base:

```cpp
{
    const auto b = Kokkos::subview(rhs.overwrite(exec), Kokkos::ALL(), 0);
    Kokkos::deep_copy(exec, b, base_rhs);

    status.begin_async();
    convection.add_correction_async(
        b, boundary, status, mass_flux, reconstruction, time);
    (void)status.check(comm);
    rhs.commit();
}
// The committed matrix and corrected RHS are now ready for the linear solver.
```

Restore the base before every update: repeatedly adding corrections to the previous corrected RHS accumulates them incorrectly. The application controls iteration and convergence; a single solve with a frozen explicit correction is not generally the converged higher-order equation.

The correction-only call does not touch matrix coefficients or revision, so the algebra layer can reuse a prepared preconditioner for the unchanged matrix. Changing mass rates, including rates changed through a time-dependent provider, requires updating the upwind matrix. Changing prescribed inflow/profile values changes the base RHS: correction-only updates do not refresh those terms. Reassemble the base or update it explicitly in application code.

## Composition with diffusion and sources

Convection and diffusion use the same integrated `r=b-A*phi` convention and can share one algebra target and status batch. For example, with prepared diffusion objects/data and a freshly initialized target:

```cpp
ts::SourceOperator<Exec, Real> source_operator(mesh.view(), exec);

status.begin_async();
diffusion.add_system_async(target, boundary, status,
    diffusion_law, diffusion_correction, time);
convection.add_system_async(target, boundary, status,
    mass_flux, reconstruction, inflow, time);
source_operator.add_async(target.rhs, status, ts::UniformSource<Real>{1});
(void)status.check(comm);
matrix.commit();
rhs.commit();
```

The source argument is a volume density, so `SourceOperator` adds `source_i * V_i`. The convection operator itself adds no source or time derivative. All contributing plans, entry maps, views and boundary geometry must describe the same local mesh numbering. Apply matching contributions when forming a matrix-free total residual.

## Asynchronous execution and diagnostics

The three contribution methods enqueue work on the plan's execution instance. They perform no work-array allocation, halo exchange, resets, fence or algebra commit. `InflowProfiles` remains on the host; the selected device descriptor is captured by value before the method returns. Actual overlap depends on the backend; a no-fence submission API does not imply that a CPU backend executes its kernels asynchronously.

Use this dependency order:

1. Prepare mass rates, incoming data and the current reconstruction inputs.
2. Complete required halo transfers/unpacking, or order unpack and reads on the same execution instance.
3. Open and initialize the target once; call `status.begin_async()` once for the complete batch.
4. Add all compatible contributions on that execution instance.
5. Call collective `status.check(comm)` on every rank, including empty ranks.
6. Commit each accessed algebra wrapper before native matrix operations or a solve.

Same-instance kernels and copies are ordered without a fence between each contribution. Data produced on a different stream/execution instance needs an explicit dependency or completion before it is read. Calling an unsplit convection operation while required ghost values are still in flight is not a valid overlap strategy.

| Operation | Completion/allocation behaviour |
| --- | --- |
| `ConvectionPlan` construction | Blocking local mesh/connectivity validation |
| `make_reconstruction(...)` | Borrows input/stencil descriptors; no gradient or work-array construction |
| `make_inflow_profiles(...)` | Allocates small host metadata; no device data transfer |
| Profile `set` / `clear` | Host metadata changes; no fence or value copy |
| `AssemblyStatus::begin_async()` | Enqueues flag reset; does not reset matrix/RHS/residual |
| `add_system_async`, `add_residual_async`, `add_correction_async` | Local kernel submission; no MPI, fence, reset or commit |
| `AssemblyStatus::check(comm)` | Copies diagnostics to host, fences its instance and performs `MPI_Allreduce`; does not commit algebra wrappers |
| Matrix/vector `commit()` | Completes the wrapper's pending access batch |

The status object must use the same execution instance as the operator. Do not restart it while a batch is pending. Profiles, flux data, fields and reconstruction inputs must remain valid until work completes. Exception paths must also drain submitted users before releasing their buffers; a pending status destructor fences its execution instance as a cleanup boundary. Arrange destruction order so the borrowed owners are still alive when such cleanup runs.

Setup and host-side checks reject incompatible meshes/catalogs, invalid time, incorrect known view extents/components, wrong patch IDs, and invalid profile registration. Kernel diagnostics report nonfinite mass rates, missing/nonfinite sampled inflow data, nonfinite reconstruction increments and nonfinite sums/conversions. Algebraic assembly also rejects nonzero mass rates that become zero in the coefficient type. Ordinary finite rounding, including loss of small RHS increments, is not exhaustively diagnosed. The shared status preserves errors from other operators in the same batch.

Checks do not prove conservation between separately supplied MPI interface rates, validate arbitrary custom-provider storage, establish physical positivity/boundedness, or verify full custom stencil indices. Errors follow KRONA's `mpi::require`/fatal policy rather than returning a recoverable validation result. Failed contributions are not transactional: preceding device writes are not rolled back.

## Current limits and possible extensions

- Plans and profile ranges describe a static mesh. Complete users before changing/reallocating borrowed data. Geometry changes require refreshed geometry and plan validation; topology/numbering changes also invalidate algebra maps, stencils and patch bindings.
- The implicit operator is upwind with frozen mass rates. Higher-order, limiter, custom-inflow and mass-rate derivatives are not assembled into a Newton Jacobian.
- Gradients, limiters, moments and extended QUICK stencils are external inputs. The selected reconstruction's accuracy and boundedness requirements remain the caller's responsibility.
- There is no built-in convection equation solver, time integrator, profile-file reader, or automatic profile registration from TOML. Scalar condition parsing remains the boundary module's responsibility.
- Profile bindings currently support borrowed rank-one float/double arrays and one common fallback policy. A separate registry is required for another boundary catalog.
- A single call processes one scalar component. Several scalars can share plans/maps and use a reconstruction data batch, but fused multi-equation transport assembly is not implemented.

Potential additions include a dedicated prescribed-inflow RHS refresh for unchanged matrices, typed analytic policies bound per patch, fused transport of several scalars, and separate interior/interface submission for halo overlap. Moving-mesh transport would also need relative mass rates and consistent volume/time/geometric-conservation treatment; those are future integration tasks rather than current capabilities.
