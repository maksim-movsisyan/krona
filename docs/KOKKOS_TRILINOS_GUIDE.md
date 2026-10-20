# Kokkos and Trilinos for KRONA Developers

This guide introduces the Kokkos and Trilinos concepts needed to read and extend KRONA.

The intended reading order is:

1. Understand memory, views, and ownership.
2. Learn how kernels execute and how dependencies are ordered.
3. Understand distributed vectors and sparse matrices.
4. Learn how Belos and preconditioners use these objects.
5. Read the corresponding KRONA module documentation.

The examples use the Kokkos-based Tpetra stack discussed for KRONA: Kokkos 5.x and Trilinos 17.x. The installed headers remain authoritative for version-specific interfaces.

Most examples are fragments to be used inside an initialized application. Section 23 contains a complete native Tpetra/Belos example.

The examples were checked against official API references but were not compiled while preparing this document.

## 1. The responsibilities of each library

The libraries provide different layers of functionality.

| Component | Main responsibility |
|---|---|
| MPI | Communication between processes |
| Kokkos Core | Portable memory abstractions and parallel execution within a process |
| Kokkos Kernels | Numerical kernels, including dense and sparse linear algebra |
| Teuchos | Utilities used throughout Trilinos: smart pointers, parameter lists, communicators |
| Tpetra | Distributed maps, vectors, sparse graphs, matrices, and operators |
| Belos | Iterative linear solvers |
| Ifpack2 | Relaxation, incomplete factorization, and domain decomposition preconditioners |
| MueLu | Algebraic multigrid solvers and preconditioners |
| KRONA | Mesh, fields, numerical operators, physical models, and application-level contracts |

A typical calculation combines these layers:

- KRONA assembles coefficients in Kokkos kernels.
- Tpetra stores the distributed matrix and vectors.
- Belos performs iterations.
- The selected preconditioner accelerates convergence.
- Tpetra and MPI exchange data required by distributed linear algebra.

Tpetra uses both MPI and Kokkos: MPI distributes the problem, while Kokkos supplies execution and memory abstractions inside each process.

### 1.1 Host and device

In this guide:

- **Host** means the CPU-side application code.
- **Device** means the execution and memory resource selected for computation.

In a Serial or OpenMP build, the computation normally runs on the CPU. In a CUDA or HIP build, it normally runs on a GPU.

“Device” does not necessarily imply “GPU” in every API name. For example, `getLocalViewDevice()` returns the view associated with Tpetra’s computational memory space, which can be host memory in a CPU build.

## 2. Execution spaces and memory spaces

Kokkos separates two questions:

1. **Where does the code execute?**
2. **Where is the data stored?**

An execution space answers the first question. A memory space answers the second.

```cpp
using Exec = Kokkos::DefaultExecutionSpace;
using Mem = typename Exec::memory_space;

Exec exec;
```

Here:

- `Exec` is a type.
- `Mem` is a type.
- `exec` is an execution space instance.

Typical combinations include:

| Execution space | Typical default memory space |
|---|---|
| `Kokkos::Serial` | `Kokkos::HostSpace` |
| `Kokkos::OpenMP` | `Kokkos::HostSpace` |
| `Kokkos::Cuda` | `Kokkos::CudaSpace` |
| `Kokkos::HIP` | `Kokkos::HIPSpace` |

Additional memory spaces support shared or pinned memory. They have different accessibility and performance properties.

The default memory space of an execution space is accessible from that execution space.

### 2.1 A type does not identify a unique execution queue

Two objects of the same execution space type do not necessarily represent independent queues. Conversely, separately configured instances can represent different queues.

Therefore:

```cpp
Exec exec;
```

should not be interpreted as “create a new independent GPU stream.”

For ordered asynchronous work, pass the intended instance explicitly:

```cpp
Kokkos::parallel_for(
    "operation",
    Kokkos::RangePolicy<Exec>(exec, 0, count),
    functor);
```

KRONA uses an execution-space alias selected by its build configuration. Prefer that alias when extending project code.

### 2.2 Accessibility is a contract

A view can contain a device pointer while the view object itself exists on the host.

This is valid:

```cpp
Kokkos::View<double*, Mem> values("values", count);
```

It does not imply that the CPU can dereference `values.data()`.

Generic code can check execution-space accessibility:

```cpp
template<class Exec, class View>
void check_accessibility()
{
    static_assert(
        Kokkos::SpaceAccessibility<
            Exec,
            typename View::memory_space>::accessible);
}
```

Accessibility and synchronization are separate requirements. Accessible memory can still contain data whose producer has not finished.

## 3. Initialization and lifetime

Kokkos must be initialized before allocating views or launching kernels.

A standalone Kokkos application can use:

```cpp
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard runtime(argc, argv);

    {
        // Allocate views and execute kernels here.
    }

    // Views have been destroyed before runtime finalization.
}
```

The inner scope makes destruction order explicit. Allocations must be released before Kokkos is finalized.

A standalone Tpetra application can instead use:

```cpp
Tpetra::ScopeGuard runtime(&argc, &argv);
```

This initializes MPI and/or Kokkos when needed and finalizes the resources it initialized.

Inside KRONA, follow KRONA’s runtime initialization path. Initialization ownership belongs at the application boundary, not inside numerical objects.

## 4. What a Kokkos::View represents

A view is an array descriptor. Conceptually, it contains:

- A handle to storage.
- Array extents.
- An index-to-address mapping.
- Memory-space and access properties.
- Allocation tracking when applicable.

The descriptor and the array storage are different objects.

```cpp
using Buffer = Kokkos::View<double*, Mem>;

Buffer empty;
Buffer allocated("allocated", count);
```

The default construction creates an empty descriptor. The allocating construction creates storage and a descriptor referring to it.

### 4.1 Reading the type

```cpp
Kokkos::View<double*, Mem>
```

means a one-dimensional array of `double`.

```cpp
Kokkos::View<double**, Kokkos::LayoutLeft, Mem>
```

means a two-dimensional array with two runtime extents.

```cpp
Kokkos::View<double*[3], Kokkos::LayoutLeft, Mem>
```

means a two-dimensional array whose second extent is fixed at three:

```cpp
Kokkos::View<double*[3], Kokkos::LayoutLeft, Mem>
    velocity("velocity", cells);

velocity(cell, component);
```

A view’s **rank** is its number of dimensions. It is unrelated to an MPI rank.

The scalar type, dimensions, layout, memory space, and memory traits are part of the view’s type.

### 4.2 A type alias is still a type

```cpp
using Buffer = Kokkos::View<double*, Mem>;

Buffer values("values", count);
```

This is ordinary C++ type aliasing.

Likewise:

```cpp
values = Buffer(
    Kokkos::view_alloc(
        exec,
        Kokkos::WithoutInitializing,
        "values"),
    count);
```

constructs a temporary `Buffer` and assigns its descriptor to `values`. This syntax invokes a constructor; it does not call a static allocation method named `Buffer`.

## 5. Allocation and initialization

### 5.1 Standard allocation

```cpp
Buffer values("values", count);
```

For a view of arithmetic values, standard initialization produces zero-initialized elements.

The ordinary allocating constructor performs initialization using the associated default execution instance and completes that initialization before returning.

### 5.2 Allocation with explicit properties

```cpp
Buffer values(
    Kokkos::view_alloc(exec, "values"),
    count);
```

`view_alloc()` creates a bundle of allocation properties. The view constructor performs the allocation.

Passing an execution instance associates initialization with that instance and avoids the constructor’s ordinary initialization fence.

```cpp
Buffer values(
    Kokkos::view_alloc(
        exec,
        Kokkos::WithoutInitializing,
        "values"),
    count);
```

`WithoutInitializing` skips element initialization. For numerical arrays, this is useful when the next operation overwrites every element.

It does not guarantee that the underlying memory allocator is nonblocking. Allocation and element initialization are separate operations.

### 5.3 Full overwrite means full overwrite

This is valid after allocating without initialization:

```cpp
using Policy =
    Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;

Kokkos::parallel_for(
    "initialize_values",
    Policy(exec, 0, count),
    KOKKOS_LAMBDA(const std::size_t i) {
        values(i) = 1.0;
    });
```

This is invalid unless `values(i)` was already initialized:

```cpp
values(i) += contribution;
```

The same rule applies to:

- Sparse matrix accumulation.
- Atomic additions.
- Partial field updates.
- Conditional writes that leave some entries untouched.

## 6. Managed views, borrowed views, and ownership

### 6.1 Managed allocation tracking

A view created through an allocating constructor participates in reference-counted allocation ownership.

```cpp
Buffer a("a", count);
Buffer b = a;
```

`a` and `b` refer to the same allocation.

This does not allocate or copy a second array. When the final owning reference is released, the allocation can be destroyed.

This resembles `std::shared_ptr`, with additional array metadata. Ordinary element access does not increment a reference count.

### 6.2 Copying a view does not copy its elements

```cpp
auto alias = a;
```

copies the descriptor.

```cpp
Kokkos::deep_copy(exec, destination, a);
```

copies elements into an already existing destination.

These operations have different purposes:

| Operation | Effect |
|---|---|
| `auto b = a` | Share the existing storage |
| `b = a` | Rebind `b` to the existing storage |
| `Buffer b("b", n)` | Allocate independent storage |
| `deep_copy(exec, b, a)` | Copy elements from `a` into `b` |

If `b` previously owned a different allocation, rebinding releases its reference to that previous allocation.

### 6.3 Explicitly unmanaged views

```cpp
using Borrowed = Kokkos::View<
    double*,
    Mem,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

Borrowed borrowed(a.data(), a.extent(0));
```

This wraps existing storage without taking ownership.

The object responsible for that storage must outlive every use of the borrowed view.

`Unmanaged` describes lifetime management. It does not mean:

- Host-accessible.
- Read-only.
- Contiguous.
- Synchronized.
- Safe to use after its owner is destroyed.

### 6.4 Raw-pointer construction does not discover an owner

An important distinction:

```cpp
Buffer shared = a;
```

shares allocation tracking.

```cpp
Buffer wrapped(a.data(), a.extent(0));
```

uses the raw-pointer wrapping constructor. It does not recover ownership information from `a.data()`.

A view type without an explicit `Unmanaged` trait is capable of managing allocations, but a particular instance constructed from a raw pointer can still be borrowed.

Similarly, converting a borrowed descriptor to another compatible view type does not create ownership of the pointed-to allocation.

For clarity, use an explicitly unmanaged type when the intended contract is borrowing.

### 6.5 Wrapping external host memory

```cpp
std::vector<double> storage(count);

using HostBorrowed = Kokkos::View<
    double*,
    Kokkos::HostSpace,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

HostBorrowed view(storage.data(), storage.size());
```

The view does not:

- Copy the vector.
- Extend its lifetime.
- Prevent reallocation.
- Transfer its data to a GPU.

Reallocation of `storage` invalidates the wrapped address. Wrapping external memory requires the caller to provide the correct address, extent, layout, memory-space declaration, and lifetime.

### 6.6 Ownership does not establish kernel completion

Keep storage alive until all operations using it have completed.

```cpp
Buffer output(
    Kokkos::view_alloc(
        exec,
        Kokkos::WithoutInitializing,
        "output"),
    count);

Kokkos::parallel_for(
    "produce_output",
    Policy(exec, 0, count),
    KOKKOS_LAMBDA(const std::size_t i) {
        output(i) = static_cast<double>(i);
    });

exec.fence("output no longer in use");
```

A captured descriptor is not a general substitute for an application-level lifetime contract. This matters especially for unmanaged views, external buffers, asynchronous copies, and MPI requests.

## 7. Constness and view aliases

These declarations express different contracts:

```cpp
const Kokkos::View<double*, Mem> handle = a;
Kokkos::View<const double*, Mem> read_only = a;
```

The first makes the **descriptor** const. The elements remain writable through it.

The second makes the **elements** read-only through that view.

A const-qualified handle remains useful inside a functor’s `const operator()`:

```cpp
struct Fill {
    Kokkos::View<double*, Mem> values;

    KOKKOS_INLINE_FUNCTION
    void operator()(std::size_t i) const
    {
        values(i) = 0.0;
    }
};
```

Read-only access and ownership are independent. A read-only view can share ownership of a managed allocation.

Useful native Kokkos aliases include:

| Alias | Meaning |
|---|---|
| `value_type` | Element type, possibly const-qualified |
| `non_const_value_type` | Element type with const removed |
| `memory_space` | Storage space |
| `execution_space` | Associated execution space |
| `array_layout` | Index mapping |
| `const_type` | Corresponding read-only view type |

A KRONA wrapper containing a Kokkos view does not automatically inherit these aliases. Generic code must use the wrapper’s declared interface.

For example, if a wrapper provides only `value_type`, this extracts its writable scalar type:

```cpp
using Value =
    std::remove_const_t<typename Field::value_type>;
```

## 8. Layouts, strides, and subviews

### 8.1 Layout describes addresses

For an array indexed as `field(entity, component)`:

- `LayoutLeft` makes the entity index contiguous.
- `LayoutRight` makes the component index contiguous.
- `LayoutStride` represents explicit strides.

For an unpadded two-dimensional array:

\[
\operatorname{offset}_{\text{Left}}(i,c)=i+N_{\text{entities}}c,
\]

\[
\operatorname{offset}_{\text{Right}}(i,c)=iN_{\text{components}}+c.
\]

Thus `LayoutLeft` gives component-wise streams of entities, commonly described as SoA-like storage. `LayoutRight` groups an entity’s components, commonly described as AoS-like storage.

These descriptions depend on the meaning of the indices. They are not universal synonyms for the layout names.

### 8.2 Choose a layout for the access pattern

If neighboring GPU threads process neighboring entities and read the same component, contiguous entities can improve memory access.

If one worker repeatedly accesses every component of an entity, contiguous components may be useful.

There is no universally optimal layout. The relevant question is which addresses neighboring workers access.

### 8.3 Subviews select existing storage

```cpp
auto component =
    Kokkos::subview(field, Kokkos::ALL(), component_index);
```

This creates a descriptor for a selected component. It does not allocate a new component array or copy its values.

The resulting view carries the correct mapping, potentially through `LayoutStride`.

```cpp
component(i);
```

therefore works for both layouts.

A subview preserves ownership tracking where applicable; a subview of borrowed storage remains borrowed.

### 8.4 .data() loses the mapping

For a native view:

```cpp
auto pointer = component.data();
```

returns the starting address of the viewed storage. It does not encode the stride.

Consequently:

```cpp
pointer[i]
```

is only equivalent to `component(i)` when the component stride is one.

For a regular AoS-style field with three components, a selected component typically has stride three:

\[
\text{address}(i)=\text{data}+3i.
\]

Use the view’s indexing operator unless a low-level interface explicitly requires a pointer and stride.

### 8.5 Extent, size, and span

These quantities answer different questions:

| Expression | Meaning |
|---|---|
| `view.extent(d)` | Logical length of dimension `d` |
| `view.size()` | Total logical element count |
| `view.stride(d)` | Address step for dimension `d`, in elements |
| `view.span()` | Address span covered by the mapping, including gaps or padding |
| `view.span_is_contiguous()` | Whether the mapping covers a contiguous span |

A strided component can contain `N` logical elements while spanning more than `N` storage locations.

`LayoutStride` can wrap such storage without repacking it.

## 9. Mirrors and explicit copies

A mirror provides a compatible representation in another memory space, usually for host access.

| Operation | Allocation and copying |
|---|---|
| `create_mirror(view)` | Allocate independent mirror storage; no data copy |
| `create_mirror_view(view)` | May reuse the original storage; no data copy |
| `create_mirror_view_and_copy(space, view)` | Obtain a suitable view and copy when needed |

Use `create_mirror()` when independence is required, including when the source already resides on the host.

Use `create_mirror_view()` when aliasing is acceptable and the goal is accessibility.

### 9.1 An explicit host snapshot

```cpp
auto host = Kokkos::create_mirror(
    Kokkos::WithoutInitializing,
    values);

Kokkos::deep_copy(exec, host, values);
exec.fence("host snapshot ready");

// Host access is now valid.
std::cout << host(0) << '\n';
```

The allocation, copy, and completion boundary are separate.

### 9.2 deep_copy does not allocate the destination

For view-to-view copying, destination storage must already exist with compatible extents.

```cpp
Buffer destination(
    Kokkos::view_alloc(
        exec,
        Kokkos::WithoutInitializing,
        "destination"),
    values.extent(0));

Kokkos::deep_copy(exec, destination, values);
```

For scalar-to-view copying:

```cpp
Kokkos::deep_copy(exec, destination, 0.0);
```

the scalar fills every logical element.

### 9.3 Layout and scalar conversion have conditions

`deep_copy` can perform element-wise scalar conversion and layout conversion when the selected execution space can access both arrays.

For a direct transfer between separate host and device spaces without a common execution space, compatible contiguous representations are required.

A reliable way to change both memory space and representation is:

1. Create a host mirror compatible with the device array.
2. Transfer between the compatible representations.
3. Perform layout or scalar conversion where both arrays are accessible.

A custom conversion kernel is useful when conversion also requires validation, scaling, packing, or finite-value checks.

## 10. Launching parallel kernels

A Kokkos kernel has:

- An iteration space.
- An execution policy.
- A callable executed for each work item.

### 10.1 RangePolicy

```cpp
using Policy =
    Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;

Kokkos::parallel_for(
    "scale",
    Policy(exec, 0, count),
    KOKKOS_LAMBDA(const std::size_t i) {
        values(i) *= 2.0;
    });
```

The range is half-open: `[0, count)`.

The label identifies the operation in profiling and diagnostics.

`IndexType<std::size_t>` selects the iteration index type explicitly. Choose a type that can represent the intended range.

The policy determines the execution space; the view’s memory space does not launch the kernel automatically.

### 10.2 Parallel iterations have no prescribed order

This loop has independent writes:

```cpp
output(i) = input(i) * input(i);
```

This generally does not:

```cpp
output(i) = output(i - 1) + input(i);
```

A kernel must remain correct for any legal execution order. A Serial backend can hide ordering bugs that become visible with OpenMP or a GPU.

### 10.3 Capture descriptors by value

`KOKKOS_LAMBDA` provides a device-compatible lambda with value capture.

```cpp
const auto input = input_view;
const auto output = output_view;
const double factor = 2.0;

Kokkos::parallel_for(
    "transform",
    Policy(exec, 0, count),
    KOKKOS_LAMBDA(const std::size_t i) {
        output(i) = factor * input(i);
    });
```

The captured view descriptors refer to existing arrays. Capturing them does not duplicate those arrays.

Inside a class, first extract the required views and scalar values. Capturing `this` can leave a device kernel with a pointer to a host object.

### 10.4 Functors express the same model

```cpp
template<class Input, class Output>
struct Square {
    Input input;
    Output output;

    KOKKOS_INLINE_FUNCTION
    void operator()(std::size_t i) const
    {
        output(i) = input(i) * input(i);
    }
};

Kokkos::parallel_for(
    "square",
    Policy(exec, 0, count),
    Square<decltype(input), decltype(output)>{
        input, output});
```

Functors are convenient for reusable kernels and explicit data dependencies.

Kernel-callable helper functions need suitable Kokkos annotations. Host-side allocation, MPI calls, and ordinary host-only objects do not become device-compatible through capture.

### 10.5 Multidimensional and hierarchical policies

| Policy | Typical purpose |
|---|---|
| `RangePolicy` | One independent iteration per entity |
| `MDRangePolicy` | A multidimensional iteration space |
| `TeamPolicy` | Cooperative work within teams |
| `TeamThreadRange` | A nested loop shared by a team |

For example:

```cpp
using Policy2D =
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>;

Kokkos::parallel_for(
    "fill_components",
    Policy2D(exec, {0, 0}, {cells, components}),
    KOKKOS_LAMBDA(const int i, const int c) {
        field(i, c) = 0.0;
    });
```

With `TeamPolicy`, multiple workers cooperate on one task. Scratch memory and team barriers apply within that team; a team barrier does not synchronize every team in the kernel.

Begin with entity-parallel kernels and introduce hierarchical execution when the workload benefits from cooperation.

## 11. Reductions, scans, and race-free accumulation

### 11.1 Reduction to a host scalar

```cpp
double sum = 0.0;

Kokkos::parallel_reduce(
    "sum",
    Policy(exec, 0, count),
    KOKKOS_LAMBDA(const std::size_t i, double& partial) {
        partial += values(i);
    },
    sum);
```

Kokkos combines partial results. The scalar result is available to the host when the call returns.

The incoming value of `partial` may already contain other contributions. Accumulate into it rather than overwriting it.

### 11.2 Reduction to a view

```cpp
Kokkos::View<double, Mem> result(
    Kokkos::view_alloc(
        exec,
        Kokkos::WithoutInitializing,
        "result"));

Kokkos::parallel_reduce(
    "sum_to_device",
    Policy(exec, 0, count),
    KOKKOS_LAMBDA(const std::size_t i, double& partial) {
        partial += values(i);
    },
    result);
```

A rank-zero view represents one scalar location.

This form can remain asynchronous and can feed a subsequent kernel on the same execution instance:

```cpp
Kokkos::parallel_for(
    "use_sum",
    Policy(exec, 0, count),
    KOKKOS_LAMBDA(const std::size_t i) {
        output(i) = values(i) - result();
    });
```

The reduction overwrites its result; preinitializing that result is unnecessary.

Both reductions are local to the MPI process. A global reduction requires MPI or a distributed operation such as a Tpetra norm.

### 11.3 Prefix scans

A prefix scan can turn row counts into CSR offsets:

```cpp
Kokkos::parallel_scan(
    "build_offsets",
    Policy(exec, 0, rows + 1),
    KOKKOS_LAMBDA(
        const std::size_t i,
        std::size_t& running,
        const bool final_pass) {
        if (final_pass)
            offsets(i) = running;

        if (i < rows)
            running += counts(i);
    });
```

For counts `[2, 3, 1]`, the offsets are `[0, 2, 5, 6]`.

The `final_pass` condition matters because the scan implementation can invoke the body in multiple passes.

### 11.4 Atomics

If different iterations can update the same location, ordinary `+=` causes a race.

```cpp
Kokkos::atomic_add(&rhs(owner), contribution);
```

provides an atomic update.

An alternative is to assign each output row to one iteration and gather all contributions to that row. This can avoid atomics but changes data access and work distribution.

A fence after a kernel does not repair races inside it. Floating-point accumulation order can also affect the final rounding error.

## 12. Asynchronous execution and fences

“Asynchronous” means the host can continue before submitted work finishes. It does not guarantee overlap or a particular amount of concurrency.

### 12.1 Ordered work on one execution instance

```cpp
Kokkos::parallel_for(
    "produce",
    Policy(exec, 0, count),
    producer);

Kokkos::parallel_for(
    "consume",
    Policy(exec, 0, count),
    consumer);
```

For supported dependent dispatches submitted to the same instance, the application can preserve order without a host fence between them.

The same principle applies to:

```cpp
Kokkos::deep_copy(exec, destination, source);
```

The explicit-instance copy is ordered with work on that instance and can return before completing.

The overload without an execution instance has stronger synchronization semantics and completes before returning.

### 12.2 Instance fence and global fence

```cpp
exec.fence("instance complete");
```

waits for work associated with that instance.

```cpp
Kokkos::fence("all Kokkos work complete");
```

provides a broader completion boundary.

Neither operation performs an MPI barrier or exchanges data between ranks.

For CUDA, an instance fence is conceptually similar to synchronizing the corresponding stream. The portable contract is completion and visibility, rather than a promise about one particular backend function.

### 12.3 Place completion boundaries where ownership changes

Typical boundaries include:

- The host reads produced data.
- Another execution instance consumes it.
- An external library consumes it without an established queue dependency.
- MPI consumes a produced buffer.
- Storage is reused or destroyed.
- Timing requires the work to be complete.

An empty C++ scope does not establish completion:

```cpp
{
    Kokkos::parallel_for(
        "work",
        Policy(exec, 0, count),
        functor);
}

// Work may still be running.
```

### 12.4 MPI is a separate ordering domain

A common exchange sequence is:

1. Pack data in a kernel.
2. Establish completion before MPI consumes the send buffer.
3. Start MPI communication.
4. Compute independent work.
5. Complete communication.
6. Unpack or consume received data.

Device-pointer support and stream integration depend on the MPI implementation.

`MPI_Wait` completes the corresponding MPI request. It does not automatically complete unrelated GPU work. Likewise, `MPI_Barrier` is not a substitute for a Kokkos completion boundary.

KRONA’s exchange module provides its own contract for this sequence; application code should use that contract.

## 13. Plain views and DualView

A plain `Kokkos::View` does not maintain a second synchronized copy automatically.

If a program creates separate host and device views, it must arrange the required transfers.

`Kokkos::DualView` associates host and device representations and tracks which side has been modified. Its explicit interface includes operations such as:

```cpp
dual.modify_host();
dual.sync_device();
```

Modification tracking does not mean that every write triggers an immediate copy.

Tpetra uses managed host/device access machinery, including wrapped dual views, so its data-access methods have more behavior than an ordinary descriptor copy.

## 14. Teuchos utilities

### 14.1 Teuchos::RCP

`RCP<T>` is a reference-counted pointer used for Trilinos objects.

```cpp
auto matrix = Teuchos::rcp(new Matrix(graph));
auto same_matrix = matrix;
```

The second statement shares the matrix object. It does not construct another matrix or duplicate its coefficients.

For this expression:

```cpp
auto matrix = Teuchos::rcp(new Matrix(graph));
```

the responsibilities are:

1. `new Matrix(graph)` constructs the matrix object.
2. Its constructor creates the numerical storage it requires.
3. `rcp(...)` manages the object’s lifetime.

The C++ matrix object normally lives on the host. Its coefficient arrays can live in device memory.

### 14.2 Raw pointers are not coefficient arrays

```cpp
auto pointer = matrix.getRawPtr();
```

returns a pointer to the matrix object.

It does not return a pointer to the matrix’s coefficient buffer, and it does not extend the object’s lifetime.

Do not create a second independently owning `RCP` from an already owned raw pointer. Copy the existing `RCP` or use the appropriate Teuchos cast.

`RCP<const T>` gives const access to the object. A const `RCP<T>` handle still refers to a mutable object.

### 14.3 Teuchos::ArrayView

`ArrayView<T>` is a borrowed array descriptor frequently used in host-side interfaces.

```cpp
Teuchos::ArrayView<const GlobalOrdinal> ids(
    owned_gids.data(),
    local_count);
```

It does not own the vector and does not transfer its contents to a device.

The receiving function determines whether it copies or retains the supplied data.

### 14.4 Teuchos::ParameterList

A parameter list stores named, typed configuration values and nested sublists.

```cpp
auto parameters = Teuchos::rcp(new Teuchos::ParameterList);

parameters->set("Maximum Iterations", 500);
parameters->set("Convergence Tolerance", 1.0e-10);
```

Names and types are part of the receiving component’s API.

A parameter accepted by one solver manager may be unsupported by another. Do not assume one universal list of parameters for all Belos or preconditioner implementations.

## 15. Tpetra types and explicit instantiation

A typical KRONA-oriented type family is:

```cpp
using Scalar = double;
using LocalOrdinal = int;
using GlobalOrdinal = long long;

using Exec = krona::execution::ExecutionSpace;

using Node =
    Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Exec>;

using Map =
    Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;

using Graph =
    Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>;

using Matrix =
    Tpetra::CrsMatrix<
        Scalar, LocalOrdinal, GlobalOrdinal, Node>;

using Vector =
    Tpetra::Vector<
        Scalar, LocalOrdinal, GlobalOrdinal, Node>;

using MultiVector =
    Tpetra::MultiVector<
        Scalar, LocalOrdinal, GlobalOrdinal, Node>;

using Operator =
    Tpetra::Operator<
        Scalar, LocalOrdinal, GlobalOrdinal, Node>;
```

Their template parameters describe:

| Parameter | Meaning |
|---|---|
| `Scalar` | Numerical value type |
| `LocalOrdinal` | Local row and column indices |
| `GlobalOrdinal` | Global identifiers |
| `Node` | Kokkos execution/memory configuration used by Tpetra |

`Node` does not mean an MPI rank or a mesh node.

Local and global ordinal types can differ to reduce index storage while retaining a large global identifier range.

### 15.1 Explicit template instantiation

A Trilinos installation commonly precompiles selected template combinations through explicit template instantiation, or ETI.

The required combination includes scalar, ordinal, and backend types.

Changing a project alias to an unsupported combination does not add that combination to the installed library. Missing ETI can appear as configuration errors or unresolved symbols.

KRONA’s mesh index widths and Tpetra’s ordinal ABI are separate choices. Convert between them with appropriate range checks.

## 16. Maps: distributions of identifiers

A `Tpetra::Map` describes which global identifiers occur on each MPI process and their local ordering.

For example:

| Rank | Local index | Global identifier |
|---|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 3 |
| 0 | 2 | 7 |
| 1 | 0 | 1 |
| 1 | 1 | 2 |
| 1 | 2 | 4 |

Local index `1` identifies different global entries on different ranks.

A map can support:

```cpp
map->getGlobalElement(local_index);
map->getLocalElement(global_id);
```

A failed lookup returns the appropriate invalid ordinal.

### 16.1 One-to-one and overlapping maps

A one-to-one map assigns every represented global identifier to one process.

An overlapping map can contain an identifier on multiple processes. This is useful for imported or ghosted data.

A map describes an indexing distribution. It does not contain matrix coefficients.

### 16.2 A map is a host-side object with device support

A map can maintain host and device indexing structures. Some simple maps can use arithmetic relationships instead of explicitly storing every lookup.

Therefore, constructing a map is not adequately described as “allocate a map on the GPU.”

For device kernels, obtain an appropriate local map descriptor:

```cpp
const auto local_map = map->getLocalMap();
```

Do not capture the host-side `RCP<Map>` into a GPU kernel.

### 16.3 Constructing a map from owned identifiers

The pattern used in KRONA resembles:

```cpp
const auto unknown_global_size =
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

auto map = Teuchos::rcp(new Map(
    unknown_global_size,
    Teuchos::ArrayView<const GlobalOrdinal>(
        owned_gids.data(),
        local_count),
    GlobalOrdinal{0},
    comm));
```

The invalid global-size sentinel requests computation of the global count from local contributions.

`0` is the index base in this example. Use the base required by the actual global numbering.

Map construction is collective over its communicator, including ranks with no local entries. The input lists can differ by rank.

### 16.4 Remote ownership lookup

```cpp
const auto status = map->getRemoteIndexList(
    remote_ids,
    owner_ranks);
```

This queries the distribution and fills `owner_ranks`. It does not redistribute the map.

For general distributed maps, this is a collective operation and can communicate. All participating ranks must follow the required call sequence, even if their query lists are empty.

## 17. The four matrix maps

A distributed matrix commonly has four relevant maps:

| Map | Meaning |
|---|---|
| Row map | Rows stored on this process |
| Column map | Column identifiers referenced by locally stored entries |
| Domain map | Distribution of the input vector |
| Range map | Distribution of the output vector |

For

\[
y=Ax,
\]

`x` belongs to the domain and `y` belongs to the range.

In a conventional row-owned square system:

- Row, domain, and range maps often coincide.
- The column map also contains remote identifiers required by local rows.

For example, a rank owning rows 0 and 1 may need `x(2)` to evaluate row 1. Its column map can therefore include global identifier 2 without owning row 2.

### 17.1 Matrix ghosts and application ghosts

Tpetra can import vector entries needed for a matrix application.

KRONA’s mesh halo exchange serves a different purpose: it updates application fields needed by mesh-based operators.

The two systems can use different orderings and communication plans. Tpetra’s column-map ordering must not be assumed to equal the mesh ghost ordering.

### 17.2 No automatic ghost rows

In the usual row-owned configuration, local nonzeros include entries coupling owned rows to remote columns.

They do not imply that complete remote rows are also stored.

The same global coupling can appear in different owned rows on different ranks because those are different matrix entries.

For rectangular or more general distributed operators, row, domain, and range maps can differ.

## 18. CSR graphs and numerical matrices

### 18.1 CSR storage

A local CSR representation consists of:

- Row offsets.
- Column indices.
- Numerical values.

For example:

| Array | Contents |
|---|---|
| `row_offsets` | `[0, 2, 5]` |
| `column_indices` | `[0, 1, 0, 1, 2]` |
| `values` | `[4, -1, -1, 4, -1]` |

Row 0 uses entries `[0, 2)`. Row 1 uses entries `[2, 5)`.

The local multiplication is:

\[
y_i=
\sum_{k=\mathrm{offsets}(i)}^{\mathrm{offsets}(i+1)-1}
a_k\,x_{\mathrm{columns}(k)}.
\]

After graph finalization, the column indices refer to the local column map.

### 18.2 CrsGraph stores the pattern

A graph records which entries exist:

```cpp
auto graph = Teuchos::rcp(new Graph(row_map, max_entries_per_row));
```

During construction:

```cpp
graph->insertGlobalIndices(global_row, columns);
```

adds column identifiers to a row.

The per-row allocation information describes capacity. It does not insert entries or assign coefficients.

### 18.3 fillComplete(domain, range)

```cpp
graph->fillComplete(domain_map, range_map);
```

finalizes the symbolic structure for computation.

This can involve sorting, duplicate handling, global-to-local conversion, map construction, communication-plan construction, and distributed communication.

For a square system, both arguments are often the same owned map because input and output use the same distribution.

`fillComplete()` is a structural lifecycle operation. It is not a general application execution-stream fence.

### 18.4 The matrix adds numerical storage

```cpp
auto matrix = Teuchos::rcp(new Matrix(graph));
```

constructs a matrix with coefficients corresponding to the finalized graph.

Matrices constructed from the same constant graph can share that graph while maintaining independent coefficient storage.

```cpp
auto first  = Teuchos::rcp(new Matrix(graph));
auto second = Teuchos::rcp(new Matrix(graph));
```

creates two coefficient arrays.

```cpp
auto second = first;
```

shares one matrix object.

A constant graph restricts structural changes. Numerical assembly updates values associated with existing entries.

### 18.5 Input order is not necessarily native order

After finalization, entry order can differ from the caller’s original CSR ordering.

A precomputed permutation or assembly map can translate:

\[
\text{application entry}\longrightarrow\text{native coefficient slot}.
\]

Such a map depends on the exact finalized graph and indexing. It cannot be reused merely because two matrices have the same number of entries.

### 18.6 row_map has two unrelated meanings

```cpp
matrix->getRowMap()
```

returns a distributed `Tpetra::Map`.

```cpp
local_graph.row_map
```

is the CSR row-offset array.

The similar names describe different objects.

## 19. Tpetra vectors and local access

### 19.1 A vector is one distributed column

```cpp
auto x = Teuchos::rcp(new Vector(domain_map, true));
```

allocates a distributed vector.

The boolean requests zero initialization. It does not choose host/device placement or define the operation’s synchronization policy.

```cpp
auto workspace = Teuchos::rcp(new Vector(domain_map, false));
```

does not request zero initialization. Write the required entries before reading them.

A `MultiVector` contains multiple distributed columns:

```cpp
auto X = Teuchos::rcp(new MultiVector(domain_map, 3, true));
```

### 19.2 Local views are two-dimensional

A vector’s local computational view is indexed as:

```cpp
local(i, 0);
```

A multivector’s view uses:

```cpp
local(i, vector_column);
```

The second index represents a multivector column. It does not automatically represent the physical components of one cell.

Tpetra accessors return views of its existing numerical storage; they can synchronize the requested representation before returning.

### 19.3 Access tags express intent

| Access tag | Meaning |
|---|---|
| `ReadOnly` | Preserve data and provide read access |
| `ReadWrite` | Preserve existing data and allow changes |
| `OverwriteAll` | Allow replacing all exposed entries without preserving their old contents |

```cpp
auto local =
    x->getLocalViewDevice(Tpetra::Access::OverwriteAll);
```

can avoid copying an old host representation to the device when those values will be replaced.

It does not mean “perform no synchronization under every circumstance.” A subview can require preserving data outside the selected region.

### 19.4 Scope local access carefully

```cpp
{
    auto local =
        x->getLocalViewDevice(Tpetra::Access::OverwriteAll);

    Kokkos::parallel_for(
        "fill_x",
        Policy(exec, 0, local.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i) {
            local(i, 0) = 1.0;
        });

    exec.fence("external vector writes complete");
}

// Local access handle has been released.
use_vector_through_tpetra(*x);
```

The fence handles execution completion. The scope releases the access handle.

Avoid retaining host and device access handles simultaneously. Also release local handles before invoking Tpetra operations that may choose another representation.

Modern Tpetra tracks these accesses. Returning a raw pointer and discarding its access handle can bypass that tracking.

## 20. Direct matrix access in Kokkos kernels

For CSR coefficients:

```cpp
auto values =
    matrix->getLocalValuesDevice(Tpetra::Access::OverwriteAll);
```

returns a native Kokkos view of the local coefficient storage.

For the local graph:

```cpp
auto graph_device =
    matrix->getCrsGraph()->getLocalGraphDevice();
```

provides CSR arrays:

```cpp
graph_device.row_map(row);
graph_device.entries(entry);
```

Obtaining these objects does not request an independent copy of the matrix. However, access can trigger representation synchronization or construct local descriptor objects.

### 20.1 Full overwrite

```cpp
{
    auto values =
        matrix->getLocalValuesDevice(Tpetra::Access::OverwriteAll);

    Kokkos::parallel_for(
        "zero_coefficients",
        Policy(exec, 0, values.extent(0)),
        KOKKOS_LAMBDA(const std::size_t k) {
            values(k) = 0.0;
        });

    // Further assembly on the same execution instance can follow.
    exec.fence("matrix assembly complete");
}
```

For accumulation into existing coefficients, use the appropriate read/write access and arrange race-free updates.

The public coefficient accessors distinguish read-only, read/write, and full-overwrite access.

### 20.2 Coefficient slots are not column identifiers

For an existing entry:

```cpp
values(entry) = coefficient;
```

`entry` is a position in the coefficient array.

It is not generally:

- A global column identifier.
- A local column identifier.
- A face index.
- A row index.

KRONA’s assembly maps connect mesh entities to the required native positions.

### 20.3 Structural and numerical updates are separate

The traditional insertion and replacement APIs have fill-state requirements. Direct local coefficient access is a separate interface for numerical storage.

Changing coefficients through an appropriate local access path does not by itself require rebuilding an unchanged graph.

If topology or coupling structure changes, the graph and dependent assembly maps may need rebuilding.

## 21. BSR matrices and component ordering

BSR stores a small dense block for each nonzero in a block graph.

For \(B\) unknowns per cell:

- A block row represents a cell.
- A block column represents a coupled cell.
- A block entry stores \(B\times B\) numerical coefficients.

The block multiplication is:

\[
\mathbf y_i=\sum_j A_{ij}\mathbf x_j,
\qquad A_{ij}\in\mathbb R^{B\times B}.
\]

### 21.1 What BSR saves

For \(N_b\) dense nonzero blocks:

- Numerical storage contains approximately \(N_bB^2\) scalars.
- BSR stores one column index per block.
- A fully expanded scalar CSR pattern generally stores \(B^2\) column indices per dense block.

BSR mainly reduces index and graph overhead and exposes block structure. It does not reduce the number of coefficients in a dense block.

### 21.2 Point vectors are compatible with a block graph

KRONA’s cell-major ordering uses:

\[
\operatorname{pointGID}(g,c)=gB+c.
\]

Locally:

\[
\operatorname{pointLID}(i,c)=iB+c.
\]

For \(B=3\):

| Point-vector index | Meaning |
|---:|---|
| 0 | Cell 0, component 0 |
| 1 | Cell 0, component 1 |
| 2 | Cell 0, component 2 |
| 3 | Cell 1, component 0 |

The matrix can keep its block graph while the solver vector uses a point map. Expanding the vector’s indexing does not require persistently expanding the block graph into scalar CSR.

### 21.3 Dense block layout is another mapping

A flat block coefficient buffer does not by itself specify whether a block is row-major or column-major.

A three-dimensional adapter can expose:

```cpp
blocks(block_entry, equation_component, unknown_component);
```

while preserving the native block layout.

Use the public KRONA block-access interface for application assembly. If working with the native type, respect its declared little-block layout.

Support and possible conversions in preconditioner adapters depend on the selected implementation. Native BSR storage alone does not guarantee that every downstream algorithm preserves BSR internally.

### 21.4 Three uses of the word “block”

| Term | Meaning |
|---|---|
| BSR block | Physical coupling between components |
| Multivector column | A separate vector or right-hand side |
| Block Krylov method | A solver algorithm processing multiple vectors together |

These concepts can be combined, but they are independent.

## 22. Operators, Belos, and preconditioners

### 22.1 The operator abstraction

A `Tpetra::Operator` represents a distributed linear action:

\[
Y=\alpha\,\operatorname{op}(A)X+\beta Y.
\]

Its interface includes domain and range maps and an `apply()` operation.

A sparse matrix is one implementation. A matrix-free action or precision-converting adapter can implement the same interface.

Belos can therefore work with an operator without knowing its CSR or BSR storage. A chosen preconditioner may still require an assembled matrix.

### 22.2 Belos::LinearProblem

A linear problem connects:

- The operator \(A\).
- The solution/initial guess \(X\).
- The right-hand side \(B\).
- Optional left and right preconditioners.

```cpp
using Problem =
    Belos::LinearProblem<Scalar, MultiVector, Operator>;

auto problem =
    Teuchos::rcp(new Problem(matrix, x, b));

const bool ready = problem->setProblem();
```

The initial contents of `x` provide the starting guess. Belos updates it in place.

`setProblem()` prepares the problem and its associated state. It does not solve the system or prove the matrix has the properties required by the selected algorithm.

### 22.3 Solver managers and factories

A solver manager implements an iterative algorithm.

```cpp
Belos::SolverFactory<Scalar, MultiVector, Operator> factory;

auto parameters = Teuchos::rcp(new Teuchos::ParameterList);
parameters->set("Maximum Iterations", 500);
parameters->set("Convergence Tolerance", 1.0e-10);
parameters->set("Num Blocks", 50);

auto solver = factory.create("GMRES", parameters);
solver->setProblem(problem);

const auto status = solver->solve();
```

`Num Blocks` has a method-specific meaning. In the usual single-RHS restarted GMRES configuration, it controls the Krylov subspace/restart size.

The factory can report supported names:

```cpp
const auto names = factory.supportedSolverNames();
const bool available = factory.isSupported("BICGSTAB");
```

Availability depends on the build, scalar/backend combination, and registered implementations. Include the Tpetra adapter when using Belos with Tpetra types.

### 22.4 Selecting a method

| Method | Main mathematical requirement or purpose |
|---|---|
| GMRES | General nonsymmetric linear systems |
| BiCGStab | General nonsymmetric systems; a short-recurrence alternative |
| CG | Symmetric/Hermitian positive-definite systems with compatible preconditioning |
| MINRES | Symmetric/Hermitian systems, including indefinite cases, with compatible preconditioning |
| Flexible GMRES | Situations with a varying preconditioner |

A symmetry declaration does not check, repair, or symmetrize a matrix:

```cpp
problem->setHermitian(true);
```

It records a mathematical property the application is responsible for satisfying.

Preconditioning can alter the relevant operator properties. A symmetric matrix combined with an unsuitable preconditioner is not automatically a valid CG configuration.

### 22.5 What a preconditioner applies

Write the preconditioning action as \(B\approx A^{-1}\).

Right preconditioning solves:

\[
AB\,y=b,\qquad x=By.
\]

Left preconditioning solves:

\[
BAx=Bb.
\]

The object passed to Belos applies the approximate inverse action. It does not usually store a dense inverse matrix.

```cpp
problem->setRightPrec(preconditioner);
```

The selected solver can impose restrictions on preconditioning side or behavior.

### 22.6 Ifpack2 setup

A basic relaxation preconditioner can be constructed as follows:

```cpp
#include <Ifpack2_Factory.hpp>

Teuchos::RCP<const Matrix> const_matrix = matrix;

auto preconditioner =
    Ifpack2::Factory::create<Matrix>(
        "RELAXATION",
        const_matrix);

Teuchos::ParameterList options;
options.set("relaxation: type", std::string("Jacobi"));
options.set("relaxation: sweeps", 1);
options.set("relaxation: damping factor", 0.8);

preconditioner->setParameters(options);
preconditioner->initialize();
preconditioner->compute();

problem->setRightPrec(preconditioner);

// Prepare or re-prepare the problem after changing its setup.
problem->setProblem();
```

The broad lifecycle is:

| Stage | Purpose |
|---|---|
| Construction/configuration | Select an implementation and its settings |
| `initialize()` | Prepare structural information |
| `compute()` | Prepare numerical data from matrix coefficients |
| `apply()` | Apply the preconditioning action |

Ifpack2 also provides incomplete factorizations and domain decomposition. Their effectiveness and parallel behavior differ.

### 22.7 MueLu and algebraic multigrid

AMG constructs a hierarchy of problems at different resolutions.

A simplified two-level description is:

1. Smooth error on the fine level.
2. Restrict the residual to a coarse level.
3. Solve or approximately solve the coarse problem.
4. Prolongate a correction.
5. Apply further smoothing as configured.

A coarse matrix commonly has the form:

\[
A_c=RAP.
\]

AMG depends on numerical coefficients as well as the sparsity pattern. Changing matrix values can affect smoothers, transfer operators, coarse matrices, and convergence.

MueLu provides multigrid preconditioners for several PDE problem classes, with configuration for application-specific behavior.

A native setup pattern is:

```cpp
#include <MueLu_CreateTpetraPreconditioner.hpp>

Teuchos::RCP<Operator> matrix_operator = matrix;

Teuchos::ParameterList options;
options.set("max levels", 10);

auto preconditioner =
    MueLu::CreateTpetraPreconditioner<
        Scalar, LocalOrdinal, GlobalOrdinal, Node>(
            matrix_operator,
            options);

problem->setRightPrec(preconditioner);
problem->setProblem();
```

MueLu also exposes reuse interfaces. Reuse is a configured setup/update strategy; it does not imply that a hierarchy should keep stale numerical data whenever the graph is unchanged.

### 22.8 Residuals and workspace

The true residual is:

\[
r=b-Ax.
\]

A separate residual vector can serve diagnostics or convergence verification:

```cpp
Vector residual(range_map, false);

matrix->apply(*x, residual);
residual.update(1.0, *b, -1.0);

const auto residual_norm = residual.norm2();
```

This is an additional allocation.

Belos also allocates internal workspace. Restarted GMRES typically stores a collection of Krylov basis vectors whose memory grows with the restart dimension.

A reported iterative residual can differ from an explicitly recomputed true residual because of scaling, preconditioning, and floating-point effects.

## 23. Complete example: native assembly and solve

This example constructs a distributed tridiagonal system:

\[
2x_i-x_{i-1}-x_{i+1}=1,
\]

with missing off-diagonal entries at the two ends.

It demonstrates:

- Tpetra initialization.
- A distributed map.
- A finalized graph.
- Independent coefficient storage.
- Direct device-side coefficient and RHS assembly.
- A Belos solve.
- An explicitly recomputed residual.

It uses native APIs to explain the underlying objects. KRONA’s wrappers add their own access and commit contracts.

```cpp
#include <Kokkos_Core.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosLinearProblem.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[])
{
    Tpetra::ScopeGuard runtime(&argc, &argv);

    using Scalar = double;
    using LO = int;
    using GO = long long;

    using Map = Tpetra::Map<LO, GO>;
    using Node = typename Map::node_type;

    using Graph = Tpetra::CrsGraph<LO, GO, Node>;
    using Matrix = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
    using Vector = Tpetra::Vector<Scalar, LO, GO, Node>;
    using MV = Tpetra::MultiVector<Scalar, LO, GO, Node>;
    using OP = Tpetra::Operator<Scalar, LO, GO, Node>;

    using Exec = typename Matrix::execution_space;
    using Policy =
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<LO>>;

    const auto comm = Tpetra::getDefaultComm();

    constexpr GO global_rows = 12;

    Teuchos::RCP<const Map> map = Teuchos::rcp(new Map(
        static_cast<Tpetra::global_size_t>(global_rows),
        GO{0},
        comm));

    const LO local_rows =
        static_cast<LO>(map->getLocalNumElements());

    // Symbolic setup on the host.
    auto graph = Teuchos::rcp(new Graph(map, std::size_t{3}));

    for (LO row = 0; row < local_rows; ++row) {
        const GO gid = map->getGlobalElement(row);

        Teuchos::Array<GO> columns;

        if (gid > 0)
            columns.push_back(gid - 1);

        columns.push_back(gid);

        if (gid + 1 < global_rows)
            columns.push_back(gid + 1);

        graph->insertGlobalIndices(gid, columns());
    }

    graph->fillComplete(map, map);

    auto matrix = Teuchos::rcp(new Matrix(graph));
    matrix->fillComplete(map, map);

    auto x = Teuchos::rcp(new Vector(map, true));
    auto b = Teuchos::rcp(new Vector(map, false));

    Exec exec;

    // Direct numerical assembly into Tpetra storage.
    {
        const auto local_graph = graph->getLocalGraphDevice();

        const auto row_map =
            matrix->getRowMap()->getLocalMap();

        const auto column_map =
            matrix->getColMap()->getLocalMap();

        auto coefficients =
            matrix->getLocalValuesDevice(
                Tpetra::Access::OverwriteAll);

        auto rhs =
            b->getLocalViewDevice(
                Tpetra::Access::OverwriteAll);

        Kokkos::parallel_for(
            "assemble_tridiagonal",
            Policy(exec, 0, local_rows),
            KOKKOS_LAMBDA(const LO row) {
                const GO row_gid =
                    row_map.getGlobalElement(row);

                const auto begin = local_graph.row_map(row);
                const auto end = local_graph.row_map(row + 1);

                for (auto entry = begin; entry < end; ++entry) {
                    const LO local_column =
                        local_graph.entries(entry);

                    const GO column_gid =
                        column_map.getGlobalElement(local_column);

                    coefficients(entry) =
                        column_gid == row_gid
                            ? Scalar{2}
                            : Scalar{-1};
                }

                rhs(row, 0) = Scalar{1};
            });

        // Handoff from the application execution instance
        // to native solver operations.
        exec.fence("native system assembly complete");
    }

    using Problem = Belos::LinearProblem<Scalar, MV, OP>;

    auto problem =
        Teuchos::rcp(new Problem(matrix, x, b));

    if (!problem->setProblem())
        throw std::runtime_error("Belos problem setup failed");

    auto parameters =
        Teuchos::rcp(new Teuchos::ParameterList);

    parameters->set("Maximum Iterations", 100);
    parameters->set("Convergence Tolerance", 1.0e-10);
    parameters->set("Num Blocks", 30);

    Belos::SolverFactory<Scalar, MV, OP> factory;

    auto solver = factory.create("GMRES", parameters);
    solver->setProblem(problem);

    const auto status = solver->solve();

    // All ranks participate in these distributed operations.
    Vector residual(map, false);

    matrix->apply(*x, residual);
    residual.update(Scalar{1}, *b, Scalar{-1});

    const auto residual_norm = residual.norm2();
    const auto rhs_norm = b->norm2();

    if (comm->getRank() == 0) {
        std::cout
            << "Converged: " << (status == Belos::Converged)
            << "\nIterations: " << solver->getNumIters()
            << "\nTrue relative residual: "
            << residual_norm / rhs_norm
            << '\n';
    }

    return status == Belos::Converged ? 0 : 1;
}
```

The example deliberately uses one explicit completion boundary before entering native solver operations. It does not fence between operations that form one ordered application kernel phase.

A production application should use its established collective error-handling policy.

## 24. Moving between application fields and solver vectors

An application field and a solver vector often have different shapes.

A KRONA field can include:

- Owned cells.
- MPI ghost cells.
- Boundary-face slots.

A conventional solver vector contains the owned algebraic unknowns described by its map.

Do not infer the solver’s local size from the entire extended field extent.

### 24.1 Packing several components

For a cell-major point map with \(B\) components:

```cpp
{
    auto rhs =
        b->getLocalViewDevice(Tpetra::Access::OverwriteAll);

    Kokkos::parallel_for(
        "pack_rhs",
        Policy(exec, 0, owned_cells),
        KOKKOS_LAMBDA(const std::size_t i) {
            for (int c = 0; c < B; ++c)
                rhs(i * B + c, 0) = residual_fields(i, c);
        });

    exec.fence("rhs ready for native solver");
}
```

This assumes:

- The point map preserves the intended cell order.
- Its local extent is `owned_cells * B`.
- The source contains the required owned entries.
- Every destination entry is written.

Several separate scalar arrays can be packed in the same way or through KRONA’s transfer utilities.

### 24.2 Unpacking the solution

```cpp
{
    const auto solution =
        x->getLocalViewDevice(Tpetra::Access::ReadOnly);

    Kokkos::parallel_for(
        "unpack_solution",
        Policy(exec, 0, owned_cells),
        KOKKOS_LAMBDA(const std::size_t i) {
            for (int c = 0; c < B; ++c)
                fields(i, c) = solution(i * B + c, 0);
        });

    exec.fence("solution extraction complete");
}
```

This writes the owned field region. Updating application ghosts and boundary slots is a subsequent application operation.

Direct local access avoids host staging. Copying between independent field storage and solver storage still performs element reads and writes; it is not automatic storage aliasing.

## 25. Precision and conversion costs

KRONA distinguishes several precision roles:

| Role | Purpose |
|---|---|
| Storage precision | Persistent field values |
| Compute precision | Local numerical calculations |
| Geometry precision | Stored mesh metrics |
| Accumulation precision | Reductions and accumulated expressions |
| Solver precision | Linear-system arithmetic |
| Preconditioner precision | Preconditioner matrix and arithmetic |

These choices affect both accuracy and memory traffic.

### 25.1 Conversion is not aliasing

An array of `float` cannot become an array of `double` through a pointer cast.

A real conversion requires reading the source values and writing converted destination values.

When the execution space can access both arrays, `deep_copy` can handle supported scalar conversion. A custom kernel can additionally reject overflow or nonfinite converted coefficients.

Finite converted values do not prove that the conversion preserved enough accuracy.

### 25.2 Mixed-precision preconditioning

The outer solver can use one scalar type while the preconditioning action uses another.

An adapter can conceptually perform:

\[
x_{\mathrm{inner}}
=\operatorname{convert}(x_{\mathrm{outer}}),
\]

\[
z_{\mathrm{inner}}
=B_{\mathrm{inner}}x_{\mathrm{inner}},
\]

\[
y_{\mathrm{outer}}
=\operatorname{convert}(z_{\mathrm{inner}}).
\]

This can require:

- A matrix in preconditioner precision.
- Conversion workspaces.
- Conversion kernels during preconditioner application.

The conversion belongs to each adapted operator application. If only the preconditioner is adapted, it is not added to every application of the original matrix.

For equal precisions, KRONA’s compile-time selection can use the native operator directly and avoid the precision adapter.

Whether mixed precision pays off depends on the preconditioner cost, conversion bandwidth, memory savings, and convergence.

## 26. Native lifecycle versus KRONA commits

Several mechanisms track different things:

| Mechanism | Tracks or establishes |
|---|---|
| Kokkos execution instance | Ordering of submitted work |
| Kokkos fence | Completion and visibility |
| Tpetra access tags | Intended data access and representation synchronization |
| Tpetra `fillComplete()` | Finalized algebraic structure and associated setup |
| KRONA commit operation | The wrapper’s pending-access and revision contract |
| Preconditioner preparation | Setup for the required matrix state |

There is no universal native Tpetra operation named `commit()` with KRONA’s semantics.

A typical KRONA workflow is:

1. Obtain application-facing writable views.
2. Submit assembly or transfer kernels on the intended instance.
3. End the access phase.
4. Commit through the wrapper.
5. Prepare or refresh the preconditioner as required.
6. Solve.
7. Extract results through the documented access path.

A fence alone does not necessarily update a wrapper’s pending state or matrix revision.

Access through a native pointer can bypass wrapper bookkeeping. Such access must preserve the wrapper’s contract; otherwise cached preconditioners or pending-operation checks can become inconsistent.

## 27. Reading KRONA source code

When encountering an unfamiliar object, answer these questions in order.

### 27.1 What kind of object is this?

Determine whether it is:

- A type alias.
- An owning storage object.
- A borrowed descriptor.
- A kernel functor.
- A distributed algebra object.
- A solver or preconditioner manager.

A class named `View` in KRONA can be a lightweight wrapper with its own API. It is not automatically the native `Kokkos::View`.

### 27.2 Who owns the numerical memory?

Look for:

- Allocating view constructors.
- Raw-pointer wrapping.
- `Unmanaged`.
- Owning storage members.
- Shared `RCP` objects.
- Borrowed views returned by access methods.

A mesh or field view can expose memory owned elsewhere. Its small size does not tell you whether its lifetime is independent.

### 27.3 What does each index mean?

Keep these namespaces separate:

| Index | Identifies |
|---|---|
| Mesh local cell index | An owned or ghost mesh cell |
| Extended field index | A cell or a boundary-face slot |
| Global cell identifier | A distributed mesh entity |
| Algebraic local row | An entry in a row or vector map |
| Algebraic local column | An entry in the column map |
| CSR entry index | A coefficient position |
| BSR entry index | A dense block position |
| Component index | A physical or algebraic component |

A correct integer type does not imply a correct index interpretation.

### 27.4 Where does the work execute?

Find the execution policy and instance.

A function can:

- Perform host setup.
- Launch asynchronous kernels.
- Run a host-result reduction.
- Enter a collective MPI operation.
- Call a native library operation with its own execution behavior.

The function name alone is insufficient.

### 27.5 What establishes completion?

Look for the actual boundary:

- Same-instance dependency.
- Instance fence.
- Blocking copy.
- Host-result reduction.
- Exchange completion.
- Wrapper commit.
- Native library contract.

Also check that all buffers survive until their last use.

### 27.6 What is cached?

Static meshes permit reuse of:

- Connectivity.
- Geometry.
- Sparse graphs.
- Assembly maps.
- Communication plans.

Numerical state can change more frequently:

- Fields.
- Matrix coefficients.
- Right-hand sides.
- Preconditioner numerical data.

Geometry changes, topology changes, and coefficient changes invalidate different caches.

## 28. C++ syntax frequently used in these interfaces

### 28.1 typename

```cpp
template<class Exec>
struct Storage {
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
};
```

`Exec::memory_space` depends on a template parameter. `typename` identifies it as a type in a dependent context.

It does not construct a memory space or allocate anything.

C++20 permits omitting it in some type-only contexts, but the explicit form remains common and clear.

### 28.2 template after a dependent object

```cpp
scheme.template increment<Real>(side, data, component);
```

The `template` keyword tells the parser that `increment` is a member template when the object’s type is dependent.

Again, this is parsing information, not runtime work.

### 28.3 Concepts and if constexpr

```cpp
if constexpr (requires { model.at(context, condition); }) {
    // Compile the provider-based path.
} else {
    // Compile the direct-model path.
}
```

The `requires` expression checks whether the expression is well-formed for the instantiated types.

`if constexpr` selects the applicable branch at compile time. This is useful for optional capabilities without runtime branching in each work item.

### 28.4 Compile-time traits do not prove runtime validity

A type-level accessibility check cannot prove that:

- An unmanaged pointer is still alive.
- A cell index is in range.
- A ghost value is current.
- A matrix coefficient is finite.
- A custom provider belongs to the expected mesh.

Those remain runtime data and application contracts.

## 29. Common mistakes

| Mistake | Correct interpretation |
|---|---|
| Copying a view to duplicate an array | View copies normally share storage |
| Treating every managed-capable view instance as an owner | Raw-pointer construction can still be borrowed |
| Expecting a wrapped pointer to keep its source alive | The actual owner must remain alive |
| Treating a const descriptor as read-only data | Const elements require a const element type |
| Using `.data()[i]` for an arbitrary component subview | The component can be strided |
| Reading device-only storage directly on the host | Obtain an appropriate host representation |
| Assuming a mirror contains copied values | Mirror creation and data copying are separate |
| Assuming `create_mirror_view()` creates an independent snapshot | It can alias |
| Assuming every `deep_copy` can convert layout across separate devices | Conversion requires a suitable accessible execution path |
| Assuming `parallel_for` has finished on return | Dispatch can be asynchronous |
| Using a fence to repair conflicting writes | Races must be removed inside the algorithm |
| Treating a Kokkos reduction as an MPI reduction | It reduces only the local process’s work |
| Treating `getLocalViewDevice()` as an unconditional no-op | It can synchronize representations |
| Using `OverwriteAll` for a partial update | Preserve untouched values with the appropriate access mode |
| Holding host and device Tpetra access handles together | Keep access phases scoped and compatible |
| Assuming matrix finalization preserves insertion order | Build against the finalized native ordering |
| Confusing a column-map index with a mesh ghost index | The orderings are independent |
| Assuming a block vector requires a scalar-expanded matrix graph | Point-vector indexing and block graph storage are compatible |
| Assuming `setHermitian(true)` checks symmetry | It declares an application-guaranteed property |
| Reusing a preconditioner solely because the graph is unchanged | Its numerical data can also depend on coefficients |
| Modifying native objects while ignoring wrapper state | Preserve KRONA’s access, commit, and revision contracts |

## 30. Performance habits that follow from the model

Allocate persistent work arrays outside repeated face or cell loops.

Keep data in the memory space where it is consumed. Avoid alternating host and device access when one longer phase on each side is sufficient.

Pass small descriptors to kernels. Leave storage ownership in longer-lived objects.

Use execution instances consistently so dependencies do not require unnecessary host waits.

Choose gathering or atomic accumulation deliberately. Measure the actual workload instead of assuming atomics or row-based assembly always wins.

Reuse a finalized graph when its structure remains valid. Reuse numerical setup according to the selected algorithm’s contract.

Remember that a solve contains more than sparse matrix-vector multiplication:

- Krylov vector updates.
- Dot products and global reductions.
- Orthogonalization.
- Preconditioner applications.
- Setup and communication.

A small case can be dominated by launch overhead, communication latency, or solver vector operations. This can limit scaling even when application kernels are efficient.

When measuring asynchronous work, include the intended completion boundary. Otherwise the measurement may capture submission time rather than execution time.

## 31. Further reading

The official references below provide the detailed contracts behind this guide.

- [Kokkos Programming Guide](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide.html)
- [Kokkos View reference](https://kokkos.org/kokkos-core-wiki/API/core/view/view.html)
- [View allocation properties](https://kokkos.org/kokkos-core-wiki/API/core/view/view_alloc.html)
- [Deep copy](https://kokkos.org/kokkos-core-wiki/API/core/view/deep_copy.html)
- [Mirrors](https://kokkos.org/kokkos-core-wiki/API/core/view/create_mirror.html)
- [Parallel reductions](https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_reduce.html)
- [Tpetra overview and tutorial lessons](https://trilinos.github.io/docs/tpetra/index.html)
- [Tpetra Map reference](https://trilinos.github.io/docs/tpetra/class_tpetra_1_1_map.html)
- [Tpetra CrsGraph reference](https://trilinos.github.io/docs/tpetra/class_tpetra_1_1_crs_graph.html)
- [Tpetra CrsMatrix reference](https://trilinos.github.io/docs/tpetra/class_tpetra_1_1_crs_matrix.html)
- [Tpetra MultiVector reference](https://trilinos.github.io/docs/tpetra/class_tpetra_1_1_multi_vector.html)
- [Belos LinearProblem reference](https://trilinos.github.io/docs/belos/class_belos_1_1_linear_problem.html)
- [Ifpack2 overview](https://trilinos.github.io/docs/ifpack2/index.html)
- [MueLu overview](https://trilinos.github.io/docs/muelu/index.html)

After this guide, read KRONA’s mesh, fields, exchange, gradient, boundary-condition, and linear-algebra documentation. Those documents define the application-specific ownership, indexing, asynchronous execution, and commit contracts layered over the native libraries.