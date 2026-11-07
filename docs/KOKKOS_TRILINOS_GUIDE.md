# Kokkos and Trilinos: A Practical Reference

**Versions:** Kokkos 5.2.1 and Trilinos 17.2.1. **Language:** C++20.

This guide covers Kokkos Core, Kokkos Kernels, Teuchos, Tpetra, Belos, Ifpack2, and MueLu. Examples use real-valued `double` coefficients. Short examples are fragments inside an initialized application; Section 23 is a complete program.

## Contents

- [1. Libraries and initialization](#1-libraries-and-initialization)
- [2. Execution spaces and memory spaces](#2-execution-spaces-and-memory-spaces)
- [3. Kokkos::View](#3-kokkosview)
- [4. Layouts and subviews](#4-layouts-and-subviews)
- [5. Mirrors and data copies](#5-mirrors-and-data-copies)
- [6. Parallel kernels](#6-parallel-kernels)
- [7. Reductions, scans, and atomics](#7-reductions-scans-and-atomics)
- [8. Asynchronous execution and fences](#8-asynchronous-execution-and-fences)
- [9. Kokkos::DualView](#9-kokkosdualview)
- [10. Teuchos utilities](#10-teuchos-utilities)
- [11. Tpetra types](#11-tpetra-types)
- [12. Tpetra::Map](#12-tpetramap)
- [13. What getLocalMap() returns](#13-what-getlocalmap-returns)
- [14. Import and Export](#14-import-and-export)
- [15. Vector and MultiVector](#15-vector-and-multivector)
- [16. The four matrix maps](#16-the-four-matrix-maps)
- [17. Tpetra::CrsGraph](#17-tpetracrsgraph)
- [18. Tpetra::CrsMatrix](#18-tpetracrsmatrix)
- [19. Local matrix and graph access](#19-local-matrix-and-graph-access)
- [20. Tpetra::BlockCrsMatrix](#20-tpetrablockcrsmatrix)
- [21. Operators and Belos](#21-operators-and-belos)
- [22. Ifpack2 and MueLu](#22-ifpack2-and-muelu)
- [23. Complete example: assemble and solve](#23-complete-example-assemble-and-solve)
- [24. Official references](#24-official-references)

## 1. Libraries and initialization

| Library | Responsibility |
|---|---|
| Kokkos Core | Arrays, memory spaces, parallel loops, reductions, and synchronization within a process |
| Kokkos Kernels | Numerical routines built on Kokkos: dense and sparse linear algebra and graph algorithms |
| Teuchos | Shared pointers, arrays, parameter lists, and communicator interfaces |
| Tpetra | Distributed vectors, sparse matrices, index distributions, and data transfers |
| Belos | Iterative methods for solving linear systems |
| Ifpack2 | Relaxation, incomplete-factorization, and domain-decomposition preconditioners |
| MueLu | Algebraic multigrid solvers and preconditioners |

MPI distributes work across processes. Kokkos runs work within each process, on the CPU or GPU selected by the build.

**Host** means the CPU executing ordinary application code. **Device** means the execution resource associated with a Kokkos object. In a Serial or OpenMP build, device data can also be ordinary CPU memory.

For a Kokkos-only program, initialize with `Kokkos::ScopeGuard` from `<Kokkos_Core.hpp>`. For a Tpetra program, use `Tpetra::ScopeGuard` from `<Tpetra_Core.hpp>`:

```cpp
int main(int argc, char* argv[]) {
    Tpetra::ScopeGuard runtime(&argc, &argv);
    auto comm = Tpetra::getDefaultComm();
    // Construct and use vectors, matrices, and Kokkos arrays here.
}
```

The Tpetra guard initializes MPI and Kokkos if necessary and finalizes what it initialized. Construct numerical objects after the guard so they are destroyed before finalization. With `Kokkos::ScopeGuard`, the constructor takes `argc, argv` without address operators.

## 2. Execution spaces and memory spaces

An **execution space** specifies where kernels run. A **memory space** specifies where array elements are stored.

```cpp
using Exec = Kokkos::DefaultExecutionSpace;
using Mem = Exec::memory_space;
using Device = Kokkos::Device<Exec, Mem>;
Exec exec;
```

`Exec`, `Mem`, and `Device` are types. `exec` is an execution-space object used when submitting work. `Device` combines an execution-space type and a memory-space type; it does not represent an allocated array.

| Execution space | Usual default memory space | Executes on |
|---|---|---|
| `Kokkos::Serial` | `Kokkos::HostSpace` | CPU, serial |
| `Kokkos::OpenMP` | `Kokkos::HostSpace` | CPU threads |
| `Kokkos::Cuda` | `Kokkos::CudaSpace` | NVIDIA GPU |
| `Kokkos::HIP` | `Kokkos::HIPSpace` | AMD GPU |

Only backends enabled in the Kokkos installation are available. `DefaultExecutionSpace` selects the installation's default backend. Shared and pinned memory spaces provide additional storage options.

An array descriptor can exist on the host while its elements are in GPU memory. Having its pointer does not make that memory accessible from the CPU. Use a host mirror when host access is required.

For generic code, accessibility can be checked at compile time:

```cpp
static_assert(Kokkos::SpaceAccessibility<Exec, Mem>::accessible);
```

An execution-space type does not identify one unique queue. A default-constructed instance does not necessarily create an independent GPU stream. Pass the same instance explicitly when operations must be ordered together.

The Kokkos examples below use `Exec`, `Mem`, and `exec` from this section, with `<Kokkos_Core.hpp>` included.

## 3. Kokkos::View

### Purpose and storage

A `View` describes a multidimensional array. It contains a pointer, dimension sizes, and the information needed to translate indices into memory addresses. Its type specifies the element type, layout, and memory space.

A managed view also shares ownership of its allocation. Copying the view copies this small descriptor and shares the array; it does not copy the elements.

```cpp
Kokkos::View<double*, Mem> temperature("temperature", 100);
auto same_array = temperature;  // Both views refer to the same 100 values.

Kokkos::View<double**, Kokkos::LayoutLeft, Mem> velocity("velocity", 100, 3);
```

In `double**`, the two stars specify two array dimensions. `velocity(i, c)` accesses row `i`, component `c`. `View<double*[3], ...>` instead fixes the second dimension to 3 at compile time. `View<double, Mem>` stores one scalar and is accessed with `view()`.

### Common members

| Member | Meaning |
|---|---|
| `view(i, j, ...)` | Access an element; the caller must be able to access its memory space |
| `extent(d)` | Number of elements along dimension `d` |
| `size()` | Product of all dimension sizes |
| `stride(d)` | Distance between adjacent elements along dimension `d`, in elements |
| `data()` | Pointer to the first element; dimensions and strides are not included |
| `span()` | Storage span in elements, including gaps or padding |
| `span_is_contiguous()` | Whether the view spans contiguous storage |
| `value_type`, `memory_space`, `execution_space` | Types describing the view |

Use `extent(0)` for the number of rows of a 2D array. `size()` counts all its elements.

### Allocation and initialization

An allocating constructor initializes elements; arithmetic elements become zero. Skip initialization only when every element will be written before it is read:

```cpp
Kokkos::View<double*, Mem> values(
    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "values"), 100);
Kokkos::deep_copy(exec, values, 2.0);  // Write every element.
```

`view_alloc` supplies allocation options: a label, an execution instance, and initialization policy. Passing `exec` also lets normal initialization be ordered on that instance. Uninitialized allocation does not zero the array.

### Read-only data

```cpp
Kokkos::View<const double*, Mem> read_only = temperature;
const auto writable_data = temperature;
```

`read_only` shares the allocation and prevents writes through that view. `writable_data` is a const descriptor whose elements remain writable. Put `const` on the element type to make the data read-only.

### Unmanaged views

An unmanaged view describes existing memory without owning it:

```cpp
using Borrowed = Kokkos::View<double*, Kokkos::HostSpace,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
std::vector<double> buffer(100);  // Include <vector>.
Borrowed values(buffer.data(), buffer.size());
values(0) = 3.0;
```

This constructor allocates nothing and copies no elements. Keep `buffer` alive and do not reallocate it while the view is used. The memory space in the view type must match the actual pointer. Wrapping a CPU pointer in a device view does not move the data.

For any view used by asynchronous work, keep its allocation alive until that work completes. This also applies when the last managed owner would otherwise be destroyed.

## 4. Layouts and subviews

### Layout

A layout determines which array elements are adjacent in memory.

| Layout | Adjacent elements in a 2D array `a(i, j)` |
|---|---|
| `Kokkos::LayoutLeft` | `a(i, j)` and `a(i + 1, j)` |
| `Kokkos::LayoutRight` | `a(i, j)` and `a(i, j + 1)` |
| `Kokkos::LayoutStride` | Defined by explicit strides for each dimension |

For an array indexed as `(entity, component)`, `LayoutLeft` places neighboring entities of one component together. `LayoutRight` places the components of one entity together. Array padding can affect the other dimension's stride; query it instead of assuming a packed layout.

### Kokkos::subview

`subview` selects part of an existing array. It returns another view of the same storage and allocates no element buffer.

```cpp
Kokkos::View<double**, Kokkos::LayoutLeft, Mem> a("a", 100, 3);
auto component = Kokkos::subview(a, Kokkos::ALL(), 1);
auto entity = Kokkos::subview(a, 4, Kokkos::ALL());
auto range = Kokkos::subview(a, Kokkos::pair<int, int>{10, 20}, Kokkos::ALL());
```

`component(i)` aliases `a(i, 1)`. `entity(c)` aliases `a(4, c)`. `range` contains rows 10 through 19. An integer argument removes that dimension; `ALL()` or a half-open range keeps it.

A subview can be strided. In this example, `entity` generally has a stride greater than one. Passing `entity.data()` to a routine expecting a packed array loses that stride information. Use `auto` for subview types because their layout can differ from the parent's.

### Explicit LayoutStride

The arguments are `(extent, stride)` pairs. For a host array containing three components per entity, this view selects component 1:

```cpp
std::vector<double> packed(30);  // Ten entities, three values each.
using Strided = Kokkos::View<double*, Kokkos::LayoutStride, Kokkos::HostSpace,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
Strided component(packed.data() + 1, Kokkos::LayoutStride(10, 3));
// component(i) refers to packed[3*i + 1].
```

## 5. Mirrors and data copies

A **mirror** is a view with compatible dimensions and layout in another memory space. Creating a mirror and copying values are separate operations.

| Function | Allocates? | Copies values? |
|---|---|---|
| `create_mirror(view)` | Always creates a separate host-accessible allocation | No |
| `create_mirror_view(view)` | Only if host access requires another allocation | No |
| `create_mirror_view_and_copy(HostSpace{}, view)` | If the requested memory space differs | Yes, when a new allocation is made; otherwise aliases |
| `deep_copy(destination, source)` | No | Yes |
| `deep_copy(destination, scalar)` | No | Fills the destination |

```cpp
Kokkos::View<double*, Mem> device_values("device_values", 4);
auto host_values = Kokkos::create_mirror(device_values);
for (int i = 0; i < 4; ++i) host_values(i) = i + 1.0;
Kokkos::deep_copy(device_values, host_values);  // Host -> device.
Kokkos::deep_copy(host_values, device_values);  // Device -> host.
```

Both allocations already exist before `deep_copy`. A `create_mirror_view` result can alias the original array in a CPU build; use `create_mirror` when independent storage is required.

`Kokkos::deep_copy(dst, src)` waits for outstanding Kokkos operations across execution instances before copying, and the copy is complete when the call returns. Kokkos performs this synchronization explicitly; it does not depend on the default CUDA stream's implicit synchronization behavior.

`Kokkos::deep_copy(exec, dst, src)` orders the copy after previously submitted work on `exec` and before subsequently submitted work on that same instance. It can return before the copy completes. Do not rely on it to wait for producer work on other execution instances; establish those dependencies explicitly.

Use `exec.fence()` when the calling host thread must wait for the copy to finish. Keep the source and destination allocations alive until completion.

For view-to-view copies, ranks and dimension sizes must match. Changing layout or element type requires a copy execution space that can access both arrays. A transfer between otherwise inaccessible memory spaces requires compatible contiguous storage with matching layout and element type. To transfer and change layout, first transfer to a compatible mirror, then convert where both arrays are accessible.

## 6. Parallel kernels

### parallel_for and RangePolicy

`parallel_for` runs a callable over an index range. Iteration order is unspecified; iterations must not depend on one another's execution order.

```cpp
const int count = 100;
Kokkos::View<double*, Mem> values("values", count);
Kokkos::parallel_for("initialize", Kokkos::RangePolicy<Exec>(exec, 0, count),
    KOKKOS_LAMBDA(const int i) {
        values(i) = 0.5 * i;
    });
```

The range is `[0, count)`. The label is for profiling and diagnostics. `KOKKOS_LAMBDA` captures by value and makes the callable usable by the enabled backends. Capturing `values` copies the descriptor; the kernel writes the original allocation.

For ranges needing a wider index type, use `RangePolicy<Exec, Kokkos::IndexType<std::size_t>>` and a matching lambda argument type.

### Objects inside kernels

Kernel code needs device-callable functions and memory accessible to the execution space. A small user-defined struct can be passed by value if its members and functions meet those requirements:

```cpp
struct Scale {
    double factor;
    KOKKOS_INLINE_FUNCTION double operator()(double x) const { return factor * x; }
};
Scale scale{2.0};
Kokkos::parallel_for("scale", Kokkos::RangePolicy<Exec>(exec, 0, count),
    KOKKOS_LAMBDA(const int i) { values(i) = scale(values(i)); });
```

Use `KOKKOS_INLINE_FUNCTION` for small functions called from kernels. Use `Kokkos::sqrt`, `Kokkos::abs`, and similar math functions, and `Kokkos::Array<T, N>` for fixed-size arrays.

Ordinary `Tpetra::Map`, `Tpetra::CrsMatrix`, `Teuchos::RCP`, and `std::vector` objects are host-side interfaces. Obtain their appropriate local views or descriptors before launching a kernel. Accessing a host object through a captured pointer does not make it usable on the GPU. Inside class member functions, copy the needed views and scalar members into local variables before capturing them.

### Multidimensional and team policies

`MDRangePolicy` describes a multidimensional index range:

```cpp
Kokkos::View<double**, Mem> a("a", 100, 3);
using Policy2D = Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>>;
Kokkos::parallel_for("fill_2d", Policy2D(exec, {0, 0}, {100, 3}),
    KOKKOS_LAMBDA(const int i, const int c) { a(i, c) = i + c; });
```

`TeamPolicy` assigns each league item to a team. `TeamThreadRange` distributes an inner loop among that team's threads:

```cpp
using Teams = Kokkos::TeamPolicy<Exec>;
Kokkos::parallel_for("fill_by_team", Teams(exec, 100, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Teams::member_type& team) {
        const int i = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 3),
            [&](const int c) { a(i, c) = i + c; });
    });
```

The nested lambda executes within the outer kernel. All team threads must participate in the same collective team operation. `team.team_barrier()` synchronizes members of that team; it does not synchronize different teams.

## 7. Reductions, scans, and atomics

### parallel_reduce

A reduction combines contributions into one result. Each iteration receives a private partial accumulator, and Kokkos combines the partial results.

```cpp
double sum = 0.0;
Kokkos::parallel_reduce("sum", Kokkos::RangePolicy<Exec>(exec, 0, count),
    KOKKOS_LAMBDA(const int i, double& partial) { partial += values(i); }, sum);
```

For the default sum reduction, Kokkos initializes partial accumulators to zero. A result passed as a host scalar is ready when the call returns. This reduction combines work within one MPI process; use an MPI reduction or a Tpetra norm for a global result.

For a minimum, provide a reducer with the correct initial value and combining operation:

```cpp
double minimum;
Kokkos::parallel_reduce("minimum", Kokkos::RangePolicy<Exec>(exec, 0, count),
    KOKKOS_LAMBDA(const int i, double& partial) {
        if (values(i) < partial) partial = values(i);
    }, Kokkos::Min<double>(minimum));
```

`Kokkos::Max<T>` and `Kokkos::Sum<T>` provide the corresponding operations. A reduction into a rank-zero device view can remain asynchronous; wait before reading its result from another execution context.

### parallel_scan

A scan computes prefix results. For input `[2, 3, 1]`, an exclusive sum scan produces `[0, 2, 5]`.

```cpp
Kokkos::View<int*, Mem> counts("counts", count);
Kokkos::View<int*, Mem> offsets("offsets", count + 1);
// Fill counts before this launch, using the same exec or an explicit fence.
Kokkos::parallel_scan("offsets", Kokkos::RangePolicy<Exec>(exec, 0, count + 1),
    KOKKOS_LAMBDA(const int i, int& prefix, const bool final) {
        if (final) offsets(i) = prefix;
        if (i < count) prefix += counts(i);
    });
```

Here `offsets(count)` stores the total. Kokkos can invoke the callable multiple times; write output only when `final` is true. The accumulation itself must run on every invocation.

### Atomic updates

An atomic operation protects one memory location against simultaneous updates:

```cpp
// Inside a kernel, with a valid target index:
Kokkos::atomic_add(&values(target), contribution);
```

Use it when multiple iterations add to the same entry. The target must already contain a valid value, usually zero before assembly. An atomic operation does not wait for other iterations or kernels. Floating-point addition order can vary, so the last bits of a result can vary too.

## 8. Asynchronous execution and fences

Kernel launches can return before the kernel finishes. Work submitted to the same execution instance is ordered:

```cpp
Kokkos::deep_copy(exec, values, 0.0);
Kokkos::parallel_for("add_one", Kokkos::RangePolicy<Exec>(exec, 0, count),
    KOKKOS_LAMBDA(const int i) { values(i) += 1.0; });
exec.fence("values ready");
```

The kernel sees the completed zero fill because both operations use `exec`. The fence waits for earlier work on that instance.

| Operation | Completion behavior |
|---|---|
| `parallel_for` | Can return before completion |
| `parallel_reduce` with a host scalar result | Result is ready on return |
| `deep_copy(dst, src)` | Waits for outstanding Kokkos work before copying; copy is complete on return |
| `deep_copy(exec, dst, src)` | Can return before completion |
| `exec.fence()` | Waits for work on that instance |
| `Kokkos::fence()` | Waits for outstanding Kokkos work across execution instances |

Fence before handing data produced by a custom kernel to a library operation that does not use the same execution instance, before host inspection, or before releasing its allocation. A fence completes work; it does not copy device memory to the host.

MPI and Kokkos track different operations. `MPI_Wait` completes an MPI request; it does not establish completion of a preceding Kokkos producer kernel. For portable host-staged communication, finish the device-to-host copy before MPI reads the buffer, and finish the MPI receive before copying that buffer back to the device.

## 9. Kokkos::DualView

**Header:** `<Kokkos_DualView.hpp>`.

A `DualView` holds corresponding host and device views plus flags recording which side was modified. In a CPU build, both views can share an allocation.

| Method | Action |
|---|---|
| `view_host()`, `view_device()` | Return the corresponding Kokkos view |
| `modify_host()`, `modify_device()` | Mark that side as modified; no copy |
| `sync_host()`, `sync_device()` | Update that side from the modified side when needed |
| `need_sync_host()`, `need_sync_device()` | Report whether an update is needed |

```cpp
Kokkos::DualView<double*, Exec> values("values", 100);
auto host = values.view_host();
for (int i = 0; i < 100; ++i) host(i) = i;
values.modify_host();
values.sync_device();
auto device = values.view_device();  // Use in a device kernel.
```

Writing through a returned view does not automatically mark the side as modified. After a device kernel, complete its writes before synchronizing to the host, and mark the device side as modified. Synchronize before modifying the other side so that there are no conflicting changes.

Tpetra manages its own host/device state through access-tagged methods described in Section 15. Those methods handle the modification bookkeeping for the requested access.

## 10. Teuchos utilities

### Teuchos::RCP

**Header:** `<Teuchos_RCP.hpp>`.

`RCP<T>` is a reference-counted pointer. Trilinos uses it to share ownership of objects.

```cpp
auto parameters = Teuchos::rcp(new Teuchos::ParameterList);
auto same_object = parameters;  // Shared ownership; no object copy.
Teuchos::RCP<const Teuchos::ParameterList> read_only = parameters;
```

| Operation | Meaning |
|---|---|
| `pointer->method()`, `*pointer` | Access the object |
| `pointer.is_null()` | Test for an empty pointer |
| `pointer.getRawPtr()` | Get a borrowed pointer; does not extend lifetime |
| `Teuchos::rcp_dynamic_cast<T>(pointer, true)` | Checked polymorphic cast; throws on failure |
| `Teuchos::null` | Empty RCP value |

The object is destroyed when its last owning RCP is released. `RCP<const T>` gives read-only object access; a const `RCP<T>` still points to a mutable object. Do not create a second owning RCP from a raw pointer already owned by another RCP.

### Array and ArrayView

**Headers:** `<Teuchos_Array.hpp>`, `<Teuchos_ArrayView.hpp>`.

`Array<T>` owns a resizable host array. `ArrayView<T>` describes a borrowed host array and its length.

```cpp
Teuchos::Array<int> indices(3);
indices[0] = 2; indices[1] = 5; indices[2] = 8;
Teuchos::ArrayView<const int> view = indices();
```

`indices()` returns a view of the whole array. `view[i]` accesses an element; `view.size()` gives its length. An ArrayView neither owns the elements nor copies them to a GPU. Keep the source array alive while its view is used.

### ParameterList

**Header:** `<Teuchos_ParameterList.hpp>`.

A `ParameterList` stores named, typed values and nested lists. Solvers and preconditioners use it for configuration.

```cpp
Teuchos::ParameterList options;
options.set("Maximum Iterations", 300);
options.set("Convergence Tolerance", 1.0e-8);
int limit = options.get<int>("Maximum Iterations");
```

`set(name, value)` assigns a parameter. `get<T>(name)` reads it and checks its type. `get<T>(name, default_value)` supplies a default for a missing entry. `isParameter(name)` tests existence. `sublist(name)` accesses or creates a nested list.

Names, capitalization, and value types must match the receiving component's interface. A Belos parameter is not automatically valid for Ifpack2 or MueLu.

### Comm

`Teuchos::Comm<int>` is a communicator interface. `Tpetra::getDefaultComm()` returns an `RCP<const Teuchos::Comm<int>>` for the default communicator. `comm->getRank()` returns the process rank and `comm->getSize()` returns the process count. Pass this communicator to Tpetra maps so their objects use the same process group.

## 11. Tpetra types

Tpetra objects specify their index types and execution/memory configuration through template parameters:

```cpp
Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>
Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>
```

| Parameter | Meaning |
|---|---|
| `Scalar` | Type of matrix coefficients and vector elements |
| `LocalOrdinal` | Type of indices within one process |
| `GlobalOrdinal` | Type of global identifiers |
| `Node` | Tpetra adapter specifying Kokkos execution and memory spaces |

### Explicit configuration

The following example selects CUDA explicitly. Kokkos types are obtained from the selected Node so that Kokkos arrays and Tpetra objects use the same configuration.

```cpp
#include <Kokkos_Core.hpp>
#include <Tpetra_KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

using Scalar = double;
using LO = int;
using GO = long long;

// Explicitly select the CUDA Node provided by this Trilinos build.
using Node = Tpetra::KokkosCompat::KokkosCudaWrapperNode;

// Use this same configuration in Kokkos.
using Exec   = Node::execution_space;
using Mem    = Node::memory_space;
using Device = Node::device_type;

// Specify every Tpetra template parameter explicitly.
using Map         = Tpetra::Map<LO, GO, Node>;
using Graph       = Tpetra::CrsGraph<LO, GO, Node>;
using Matrix      = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
using Vector      = Tpetra::Vector<Scalar, LO, GO, Node>;
using MultiVector = Tpetra::MultiVector<Scalar, LO, GO, Node>;
using Operator    = Tpetra::Operator<Scalar, LO, GO, Node>;

// Names used in the following sections.
using TpetraExec = Exec;
using TpetraMem  = Mem;
```

To select another backend, change the `Node` alias:

| Backend | Node type |
|---|---|
| Serial | `Tpetra::KokkosCompat::KokkosSerialWrapperNode` |
| OpenMP | `Tpetra::KokkosCompat::KokkosOpenMPWrapperNode` |
| CUDA | `Tpetra::KokkosCompat::KokkosCudaWrapperNode` |

The selected backend must be enabled in both Kokkos and Tpetra.

### What Node represents

A Node is a specialization of `Tpetra::KokkosCompat::KokkosDeviceWrapperNode<ExecutionSpace, MemorySpace>`. It provides three type aliases:

- `execution_space`: where kernels execute.
- `memory_space`: where numerical data is stored.
- `device_type`: the corresponding `Kokkos::Device<execution_space, memory_space>`.

Node is a type, not an execution instance. It does not create an execution queue or store a CUDA stream.

The named CUDA Node uses the memory space selected by the Trilinos build: `CudaSpace`, or `CudaUVMSpace` when `Tpetra_ALLOCATE_IN_SHARED_SPACE=ON`. Taking `Mem` and `Device` from Node preserves that choice.

After runtime initialization, matching Kokkos objects can be constructed as follows:

```cpp
Exec exec;
Kokkos::View<Scalar*, Device> values("values", 100);

Kokkos::parallel_for(
    "initialize",
    Kokkos::RangePolicy<Exec>(exec, 0, 100),
    KOKKOS_LAMBDA(const int i) {
        values(i) = 1.0;
    });

exec.fence();
```

### Supported template combinations

Trilinos commonly precompiles template combinations through **explicit template instantiation (ETI)**. The selected combination of `Scalar`, `LO`, `GO`, and `Node` must be available in the installation. Enabling a Kokkos backend alone does not necessarily enable the corresponding Tpetra instantiations.

## 12. Tpetra::Map

### Purpose and storage

A `Map` describes how global identifiers are distributed across processes and how they are numbered locally. It stores distribution metadata and, where needed, lookup tables. It stores no vector values or matrix coefficients.

| Process | Global IDs present here | Local IDs |
|---|---|---|
| 0 | `[0, 3, 7]` | `[0, 1, 2]` |
| 1 | `[1, 2, 4]` | `[0, 1, 2]` |

On process 0, local ID 1 means global ID 3. On process 1, local ID 1 means global ID 2. Local indices start at zero even when global identifiers use another base.

A **one-to-one map** contains each global identifier on exactly one process. An **overlapping map** can contain the same identifier on several processes, for example when each process needs copies of neighboring values.

### Construction

For 1000 consecutively numbered entries distributed approximately evenly:

```cpp
auto map = Teuchos::rcp(new Map(Tpetra::global_size_t{1000}, GO{0}, comm));
```

For an explicit list of global IDs on each process, assume that
`owned_gids` is a `std::vector<GO>` already filled by the application.

```cpp
// Describe the existing host array without copying its elements.
const Teuchos::ArrayView<const GO> local_ids(
    owned_gids.data(), owned_gids.size());

// Ask Tpetra to compute the global count.
const auto automatic_count =
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

auto map = Teuchos::rcp(new Map(
    automatic_count,
    local_ids,
    GO{0},  // Zero-based global numbering.
    comm));
```

Each process supplies its own `owned_gids`. The array order defines local
numbering: `owned_gids[i]` receives local index `i`. For example, a process
with IDs `[7, 42, 105]` maps local indices `0, 1, 2` to those global IDs.

`local_ids` is a borrowed view of the CPU array. The Map constructor builds
its indexing structures from this list.

The invalid global-count value tells Tpetra to sum the local list sizes
across all processes. For a one-to-one map, this is the total number of
unique global IDs. For an overlapping map, repeated IDs on different
processes are counted separately.

Map construction is collective: every process in `comm` must participate, including processes with no local entries.

### Common methods

| Method | Returns |
|---|---|
| `getLocalNumElements()` | Number of entries on this process |
| `getGlobalNumElements()` | Global entry count recorded by the map |
| `getGlobalElement(local_id)` | Global ID at this local position |
| `getLocalElement(global_id)` | Local position, if the ID is present on this process |
| `getLocalElementList()` | Host view of the local global-ID list, in local order |
| `getLocalMap()` | Kernel-compatible local lookup object; see Section 13 |
| `isOneToOne()` | Whether every global ID appears on exactly one process |
| `isSameAs(other)` | Whether the maps describe the same distribution and local ordering |

An unsuccessful lookup returns `Teuchos::OrdinalTraits<LO>::invalid()` or the corresponding `GO` value. `getLocalElement` does not search another process. Operations that determine a global map property, such as `isOneToOne()` and `isSameAs()`, can communicate and must be called consistently across the communicator.

## 13. What getLocalMap() returns

`Tpetra::Map` is a host-side object with distributed operations. Its `getLocalMap()` method returns a smaller object of type `Map::local_map_type`, by value.

In Trilinos 17.2.1 this type is `Tpetra::Details::LocalMap<LO, GO, Map::device_type>`. It contains local indexing metadata and Kokkos-based lookup structures. A contiguous map can use arithmetic instead of storing every identifier.

The returned object is suitable for capture in kernels running in `Map::execution_space`. It exposes local lookups with device-callable methods. It performs no MPI communication and has no way to look up ownership on another process.

| Object | Used for | In a Kokkos device kernel? |
|---|---|---|
| `Teuchos::RCP<const Map>` | Sharing a distributed map in host code | No |
| `Map` | Construction, distribution queries, host lookups | Use its local descriptor instead |
| `Map::local_map_type` | Local index conversion and local metadata | Yes, in the matching execution space |

```cpp
const auto local_map = map->getLocalMap();
const LO count = static_cast<LO>(map->getLocalNumElements());
Kokkos::View<GO*, TpetraMem> global_ids("global_ids", count);
TpetraExec exec;
Kokkos::parallel_for("local_to_global",
    Kokkos::RangePolicy<TpetraExec, Kokkos::IndexType<LO>>(exec, 0, count),
    KOKKOS_LAMBDA(const LO i) {
        global_ids(i) = local_map.getGlobalElement(i);
    });
exec.fence();
```

The host obtains `local_map` before the launch. The kernel receives a copy of that small descriptor and uses its device-accessible lookup data. The call does not copy a distributed map or its complete data onto the GPU at each iteration.

Common methods are `getGlobalElement(local_id)`, `getLocalElement(global_id)`, `getLocalNumElements()`, `getIndexBase()`, and `isContiguous()`. Failed lookups return invalid ordinals as for the full map. A local lookup can find an overlapping entry if that entry is present in this map; “local” does not necessarily mean uniquely owned.

Keep the original map alive until kernels finish. The `Details::LocalMap` interface is version-dependent; obtain it through `getLocalMap()` and use `auto` rather than constructing it directly.

## 14. Import and Export

**Headers:** `<Tpetra_Import.hpp>`, `<Tpetra_Export.hpp>`.

An `Import` or `Export` stores a reusable plan for moving entries between two maps. It contains local permutations and communication information, not the numerical vector values themselves.

- `Import` commonly gathers uniquely owned values into an overlapping distribution.
- `Export` commonly combines overlapping contributions into an owned distribution.

```cpp
using Import = Tpetra::Import<LO, GO, Node>;
Import plan(owned_map, overlap_map);  // Source map, then target map.
Vector owned(owned_map);
Vector overlap(overlap_map);
owned.putScalar(1.0);
overlap.doImport(owned, plan, Tpetra::INSERT);
```

Here `owned_map` is one-to-one, and `overlap_map` lists the IDs needed locally, including remote IDs. After the import, `overlap` contains the corresponding owned values. `INSERT` is appropriate because the source value for each ID is unique.

To add contributions in the reverse direction:

```cpp
using Export = Tpetra::Export<LO, GO, Node>;
Export plan(overlap_map, owned_map);
owned.putScalar(0.0);
owned.doExport(overlap, plan, Tpetra::ADD);
```

`ADD` adds received contributions to existing destination values. Construct plans and perform transfers consistently across the communicator. Reuse a plan while its maps remain unchanged. Ordinary distributed matrix `apply()` already performs its required transfers internally.

## 15. Vector and MultiVector

### Purpose and common operations

A `Vector` stores one distributed dense column. A `MultiVector` stores several columns with the same row distribution. A multivector with three columns represents three vectors; it does not automatically represent three components at each row of a block matrix.

```cpp
auto x = Teuchos::rcp(new Vector(map));
x->putScalar(1.0);
auto multiple_rhs = Teuchos::rcp(new MultiVector(map, 3));
```

Constructors initialize entries to zero by default. An optional `zeroOut=false` skips initialization and requires a full write before reading.

| Method | Action or result |
|---|---|
| `getMap()` | Row distribution |
| `getLocalLength()`, `getGlobalLength()` | Local and global row counts |
| `getNumVectors()` | Number of columns; 1 for Vector |
| `putScalar(a)` | Fill all entries with `a` |
| `scale(a)` | Multiply entries by `a` |
| `update(a, other, b)` | Set `this = a*other + b*this` |
| `Vector::dot(other)` | Global inner product |
| `Vector::norm1()`, `norm2()`, `normInf()` | Global norms |
| `getLocalViewHost(access)` | Local 2D Kokkos view in host-accessible memory |
| `getLocalViewDevice(access)` | Local 2D Kokkos view in the computational memory space |

Vector dot products and norms include MPI reductions. MultiVector norm and dot overloads return one result per column through output arrays. Arithmetic operands need compatible maps and column counts.

`Tpetra::deep_copy(destination, source)` copies multivector values into existing storage. Copying an `RCP` shares the same object; it does not copy its values.

### Local views and access tags

`getLocalViewDevice()` returns a Kokkos view of local numerical storage, not a new distributed vector. Even a Vector returns a rank-two view: `local(row, 0)`. A MultiVector uses `local(row, column)`.

| Access tag | Meaning |
|---|---|
| `Tpetra::Access::ReadOnly` | Read existing values; return a const-element view |
| `Tpetra::Access::ReadWrite` | Read and modify existing values |
| `Tpetra::Access::OverwriteAll` | Write every exposed value; previous contents need not be preserved |

Access can synchronize host/device representations. `OverwriteAll` does not initialize or zero anything. `+=` needs valid previous contents and normally uses `ReadWrite`.

```cpp
TpetraExec exec;
{
    auto local = x->getLocalViewDevice(Tpetra::Access::OverwriteAll);
    const auto local_map = x->getMap()->getLocalMap();
    const LO rows = static_cast<LO>(x->getLocalLength());
    Kokkos::parallel_for("fill_vector",
        Kokkos::RangePolicy<TpetraExec, Kokkos::IndexType<LO>>(exec, 0, rows),
        KOKKOS_LAMBDA(const LO i) { local(i, 0) = local_map.getGlobalElement(i); });
    exec.fence();
}  // Release the device access view before requesting host access.
{
    auto local = x->getLocalViewHost(Tpetra::Access::ReadOnly);
    // Host code may inspect local(i, 0) here.
}
```

Keep local access scopes short. Finish custom kernels before calling distributed operations on the same vector. Release host views before obtaining device views, and device views before obtaining host views. Keep the parent vector alive throughout.

`MultiVector::getVector(j)` and `getVectorNonConst(j)` return aliases for one column. Selected-column multivector views can have nonconstant stride: their raw local storage does not necessarily describe only the selected columns. Use column-aware Tpetra operations or individual column vectors in that case.

## 16. The four matrix maps

For an operator `y = A*x`, the maps have different roles:

| Map | Meaning |
|---|---|
| Row map | Rows stored on each process |
| Column map | Columns referenced by those local rows, in local column order |
| Domain map | Distribution expected for input vector `x` |
| Range map | Distribution expected for output vector `y` |

For a typical square row-owned matrix, row, domain, and range maps coincide. The column map additionally includes remote columns needed by the local rows.

Domain and range maps supplied to `fillComplete` must be one-to-one.

For example, if a process stores rows 0 and 1 of a tridiagonal matrix, it may need input entries 0, 1, and 2. Its row map then contains two entries, while its column map contains three.

```cpp
auto x = Teuchos::rcp(new Vector(matrix->getDomainMap()));
auto y = Teuchos::rcp(new Vector(matrix->getRangeMap()));
matrix->apply(*x, *y);
```

Tpetra imports remote input entries needed for multiplication and exports results if required by the row/range distributions. Referencing remote columns does not create complete remote matrix rows.

A local row index belongs to the row map; a local column index belongs to the column map. Equal integers in these two maps need not represent the same global ID. Use the corresponding map to convert each one.

## 17. Tpetra::CrsGraph

### Purpose and storage

A `CrsGraph` stores a sparse matrix's structure: row offsets, column indices, and associated maps. It has no numerical coefficients. CRS and CSR both mean compressed sparse row storage.

For local row `i`, entries occupy `[row_offsets(i), row_offsets(i+1))`. For example, offsets `[0, 2, 5]` describe two rows with 2 and 3 stored entries.

```cpp
auto graph = Teuchos::rcp(new Graph(map, 1));  // Capacity: one entry per row.
for (std::size_t i = 0; i < map->getLocalNumElements(); ++i) {
    const GO gid = map->getGlobalElement(static_cast<LO>(i));
    Teuchos::Array<GO> columns(1);
    columns[0] = gid;
    graph->insertGlobalIndices(gid, columns());
}
graph->fillComplete(map, map);  // Domain map, then range map.
```

This constructs a diagonal graph. The constructor's per-row capacity reserves storage; it does not insert entries.

| Method | Action or result |
|---|---|
| `insertGlobalIndices(row_gid, column_gids)` | Insert structural entries using global IDs |
| `fillComplete(domain, range)` | Finalize structure for computation |
| `isFillComplete()` | Whether finalization is complete |
| `getRowMap()`, `getColMap()` | Distributed row and column maps |
| `getLocalNumRows()`, `getLocalNumEntries()` | Local row and stored-entry counts |
| `getLocalGraphDevice()`, `getLocalGraphHost()` | Local CSR descriptors for the selected memory side |

`fillComplete` is collective. It builds communication information, constructs a column map if needed, converts column IDs to local indices, and sorts/merges entries. The resulting entry order need not match insertion order. Repeated structural indices become a single graph entry.

After finalization, a local graph's column indices refer to its column map. Its `row_map` member is a Kokkos array of CSR offsets; it is unrelated to the distributed Map returned by `getRowMap()`.

## 18. Tpetra::CrsMatrix

### Purpose and assembly

A `CrsMatrix` stores a distributed sparse matrix: its graph, numerical coefficients, maps, and communication plans for operations such as matrix-vector multiplication.

One construction path inserts structure and values together:

```cpp
auto matrix = Teuchos::rcp(new Matrix(map, 1));
for (std::size_t i = 0; i < map->getLocalNumElements(); ++i) {
    const GO gid = map->getGlobalElement(static_cast<LO>(i));
    Teuchos::Array<GO> columns(1, gid);
    Teuchos::Array<Scalar> values(1, 2.0);
    matrix->insertGlobalValues(gid, columns(), values());
}
matrix->fillComplete(map, map);
```

This creates `A = 2I`. Before finalization, the matrix accepts inserted entries. `fillComplete` prepares it for `apply`; duplicate inserted entries are summed. Every process participates in finalization, including those with no local rows.

Another path reuses an already finalized graph:

```cpp
Teuchos::RCP<const Graph> fixed_graph = graph;
auto matrix = Teuchos::rcp(new Matrix(fixed_graph));
matrix->setAllToScalar(2.0);
matrix->fillComplete(graph->getDomainMap(), graph->getRangeMap());
```

Here the coefficient storage belongs to the new matrix, while its graph can be shared. A fixed graph allows changes to existing coefficients, not insertion of new structural entries. Two separately constructed matrices can share the same graph and have independent coefficients.

### Common methods

| Method | Action or result |
|---|---|
| `insertGlobalValues(row, columns, values)` | Insert global entries during assembly |
| `replaceGlobalValues(row, columns, values)` | Replace coefficients at existing global entries |
| `sumIntoGlobalValues(row, columns, values)` | Add to coefficients at existing global entries |
| `resumeFill()` | Re-enter assembly state for insertion/replacement APIs |
| `fillComplete(domain, range)` | Finalize assembly and prepare distributed operations |
| `setAllToScalar(a)` | Assign `a` to existing coefficients |
| `apply(x, y)` | Compute distributed `y = A*x` |
| `getCrsGraph()` | RCP to the graph |
| `getLocalValuesDevice(access)` | Rank-one local coefficient view |
| `getLocalMatrixDevice()` | Local sparse matrix descriptor; see Section 19 |

The insertion/replacement APIs have fill-state requirements. For repeated assembly through those APIs, use `resumeFill()`, modify entries, then `fillComplete()` collectively. A fixed graph remains structurally fixed. Direct access to finalized numerical storage is a separate path and does not require refinalizing an unchanged graph.

`getLocalNumEntries()` counts stored slots, including zeros and couplings from local rows to remote columns.

## 19. Local matrix and graph access

### What each accessor returns

| Accessor | Result | Contents |
|---|---|---|
| `matrix->getCrsGraph()` | RCP to a distributed Tpetra graph | Structure, maps, communication information |
| `graph->getLocalGraphDevice()` | Local CSR graph descriptor | Kokkos views `row_map` and `entries` |
| `matrix->getLocalValuesDevice(access)` | Rank-one Kokkos view | Coefficients, one scalar per stored entry |
| `matrix->getLocalMatrixDevice()` | `Matrix::local_matrix_device_type` | Local graph and coefficient views together |

The local matrix type is a `KokkosSparse::CrsMatrix` specialization. It describes the matrix rows stored on this process and can be passed into kernels using its execution space. It does not implement MPI imports or exports. Use `Tpetra::CrsMatrix::apply` for distributed multiplication.

The local accessors share numerical storage; they do not create an independent coefficient array. Access may synchronize host/device representations. Obtain local matrix data after finalization, and do not retain descriptors across structural changes. Keep the matrix alive and finish custom kernels before changing access side or starting distributed operations on it.

### Local CSR indexing

For `auto local = matrix->getLocalMatrixDevice()`:

| Expression | Meaning |
|---|---|
| `local.graph.row_map(i)` | Beginning offset of local row `i` |
| `local.graph.row_map(i + 1)` | End offset of that row |
| `local.graph.entries(k)` | Local column ID at stored position `k` |
| `local.values(k)` | Numerical coefficient at stored position `k` |

The same position `k` selects matching entries in the column-index and coefficient arrays. It is neither a row ID nor a column ID. To obtain a global column ID, convert `entries(k)` through the matrix's column map.

### Example: modify the diagonal in a kernel

This assumes a finalized matrix with existing diagonal entries:

```cpp
TpetraExec exec;
{
    const auto graph = matrix->getCrsGraph()->getLocalGraphDevice();
    auto values = matrix->getLocalValuesDevice(Tpetra::Access::ReadWrite);
    const auto rows = matrix->getRowMap()->getLocalMap();
    const auto columns = matrix->getColMap()->getLocalMap();
    const LO count = static_cast<LO>(matrix->getLocalNumRows());
    Kokkos::parallel_for("shift_diagonal",
        Kokkos::RangePolicy<TpetraExec, Kokkos::IndexType<LO>>(exec, 0, count),
        KOKKOS_LAMBDA(const LO i) {
            const GO row_gid = rows.getGlobalElement(i);
            for (auto k = graph.row_map(i); k < graph.row_map(i + 1); ++k) {
                if (columns.getGlobalElement(graph.entries(k)) == row_gid)
                    values(k) += 1.0;
            }
        });
    exec.fence();
}
```

Each iteration updates a different row, so this example needs no atomics. If several iterations can update the same coefficient, arrange exclusive writes or use atomic additions.

In Trilinos 17.2.1, `getLocalMatrixDevice()` obtains read/write coefficient access even though the accessor is a const member function. Use `getLocalValuesDevice(ReadOnly)` plus a local graph when only reads are intended. The coefficient accessor also supports `OverwriteAll` for a full replacement, without zeroing the buffer automatically.

## 20. Tpetra::BlockCrsMatrix

**Header:** `<Tpetra_BlockCrsMatrix.hpp>`.

### Purpose and storage

A `BlockCrsMatrix` stores one dense square block at each graph entry. All blocks have the same size `B`. For example, a block row with two entries and `B = 3` stores two dense 3-by-3 blocks: 18 scalar coefficients and two column indices.

The graph counts block rows and block columns. Solver vectors use **point maps**, where each block entry expands to `B` scalar unknowns. With zero-based global IDs, the usual expansion is `point_gid = block_gid * B + component`; local ordering groups a block's components together.

```cpp
using BlockMatrix = Tpetra::BlockCrsMatrix<Scalar, LO, GO, Node>;
const LO block_size = 3;
auto blocks = Teuchos::rcp(new BlockMatrix(*graph, block_size));
blocks->setAllToScalar(0.0);
Vector x(blocks->getDomainMap());
Vector y(blocks->getRangeMap());
```

`graph` must already be fill-complete and describe block couplings. This constructor computes the point domain/range maps. A four-argument constructor accepts an existing graph, point domain map, point range map, and block size. Block matrices do not use a subsequent CSR-style `fillComplete()` call.

### Common methods and units

| Method | Result or action |
|---|---|
| `getBlockSize()` | Number of scalar components per block |
| `getRowMap()`, `getColMap()` | Block row/column maps |
| `getDomainMap()`, `getRangeMap()` | Scalar point maps for solver vectors |
| `getLocalNumRows()`, `getLocalNumEntries()` | Block-row count and stored-block count |
| `setAllToScalar(a)` | Fill all scalar coefficients |
| `replaceLocalValues(row, columns, values, count)` | Replace existing blocks, using block-local indices |
| `sumIntoLocalValues(row, columns, values, count)` | Add to existing blocks |
| `getValuesDevice()` | Read-only rank-one view of scalar coefficients |
| `getValuesDeviceNonConst()` | Mutable rank-one coefficient view |
| `apply(x, y)` | Multiply using point-map vectors |

The pointer-based replacement methods take consecutive dense blocks in row-major input order. Check the returned block count to confirm all requested entries were found. Native block storage can use a build-dependent layout; do not infer it from that input convention.

### Access one block on the host

```cpp
const LO local_row = 0;     // Assumes this process has row 0.
const LO local_column = 0;  // Must identify an existing block in that row.
{
    auto block = blocks->getLocalBlockHostNonConst(local_row, local_column);
    block(0, 0) = 2.0;
    block(0, 1) = -1.0;
}
```

This returns a 2D view into one existing block, without allocating coefficients. It uses local row and column maps separately. The `little_block_type` and `little_block_host_type` aliases describe the native block view types and their layouts.

The flat device coefficient view has `local_block_count * B * B` scalar entries. A graph offset selects a block, so it cannot directly index that scalar buffer without accounting for block size and layout. `getValuesDeviceNonConst()` uses read/write access; unlike CSR, it has no `OverwriteAll` argument in this version.

BSR storage, multivector columns, and block Krylov solvers are separate concepts. A block matrix can be solved for one right-hand side. Support for block matrices in preconditioners and multigrid components depends on the chosen implementation.

## 21. Operators and Belos

### Tpetra::Operator

An `Operator` is an interface for a distributed linear transformation. It specifies domain/range maps and an `apply` operation. It does not prescribe storage: a sparse matrix and a matrix-free implementation can both implement this interface.

```cpp
Teuchos::RCP<const Operator> op = matrix;
op->apply(*x, *y, Teuchos::NO_TRANS, 2.0, 0.5);
// y = 2*A*x + 0.5*y.
```

The full operation is `Y = alpha * op(A) * X + beta * Y`. The transpose argument selects the original, transposed, or conjugate-transposed operator when supported. `getDomainMap()`, `getRangeMap()`, and `hasTransposeApply()` describe its interface. The default `apply(x, y)` computes `y = A*x`.

### Belos::LinearProblem

**Header:** `<BelosLinearProblem.hpp>`.

A `LinearProblem` connects the operator, solution, right-hand side, and optional preconditioners. It holds references through RCPs; it does not duplicate the matrix and vectors.

```cpp
using Problem = Belos::LinearProblem<Scalar, MultiVector, Operator>;
auto problem = Teuchos::rcp(new Problem(matrix, x, b));
const bool ready = problem->setProblem();
```

The existing values in `x` are the initial guess. The solver later overwrites `x` with its computed solution. `setProblem()` prepares and checks the problem setup and returns whether it succeeded; it does not solve the system.

| Method | Action |
|---|---|
| `setOperator(A)` | Set the operator |
| `setLHS(x)`, `setRHS(b)` | Set solution/initial guess and right-hand side |
| `setLeftPrec(P)`, `setRightPrec(P)` | Attach a preconditioner |
| `setProblem()` | Prepare the current system before solving |

A preconditioner applies an approximate inverse action, `P*r ≈ A⁻¹*r`. Right preconditioning corresponds to solving `A*P*z = b` and recovering `x = P*z`. Left preconditioning corresponds to `P*A*x = P*b`. Neither requires storing a dense inverse matrix.

### SolverFactory and SolverManager

**Header for Tpetra integration:** `<BelosSolverFactory_Tpetra.hpp>`. It includes the Tpetra traits and factory registration needed by these types.

A factory creates a solver manager. The manager stores algorithm settings and allocates the work vectors needed for the solve.

```cpp
Belos::SolverFactory<Scalar, MultiVector, Operator> factory;
auto options = Teuchos::rcp(new Teuchos::ParameterList);
options->set("Maximum Iterations", 300);
options->set("Convergence Tolerance", 1.0e-8);
options->set("Num Blocks", 40);
auto solver = factory.create("GMRES", options);
solver->setProblem(problem);
const auto status = solver->solve();
const int iterations = solver->getNumIters();
```

Check `status == Belos::Converged`; other return values report failure to satisfy convergence or another termination condition. `factory.isSupported(name)` tests a solver name; `factory.supportedSolverNames()` lists registered choices. `solver->setParameters(options)` changes settings.

| Method family | Intended systems |
|---|---|
| GMRES | General nonsymmetric systems |
| BiCGStab | General nonsymmetric systems, with short recurrences |
| CG | Symmetric/Hermitian positive-definite systems with compatible preconditioning |
| Flexible GMRES | Systems using a preconditioner that can vary between iterations |

For single-right-hand-side restarted GMRES with block size 1, `Num Blocks` controls the Krylov basis length before restart. Larger values require more vector storage. Parameters are solver-specific; pass a GMRES restart parameter only to a method that supports it.

After changing the matrix, initial guess, right-hand side, or preconditioner, prepare the system again with `problem->setProblem()` before solving. Refresh the preconditioner when its matrix data changes.

Belos convergence tests can use scaled or preconditioned residuals. To check the original equation directly, compute `r = b - A*x` and a global norm; the complete example below does this.

## 22. Ifpack2 and MueLu

### Ifpack2

**Header:** `<Ifpack2_Factory.hpp>`.

Ifpack2 constructs preconditioners from an assembled matrix. Common choices include diagonal scaling, Jacobi/Gauss–Seidel relaxation, incomplete LU, and additive Schwarz.

The factory returns an RCP to an `Ifpack2::Preconditioner` implementation. It retains the matrix and stores the data needed to apply its preconditioning action.

```cpp
Teuchos::RCP<const Matrix> input = matrix;  // Matrix must be fill-complete.
auto prec = Ifpack2::Factory::create<Matrix>("RELAXATION", input);
Teuchos::ParameterList options;
options.set("relaxation: type", std::string("Jacobi"));
options.set("relaxation: sweeps", 1);
options.set("relaxation: damping factor", 1.0);
prec->setParameters(options);
prec->initialize();
prec->compute();
problem->setRightPrec(prec);
problem->setProblem();
```

| Method | Action |
|---|---|
| `setParameters(options)` | Configure the implementation |
| `initialize()` | Prepare data depending on the sparse structure |
| `compute()` | Build numerical preconditioner data from coefficients |
| `apply(r, z)` | Compute the preconditioned vector `z` |
| `isInitialized()`, `isComputed()` | Query setup state |

Use `initialize()` followed by `compute()` for initial setup. If coefficients change with the same structure, call `compute()` again. Structural changes require structural setup again, with matrix replacement or reconstruction as required by the implementation. Updating the matrix does not automatically update cached factors or diagonals.

| Factory name | Preconditioner |
|---|---|
| `"DIAGONAL"` | Inverse-diagonal scaling |
| `"RELAXATION"` | Jacobi, Gauss–Seidel, or related relaxation |
| `"RILUK"` | Incomplete LU with a prescribed fill level |
| `"ILUT"` | Incomplete LU with thresholding |
| `"SCHWARZ"` | Additive Schwarz with a configured subdomain solver |

Local incomplete factorizations operate on process-local subproblems. Overlap and interprocess coupling depend on the surrounding domain-decomposition configuration.

### MueLu

**Header:** `<MueLu_CreateTpetraPreconditioner.hpp>`.

MueLu builds an algebraic multigrid hierarchy. It stores coarse operators, transfer operators, smoothers, and a coarse solver. A multigrid cycle smooths the fine-level error, transfers the residual to coarser levels, computes a correction, and transfers that correction back.

```cpp
Teuchos::RCP<Operator> input = matrix;
Teuchos::ParameterList options;
options.set("max levels", 10);
options.set("coarse: max size", 100);
auto prec = MueLu::CreateTpetraPreconditioner<Scalar, LO, GO, Node>(input, options);
problem->setRightPrec(prec);
problem->setProblem();
```

`CreateTpetraPreconditioner` builds the hierarchy and returns an RCP to `MueLu::TpetraOperator`. Its `apply(r, z)` applies the configured multigrid action, usable as a Belos preconditioner. It is not generally an exact solve of the original system.

| Interface | Action |
|---|---|
| `CreateTpetraPreconditioner(A, options)` | Build an initialized hierarchy |
| `prec->apply(r, z)` | Apply multigrid to a residual |
| `ReuseTpetraPreconditioner(A, *prec)` | Update an existing hierarchy using its configured reuse policy |

For the reuse call, retain the concrete RCP returned by the creation function and use matching template parameters. Reuse requires compatible structure and configuration; it does not mean keeping stale numerical data after changing matrix coefficients. Rebuilding is the simplest update path when those requirements are not established.

For systems of PDEs, correct block information and near-nullspace vectors can be necessary for effective coarsening. Supply coordinates and near-nullspace vectors through the `"Coordinates"` and `"Nullspace"` entries in the `"user data"` sublist when needed.

## 23. Complete example: assemble and solve

This program builds a distributed tridiagonal matrix with diagonal 2 and adjacent entries −1. It constructs `b = A*1`, solves with GMRES and Jacobi preconditioning, and reports `||b - A*x||₂ / ||b||₂`. The exact solution is the vector of ones.

Build with C++20 and a Trilinos installation providing Tpetra, Belos, and Ifpack2. Use the compiler and backend configuration required by that installation. In an MPI build, the same program runs with one or several processes.

```cpp
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosSolverFactory_Tpetra.hpp>
#include <Ifpack2_Factory.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    Tpetra::ScopeGuard runtime(&argc, &argv);
    using Scalar = double;
    using Map = Tpetra::Map<>;
    using LO = Map::local_ordinal_type;
    using GO = Map::global_ordinal_type;
    using Node = Map::node_type;
    using Matrix = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
    using Vector = Tpetra::Vector<Scalar, LO, GO, Node>;
    using MV = Tpetra::MultiVector<Scalar, LO, GO, Node>;
    using Op = Tpetra::Operator<Scalar, LO, GO, Node>;
    using Problem = Belos::LinearProblem<Scalar, MV, Op>;

    auto comm = Tpetra::getDefaultComm();
    const GO n = 64;
    auto map = Teuchos::rcp(new Map(static_cast<Tpetra::global_size_t>(n), GO{0}, comm));
    auto A = Teuchos::rcp(new Matrix(map, 3));
    for (std::size_t i = 0; i < map->getLocalNumElements(); ++i) {
        const GO row = map->getGlobalElement(static_cast<LO>(i));
        Teuchos::Array<GO> columns;
        Teuchos::Array<Scalar> values;
        if (row > 0) { columns.push_back(row - 1); values.push_back(-1.0); }
        columns.push_back(row); values.push_back(2.0);
        if (row + 1 < n) { columns.push_back(row + 1); values.push_back(-1.0); }
        A->insertGlobalValues(row, columns(), values());
    }
    A->fillComplete(map, map);

    auto x = Teuchos::rcp(new Vector(A->getDomainMap()));
    auto b = Teuchos::rcp(new Vector(A->getRangeMap()));
    Vector exact(A->getDomainMap());
    exact.putScalar(1.0);
    A->apply(exact, *b);
    x->putScalar(0.0);

    Teuchos::RCP<const Matrix> const_A = A;
    auto prec = Ifpack2::Factory::create<Matrix>("RELAXATION", const_A);
    Teuchos::ParameterList prec_options;
    prec_options.set("relaxation: type", std::string("Jacobi"));
    prec_options.set("relaxation: sweeps", 1);
    prec_options.set("relaxation: damping factor", 1.0);
    prec->setParameters(prec_options);
    prec->initialize();
    prec->compute();

    auto problem = Teuchos::rcp(new Problem(A, x, b));
    problem->setRightPrec(prec);
    if (!problem->setProblem()) return 1;

    auto options = Teuchos::rcp(new Teuchos::ParameterList);
    options->set("Maximum Iterations", 200);
    options->set("Convergence Tolerance", 1.0e-10);
    options->set("Num Blocks", 40);
    Belos::SolverFactory<Scalar, MV, Op> factory;
    auto solver = factory.create("GMRES", options);
    solver->setProblem(problem);
    const auto status = solver->solve();

    Vector residual(A->getRangeMap());
    A->apply(*x, residual);
    residual.update(1.0, *b, -1.0);  // residual = b - A*x.
    const double relative_residual = residual.norm2() / b->norm2();
    if (comm->getRank() == 0) {
        std::cout << "Converged: " << (status == Belos::Converged)
                  << ", iterations: " << solver->getNumIters()
                  << ", relative residual: " << relative_residual << '\n';
    }
    return status == Belos::Converged && relative_residual < 1.0e-8 ? 0 : 1;
}
```

All processes construct maps, finalize the matrix, solve, and compute global norms. Only printing is restricted to rank 0. Declaring the runtime guard first ensures that all numerical objects are destroyed before MPI/Kokkos finalization.

## 24. Official references

Version-specific API details in this guide follow the release sources. Online generated API pages can track a newer version.

- [Kokkos 5.2.1 source](https://github.com/kokkos/kokkos/tree/5.2.1) and [Trilinos 17.2.1 source](https://github.com/trilinos/Trilinos/tree/trilinos-release-17-2-1).
- [Kokkos View](https://kokkos.org/kokkos-core-wiki/API/core/view/view.html), [layouts and subviews](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Subviews.html), [mirrors](https://kokkos.org/kokkos-core-wiki/API/core/view/create_mirror.html), and [deep_copy](https://kokkos.org/kokkos-core-wiki/API/core/view/deep_copy.html).
- [Kokkos parallel dispatch](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html), [hierarchical parallelism](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/HierarchicalParallelism.html), and [DualView](https://kokkos.org/kokkos-core-wiki/API/containers/DualView.html).
- [Tpetra tutorial](https://trilinos.github.io/docs/tpetra/index.html), [Map](https://trilinos.github.io/docs/tpetra/class_tpetra_1_1_map.html), [MultiVector](https://trilinos.github.io/docs/tpetra/class_tpetra_1_1_multi_vector.html), and [CrsMatrix](https://trilinos.github.io/docs/tpetra/class_tpetra_1_1_crs_matrix.html).
- [LocalMap implementation in Trilinos 17.2.1](https://github.com/trilinos/Trilinos/blob/trilinos-release-17-2-1/packages/tpetra/core/src/Tpetra_Details_LocalMap.hpp) and [BlockCrsMatrix declarations](https://github.com/trilinos/Trilinos/blob/trilinos-release-17-2-1/packages/tpetra/core/src/Tpetra_BlockCrsMatrix_decl.hpp).
- [Belos](https://trilinos.github.io/docs/belos/index.html), [Ifpack2](https://trilinos.github.io/docs/ifpack2/index.html), and [MueLu](https://trilinos.github.io/docs/muelu/index.html).
