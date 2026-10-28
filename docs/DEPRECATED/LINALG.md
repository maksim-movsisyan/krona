# KRONA linear algebra

This document describes the current `krona::linalg` implementation, including scalar CSR and fixed-block BSR matrices. It covers the interfaces in `include/krona/linalg/` and the mesh adapter in `src/linalg/mesh_adapter.cpp`. Native behavior and restrictions discussed here refer to Trilinos 17.2.1.

The module separates distributed topology, coefficient storage, application fields, and solution algorithms. Topology is prepared once on the host. Coefficients and vectors can then be assembled directly in their native Kokkos-accessible memory. Belos and preconditioner operations form blocking boundaries around this asynchronous application work.

## Contents

1. [Objects and ownership](#1-objects-and-ownership)
2. [Types, execution spaces, and initialization](#2-types-execution-spaces-and-initialization)
3. [Distributions and sparsity](#3-distributions-and-sparsity)
4. [Complete CSR example](#4-complete-csr-example)
5. [Complete BSR example](#5-complete-bsr-example)
6. [Assembly and coefficient access](#6-assembly-and-coefficient-access)
7. [Asynchronous access and commit](#7-asynchronous-access-and-commit)
8. [Right-hand sides and application fields](#8-right-hand-sides-and-application-fields)
9. [Solver methods and options](#9-solver-methods-and-options)
10. [Preconditioners](#10-preconditioners)
11. [Mixed precision](#11-mixed-precision)
12. [Integration into a mesh-based solver](#12-integration-into-a-mesh-based-solver)
13. [Native behavior and hidden costs](#13-native-behavior-and-hidden-costs)
14. [Contracts, diagnostics, and extension points](#14-contracts-diagnostics-and-extension-points)

## 1. Objects and ownership

| KRONA object | Responsibility | Native representation |
| --- | --- | --- |
| `Distribution<Exec>` | Owned global IDs and their local order | `Tpetra::Map` |
| `CsrPattern<Exec>` | Immutable graph and optional input-to-native permutation | `Tpetra::CrsGraph` |
| `BsrPattern<Exec>` | Block graph, uniform block size, and expanded scalar distribution | Shared `CrsGraph` and point `Map` |
| `CsrMatrix<Scalar, Exec>` | Scalar coefficients on a fixed graph | `Tpetra::CrsMatrix` |
| `BsrMatrix<Scalar, Exec>` | Dense square blocks on a fixed graph | `Tpetra::BlockCrsMatrix` |
| `Vector<Scalar, Exec>` | Owned scalar unknowns | `Tpetra::Vector` |
| `Preconditioner<Matrix, PrecScalar>` | Revision-aware construction and numerical updates | Ifpack2, MueLu, or a supplied operator |
| `Solver<Matrix, PrecScalar>` | Belos manager, stopping policy, and true-residual verification | `Belos::SolverManager` |
| `FieldPack<...>` | Lightweight description of fields to pack or unpack | Device-copyable descriptors |
| `CellNumbering` | Host cell-to-global numbering for the mesh adapter | Host vectors and counts |

`Distribution`, `CsrPattern`, and `BsrPattern` are inexpensive to copy: copies share native maps, graph storage, and retained permutations. Constructing another matrix from a pattern allocates another coefficient array, while sharing its graph. BSR has an additional native host-index storage caveat described in [section 13](#13-native-behavior-and-hidden-costs).

Matrices, vectors, solvers, and preconditioners are noncopyable. A matrix or vector stores its distribution/pattern descriptor by value. The original descriptor variable can therefore go out of scope after construction. A solver and a preconditioner instead retain a **reference to the matrix**: that matrix must outlive them.

Native objects use `Teuchos::RCP`, a reference-counted owning handle. Copying an `RCP` shares the same object; it does not duplicate its coefficients. Taking `getRawPtr()` borrows a pointer without extending its lifetime. Several preconditioner handles can refer to different interfaces of the same object.

Application fields remain separate from native vectors. Packing and unpacking copy values in device-accessible memory; they do not make the vector an alias of the fields.

## 2. Types, execution spaces, and initialization

### 2.1 Native type policy

The module fixes native ordinals independently of mesh index types:

```cpp
using LocalOrdinal = int;
using GlobalOrdinal = long long;
```

Local row/column counts must fit the native local ordinal. Global IDs are nonnegative and must be smaller than the reserved maximum `GlobalOrdinal` value. Block expansion performs additional overflow checks.

`IndexTypes<Exec>` supplies the native node, memory space, map, and graph types. `Types<Scalar, Exec>` additionally supplies scalar and block matrices, vectors, multivectors, and the common operator interface.

| Choice | Current support |
| --- | --- |
| Matrix/vector scalar | `float` or `double` |
| Default matrix/vector scalar | `SolverReal` |
| Default preconditioner scalar | `PreconditionerReal` |
| Execution space | Configured KRONA Serial, OpenMP, CUDA, or HIP backend, with matching Trilinos instantiations |
| Native node | Backend-specific Tpetra wrapper node |
| Matrix/vector memory space | `Types<Scalar, Exec>::memory_space` |

Matching Tpetra explicit instantiations are required for the scalar, `int/long long` ordinals, and backend. Selecting `float` for the preconditioner in a `double` solve requires **both** scalar instantiations, as well as support in the chosen native preconditioner.

Use the wrapper's `memory_space` for application buffers intended to match native coefficient/vector storage. The native node memory space can differ from `Exec::memory_space`, for example with particular CUDA/HIP shared-memory allocation configurations. The graph permutation itself is allocated in `Exec::memory_space`. All custom kernels still require execution-space accessibility to every captured view.

### 2.2 Runtime and build integration

Tpetra must be initialized before constructing distributions. MPI and Kokkos must remain active until all native objects and wrappers are destroyed. The communicator passed to `Distribution` is borrowed: keep the underlying MPI communicator alive for the whole lifetime of the objects using it.

For an independent executable, `Tpetra::ScopeGuard` can initialize and finalize the required runtime. The complete examples below use this arrangement. Inside KRONA, retain the existing application runtime initialization and ensure Tpetra is initialized there; do not introduce a second independent owner of MPI/Kokkos finalization.

Link an application target to the module target:

```cmake
target_link_libraries(my_solver PRIVATE krona::linalg)
```

The KRONA Trilinos configuration must provide Tpetra, Belos, Ifpack2, and MueLu. There is no wrapper-side runtime solver/plugin discovery or configuration-file parser: options are C++ structures with optional native `Teuchos::ParameterList` values.

### 2.3 Common example declarations

The C++ examples use these includes and aliases. `S` is explicitly `double` to make their numerical settings unambiguous; production code can use `krona::SolverReal`.

<!-- example: preamble -->
```cpp
#include <krona/linalg/solver.hpp>
#include <krona/linalg/field_transfer.hpp>
#include <krona/linalg/mesh_adapter.hpp>

#include <Tpetra_Core.hpp>
#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace la = krona::linalg;
using Exec = krona::execution::ExecutionSpace;
using S = double;
using LO = la::LocalOrdinal;
using GO = la::GlobalOrdinal;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
```

## 3. Distributions and sparsity

### 3.1 What a distribution means

`Distribution<Exec>(owned_gids, comm)` collectively constructs a one-to-one map. Each supplied ID has exactly one owner across the communicator. Its position in `owned_gids` determines its local row/vector index. IDs need not be contiguous, sorted, or grouped by rank.

`global_size()` is the total number of owned entries, **not** the largest GID plus one. `local_size()` counts only this rank's owned entries. `global_id(local)` returns the GID for a local row. The constructor checks local duplicates and global uniqueness.

A distribution does not enumerate all remote columns needed by a matrix. Those belong to the finalized graph's **column map**:

| Map | Meaning in this module |
| --- | --- |
| Row map | Ownership and local order of stored rows |
| Domain map | Ownership of the input vector to `A*x` |
| Range map | Ownership of the output vector |
| Column map | Local indexing of all columns referenced by this rank, including remote ones |

The wrappers currently implement square systems with the same owned distribution for rows, domain, and range. A column map may overlap across ranks. Tpetra derives imports from the domain map to the column map and communicates remote vector entries when applying the matrix. KRONA vectors do not contain application-managed ghost slots for this purpose.

**Reuse the same distribution descriptor.** Matrix application and `Solver::solve()` require the vectors and matrix to hold the exact same native map object. Independently constructing mathematically equivalent maps is insufficient for these wrapper checks. Copying an existing `Distribution` preserves the required identity.

### 3.2 CSR input and finalization

`CsrInput` contains host topology:

- `row_offsets`: `owned_rows + 1` offsets, beginning with zero;
- `column_gids`: global column IDs for each row;
- rows in distribution order; columns in any order within a row.

Offsets must be monotone, the final offset must equal the column count, and every column must have an owner in the domain distribution. Duplicate columns within a row are rejected rather than accumulated. An empty rank uses `row_offsets = {0}` and still participates in collective construction and solves.

`CsrPattern` validates this topology, inserts global indices, and calls native `fillComplete(domain, range)`. Finalization builds the column map and communication structure and sorts entries by **local column ordinal**. This order can differ from the supplied global-column order.

The optional retained permutation maps each original input slot to the finalized flat graph entry:

```text
input_to_entry(original_slot) = native_entry
```

The mapping is available on host and device. It is used by `set_values()` and `set_values_async()`. `local_nonzeros()` counts entries stored in this rank's owned rows, including entries with remote columns. It does not count the neighbors' owned rows.

If assembly exclusively traverses the finalized graph, omit the input permutation:

<!-- example: no_input_mapping -->
```cpp
la::CsrPattern<Exec> pattern(rows, input, exec, false);
```

Here `rows`, `input`, and `exec` are an existing distribution, `CsrInput`, and execution instance. With this option, requesting either input mapping or calling an input-order setter is invalid. Direct native-order assembly remains available.

### 3.3 Block and scalar distributions

BSR starts with a graph whose rows and columns represent cells or other block entities. For a uniform block size `B`, `BsrPattern` expands the owned distribution into scalar degrees of freedom:

```text
global_dof(cell_gid, component) = cell_gid * B + component
local_dof(local_cell, component) = local_cell * B + component
```

This is cell-major/AoS ordering, independently of how application fields are stored. Block size must be positive and identical on all ranks. Each block is dense, square, and `B x B`.

`block_distribution()` addresses cells, while `distribution()` addresses scalar unknowns. Likewise, `local_nonzeros()` counts blocks and `local_values()` counts scalar coefficients: `blocks * B * B`.

For `B > 1`, a new `Distribution::expanded(B)` call constructs a new native point map. Therefore, create one `BsrPattern` and reuse it for matrices/vectors that must interoperate. Construct vectors from `block_pattern.distribution()` or `A.distribution()`, not from an independent expansion of the cell distribution. For `B == 1`, expansion shares the original map.

## 4. Complete CSR example

This example builds a distributed one-dimensional graph and assembles a simple square system directly into native device-accessible storage. The topology is host data; no host coefficient array is created.

The diagonal is `4`, each existing neighbor coefficient is `-1`, and the RHS is the row sum. The exact solution is therefore the all-ones vector, including across MPI partition boundaries. The example is a linear-algebra demonstration, not a mesh discretization.

### 4.1 Shared topology helper

The helper is also used by the BSR example. Its input column order is deliberately not assumed to be native order.

<!-- example: chain_topology -->
```cpp
struct ChainTopology {
    std::vector<GO> owned_gids;
    la::CsrInput input;
};

ChainTopology chain_topology(GO first, std::size_t count, GO global_count)
{
    ChainTopology result;
    result.input.row_offsets.push_back(0);
    for (std::size_t row = 0; row < count; ++row) {
        const GO gid = first + static_cast<GO>(row);
        result.owned_gids.push_back(gid);
        result.input.column_gids.push_back(gid);          // diagonal
        if (gid + 1 < global_count)
            result.input.column_gids.push_back(gid + 1);
        if (gid > 0)
            result.input.column_gids.push_back(gid - 1);
        result.input.row_offsets.push_back(result.input.column_gids.size());
    }
    return result;
}
```

### 4.2 Device assembly, solve, and extraction

<!-- example: csr_chain -->
```cpp
la::SolveResult solve_csr_chain(krona::mpi::CommView comm,
                               GO first, std::size_t count, GO global_count)
{
    Exec exec;
    const auto topology = chain_topology(first, count, global_count);
    la::Distribution<Exec> rows(topology.owned_gids, comm);
    la::CsrPattern<Exec> pattern(rows, topology.input, exec);
    la::CsrMatrix<S, Exec> A(pattern);
    la::Vector<S, Exec> b(A.distribution()), x(A.distribution());

    const auto graph = pattern.native()->getLocalGraphDevice();
    const auto row_map = rows.native()->getLocalMap();
    const auto col_map = pattern.native()->getColMap()->getLocalMap();
    const auto values = A.overwrite_values(exec);
    const auto rhs = b.overwrite(exec);

    Kokkos::parallel_for("example/assemble_csr", Policy(exec, 0, count),
        KOKKOS_LAMBDA(std::size_t row) {
            const GO row_gid = row_map.getGlobalElement(static_cast<LO>(row));
            S row_sum = 0;
            for (auto entry = graph.row_map(row);
                 entry < graph.row_map(row + 1); ++entry) {
                const GO col_gid = col_map.getGlobalElement(graph.entries(entry));
                const S value = col_gid == row_gid ? S{4} : S{-1};
                values(entry) = value;
                row_sum += value;
            }
            rhs(row, 0) = row_sum;
        });
    A.commit();
    b.commit();

    la::SolverOptions options;
    options.method = "GMRES";
    options.relative_tolerance = 1e-10;
    options.maximum_iterations = 300;
    options.restart_length = 40;
    options.preconditioner.type = "SCHWARZ";
    options.preconditioner.parameters.set("schwarz: overlap level", 0);
    options.preconditioner.parameters.set("inner preconditioner name", "RILUK");
    options.preconditioner.parameters.sublist("inner preconditioner parameters")
        .set("fact: iluk level-of-fill", 0);

    // Both matrix and preconditioner use S in this example.
    la::Solver<decltype(A), S> solver(A, options);
    const auto result = solver.solve(b, x); // x initially contains zeros

    using Space = typename decltype(x)::memory_space;
    Kokkos::View<S*, Space> owned_solution("example/csr_solution", count);
    x.copy_to_component(owned_solution, exec);
    x.commit(); // completes the read of x and the write to owned_solution
    // Application kernels can now use owned_solution on exec.
    return result;
}
```

`graph.row_map(row)` and `graph.row_map(row + 1)` delimit the finalized row. `graph.entries(entry)` is a local **column-map** index; it is not necessarily a local owned-cell index. The local map descriptors above convert local indices to global IDs inside the kernel without building application-side index copies.

Native vector views have rank two. A single `Tpetra::Vector` uses the second index `0`: `rhs(local_dof, 0)`.

### 4.3 Standalone entry point

Combine the common declarations, topology helper, CSR function, and this entry point to obtain a complete program. Each rank owns four rows.

<!-- example: main -->
```cpp
int main(int argc, char** argv)
{
    Tpetra::ScopeGuard runtime(&argc, &argv);
    const auto comm = krona::mpi::CommView::world();
    constexpr std::size_t count = 4;
    const GO first = static_cast<GO>(comm.rank()) * static_cast<GO>(count);
    const GO global_count = static_cast<GO>(comm.size()) * static_cast<GO>(count);
    const auto result = solve_csr_chain(comm, first, count, global_count);
    return result.converged ? 0 : 1;
}
```

All objects created inside the example function are destroyed before the runtime guard. All ranks must call the example with a consistent global partition.

## 5. Complete BSR example

The same cell graph can hold a coupled system with five unknowns per cell. No scalar-expanded CSR graph is constructed by `BsrPattern` or `BsrMatrix`.

The kernel below assembles dense diagonal blocks and diagonal off-cell blocks through the logical block accessor. Every coefficient is written, including zeros. The RHS again equals the scalar row sum, so every scalar solution component is one.

<!-- example: bsr_chain -->
```cpp
la::SolveResult solve_bsr_chain(krona::mpi::CommView comm,
                               GO first, std::size_t count, GO global_count)
{
    Exec exec;
    constexpr LO B = 5;
    const auto topology = chain_topology(first, count, global_count);
    la::Distribution<Exec> cells(topology.owned_gids, comm);
    la::CsrPattern<Exec> cell_graph(cells, topology.input, exec);
    la::BsrPattern<Exec> pattern(cell_graph, B);
    la::BsrMatrix<S, Exec> A(pattern);
    la::Vector<S, Exec> b(A.distribution()), x(A.distribution());

    const auto graph = pattern.native()->getLocalGraphDevice();
    const auto row_map = cells.native()->getLocalMap();
    const auto col_map = pattern.native()->getColMap()->getLocalMap();
    const auto blocks = A.overwrite_blocks(exec);
    const auto rhs = b.overwrite(exec);

    Kokkos::parallel_for("example/assemble_bsr", Policy(exec, 0, count),
        KOKKOS_LAMBDA(std::size_t cell) {
            const GO row_gid = row_map.getGlobalElement(static_cast<LO>(cell));
            for (LO equation = 0; equation < B; ++equation) {
                S row_sum = 0;
                for (auto entry = graph.row_map(cell);
                     entry < graph.row_map(cell + 1); ++entry) {
                    const GO col_gid = col_map.getGlobalElement(graph.entries(entry));
                    const bool diagonal_block = col_gid == row_gid;
                    for (LO variable = 0; variable < B; ++variable) {
                        const S value = diagonal_block
                            ? (equation == variable ? S{4} : S{0.05})
                            : (equation == variable ? S{-1} : S{0});
                        blocks(entry, equation, variable) = value;
                        row_sum += value;
                    }
                }
                rhs(cell * static_cast<std::size_t>(B)
                    + static_cast<std::size_t>(equation), 0) = row_sum;
            }
        });
    A.commit();
    b.commit();

    la::SolverOptions options;
    options.method = "FGMRES";
    options.relative_tolerance = 1e-10;
    options.preconditioner.type = "SCHWARZ";
    options.preconditioner.parameters.set("schwarz: overlap level", 0);
    options.preconditioner.parameters.set("inner preconditioner name", "RBILUK");
    options.preconditioner.parameters.sublist("inner preconditioner parameters")
        .set("fact: iluk level-of-fill", 0);

    la::Solver<decltype(A), S> solver(A, options);
    const auto result = solver.solve(b, x);

    using Space = typename decltype(x)::memory_space;
    Kokkos::View<S*, Space> owned_solution("example/bsr_solution", count * B);
    x.copy_to_component(owned_solution, exec);
    x.commit();
    // owned_solution(cell * B + component) is the extracted scalar DOF.
    return result;
}
```

To run this example, use the entry point above with `solve_bsr_chain` in place of `solve_csr_chain`.

`blocks(entry, equation, variable)` is an unmanaged rank-three `LayoutStride` view of the actual native coefficient buffer. It neither allocates another block array nor copies the coefficients. The accessor accounts for the native small-block layout. Prefer it over hard-coded flat block offsets.

The native graph still contains **block** entries. Its column map addresses block entities, while `A.distribution()` and the vectors address scalar DOFs.

## 6. Assembly and coefficient access

There are three assembly routes. They differ in ordering and data movement:

| Route | Input order | Additional coefficient copy |
| --- | --- | --- |
| Direct `overwrite_*()` / `edit_*()` and custom kernel | Finalized native graph order | None introduced by the wrapper |
| `set_values_async(input, exec)` | Original graph input order | One device-accessible permutation/cast kernel |
| `set_values(span)` | Original graph input order | Host assembly into native storage; native host/device synchronization may follow |

### 6.1 Direct access

| Operation | CSR | BSR |
| --- | --- | --- |
| Mutable flat coefficients | `edit_values(exec)` | `edit_values(exec)` |
| Full flat overwrite | `overwrite_values(exec)` | `overwrite_values(exec)` |
| Read-only flat coefficients | `read_values(exec)` | `read_values(exec)` |
| Logical dense blocks | — | `edit_blocks(exec)`, `overwrite_blocks(exec)`, `read_blocks(exec)` |

`edit_*()` preserves current values. `overwrite_*()` declares that the caller will replace every coefficient. Initialize every coefficient before using the matrix, including entries that should be zero. The wrapper does not track which entries a kernel has written.

Flat CSR values have one element per native graph entry. Flat BSR values have `B * B` elements per graph entry in the **native** small-block layout. Do not assume that flat BSR offsets match the row-major layout accepted by `set_values()`.

Obtaining a view introduces no wrapper-owned coefficient copy. Native Tpetra access can still allocate or synchronize a mirror if necessary; `overwrite_blocks()` also has a native `ReadWrite` restriction. See [section 13](#13-native-behavior-and-hidden-costs).

### 6.2 Original input order

For CSR, host `set_values()` accepts exactly one `Scalar` value per original input column slot. BSR accepts flat, row-major logical blocks:

```text
input[(original_entry * B + equation) * B + variable]
```

The setters apply the retained permutation internally. Do not pre-permute their input.

`set_values_async()` accepts an execution-accessible Kokkos view and performs conversion with `static_cast` while copying:

- CSR: rank-one view of length `pattern.local_nonzeros()`;
- BSR: rank-one view of length `pattern.local_values()`, in the row-major order above;
- BSR: rank-three view `(original_entry, equation, variable)`, with any accessible layout.

The input must remain valid until the submitted work completes. `set_values_async()` joins the matrix's tracked batch; call `A.commit()` before native operations. Host `set_values()` instead requires the previous batch to have been committed and returns after its host writes.

For example, with an existing CSR `pattern`, a host vector in original input order can be installed as follows:

<!-- example: host_values -->
```cpp
std::vector<S> input_values(pattern.local_nonzeros(), S{0});
// Fill every input_values[original_entry] with the intended coefficient.
A.set_values(std::span<const S>(input_values));
```

This fragment deliberately initializes zeros; a usable physical operator must be assembled by the application.

### 6.3 Mapping precomputed assembly slots

Finite-volume face kernels often know an original input slot for each contribution. Translate it once using `pattern.input_to_entry()` and retain a device slot table, or capture the permutation and use `permutation(original_slot)` in the kernel.

The module does not construct a face-to-matrix-slot lookup. Build such a lookup as part of discretization setup and rebuild it when the graph changes. The permutation remains valid for the lifetime of the immutable pattern.

Parallel faces can contribute to the same diagonal or off-diagonal entry. In that case, use atomics, coloring, or an equivalent application assembly strategy. An editable view does not make concurrent additions safe. This BSR helper demonstrates additive assembly with precomputed native slots:

<!-- example: add_block_entries -->
```cpp
template<class SlotView, class IncrementView>
void add_block_entries(la::BsrMatrix<S, Exec>& A,
                       SlotView native_slots, IncrementView increments,
                       const Exec& exec)
{
    // increments(contribution, equation, variable) is application-owned data.
    const auto blocks = A.edit_blocks(exec);
    const LO B = A.block_size();
    Kokkos::parallel_for("example/add_block_entries",
        Policy(exec, 0, native_slots.extent(0)),
        KOKKOS_LAMBDA(std::size_t contribution) {
            const auto entry = native_slots(contribution);
            for (LO a = 0; a < B; ++a)
                for (LO b = 0; b < B; ++b)
                    Kokkos::atomic_add(&blocks(entry, a, b),
                                      static_cast<S>(increments(contribution, a, b)));
        });
}
```

Zero the matrix first if these are the first contributions. This helper leaves the matrix pending; a later `A.commit()` ends the batch. CSR uses the same approach with `&values(native_entry)`.

### 6.4 Native interoperability

`A.native()` exposes an `RCP` to the mutable native matrix, even when accessed through a const wrapper. This is an expert escape hatch: native writes bypass pending-access tracking and the coefficient revision counter.

Before using native access, commit wrapper work. After native writes, finish the required execution-space synchronization and call `A.mark_modified()` so preconditioners observe the change. `mark_modified()` increments the revision; it does **not** wait for external kernels.

`create_native_structure<OtherScalar>()` creates independent native coefficient storage on the same graph and point maps. It is used internally for mixed-precision preconditioners and is available for advanced interoperability. It is not a coefficient conversion or a filled copy of `A`.

## 7. Asynchronous access and commit

### 7.1 Access batches

Every wrapper device read or write opens, or joins, an access batch on a particular `Exec` instance:

1. Acquire views using that instance.
2. Submit all consumers/producers of those views to the **same instance**.
3. Call `commit()` before a native matrix operation, preconditioner preparation, solve, host-native access, or a switch to another instance.

Multiple accesses on the same instance can share one batch. There is no need to commit after each kernel. The wrapper rejects a different execution instance while a batch is pending.

`commit()` fences the recorded instance and clears the pending flag. It is not a communicator barrier or a global Kokkos fence. It does nothing when no access is pending. An external `exec.fence()` alone does not clear the wrapper flag: still call `commit()`.

**Reads are tracked too.** A solver must not overwrite a vector while a custom kernel is still reading it. Consequently, `read()`, `read_values()`, `read_blocks()`, and solution-extraction helpers all require a subsequent commit before that wrapper participates in native work.

### 7.2 Operation boundaries

| Operation | Completion contract |
| --- | --- |
| Custom kernels through acquired views | Submitted on the caller's instance; caller commits the wrapper |
| `set_values_async()`, field packing/unpacking, component copies | No wrapper completion fence; join the vector/matrix batch |
| `Vector::fill_async(value, exec)` | Submitted deep copy; leaves a pending batch |
| `Vector::fill(value, exec)` | Fill followed by instance commit |
| `commit()` | Completes tracked work on one execution instance |
| Distribution/pattern construction | Collective setup; pattern construction includes native work and a global Kokkos fence |
| Matrix construction | Allocation/native initialization with a global Kokkos fence |
| Host `set_values()` | Blocking host writes; no pending wrapper batch |
| `Matrix::apply()` | Collective, blocking native operation, followed by a global Kokkos fence |
| Actual preconditioner setup/update | Collective and blocking, ending in a global Kokkos fence |
| `Solver::solve()` | Collective, blocking; computes global residual norms; the iteration path ends in a global Kokkos fence |

“Async” here describes wrapper submission and completion, not a guarantee that obtaining a native device view is always nonblocking. Tpetra may first synchronize its current host data.

For example, packing on one stream and passing a different `Exec` to a kernel that consumes the packed vector is invalid until a proper dependency and wrapper commit have been established. Merely passing an execution instance to a view getter does not bind the view to that stream; the kernel launch must use it too.

Matrix/vector destructors commit pending work as a final lifetime measure. Explicit commits at application/native boundaries make the synchronization and object lifetime clear. Do not retain an old editable view and write through it after a commit and solve: reacquire an editable view so tracking and matrix revisions remain correct.

## 8. Right-hand sides and application fields

### 8.1 Native vector access

`Vector` stores owned scalar DOFs only. Construction requests native zero initialization, making a new solution vector a zero initial guess. A later solve retains the previous solution unless the application resets it.

| Method | Access intent |
| --- | --- |
| `edit(exec)` | Read/write current values |
| `overwrite(exec)` | Replace all local values, avoiding preservation of old host values |
| `read(exec)` | Read-only device view |
| `fill_async(value, exec)` | Asynchronous full fill |
| `fill(value, exec)` | Full fill and commit |
| `copy_from_component(view, exec)` | Copy a rank-one view's owned prefix into the vector |
| `copy_to_component(view, exec)` | Copy the vector into a rank-one view's owned prefix |

The direct accessors return native rank-two views `(local_dof, 0)`. Component copies respect the source/destination view's stride and convert element types explicitly. They do not require contiguous components.

For a coupled vector, `copy_to_component()` copies the **entire flattened DOF sequence**. It does not select one physical component per cell. To extract only component `c`, use a custom kernel reading `x_values(cell * B + c, 0)`, or unpack the whole vector into the intended fields.

### 8.2 One multicomponent field

`copy_from_field()` and `copy_to_field()` accept `fields::FieldView<T, N, Space, Layout>`. They pack/unpack `N` components per owned cell in cell-major order. A scalar field uses `N == 1`.

Given compatible `b`, `x`, and existing five-component field descriptors `residual` and `increment`:

<!-- example: single_field -->
```cpp
la::copy_from_field(b, residual.as_const(), exec);
b.commit();

const auto result = solver.solve(b, x);
if (result.converged) {
    la::copy_to_field(x, increment, exec);
    x.commit();
}
```

The vector length must be divisible by `N`; the field must contain at least `vector.local_size() / N` cells. Fields can have extended ghost/boundary storage. Only the owned prefix is transferred, and the extended tail is untouched.

### 8.3 Several scalar fields, or a mixture of fields

`field_pack()` concatenates components **within each cell**, in argument order. It constructs descriptors without allocating value storage. Packing all descriptors launches one kernel, rather than one kernel per field.

For five scalar residuals `R1` through `R5`, the vector order is:

```text
R1(cell0), R2(cell0), R3(cell0), R4(cell0), R5(cell0),
R1(cell1), R2(cell1), R3(cell1), R4(cell1), R5(cell1), ...
```

With five scalar destination fields `dU1` through `dU5`:

<!-- example: five_fields -->
```cpp
const auto residuals = la::field_pack(R1, R2, R3, R4, R5);
static_assert(decltype(residuals)::components == 5);
la::copy_from_fields(b, residuals, exec);
b.commit();

const auto result = solver.solve(b, x);
if (result.converged) {
    const auto increments = la::field_pack(dU1, dU2, dU3, dU4, dU5);
    la::copy_to_fields(x, increments, exec);
    x.commit();
}
```

Accepted pack entries include:

- scalar and multicomponent `FieldView` descriptors;
- rank-one Kokkos views;
- strided rank-one component subviews;
- differing accessible layouts and scalar types;
- const descriptors for input; mutable descriptors are required for output.

For example, mass, three momentum components, and energy can be packed either as three fields or as selected components. Here `mass` and `energy` are scalar `FieldView`s and `momentum` has three components:

<!-- example: mixed_fields -->
```cpp
const auto conserved = la::field_pack(mass, momentum, energy);
static_assert(decltype(conserved)::components == 5);
la::copy_from_fields(b, conserved, exec);
b.commit();

// Equivalent component order, using the explicit-execution convenience form:
la::copy_from_fields(b, exec, mass.component(0),
                    momentum.component(0), momentum.component(1),
                    momentum.component(2), energy.component(0));
b.commit();
```

The supported call forms are:

```text
copy_from_fields(vector, field_pack(...), exec)
copy_to_fields(vector, field_pack(...), exec)
copy_from_fields(vector, exec, field1, field2, ...)
copy_to_fields(vector, exec, field1, field2, ...)
```

The pack form has a default execution instance. The variadic convenience form requires `exec` as the second argument.

### 8.4 Transfer contracts

Packing is a copy with a possible scalar conversion, not zero-copy aliasing. It does not change signs, apply scaling, or transform physical variables. If the nonlinear convention is `J * delta = -R`, supply `-R` or construct the RHS through a custom kernel; packing `R` alone does not negate it.

The application must ensure all of the following:

- The pack width equals the intended number of scalar unknowns per cell; for a BSR system this normally equals `A.block_size()`.
- The local cell order matches the block distribution. Divisibility checks cannot detect all semantic ordering or component-count mistakes.
- The field/component order matches the equation and variable order used in the matrix.
- Input storage outlives the submitted kernel. Destination fields do not overlap each other or the source vector, and inputs do not alias the destination vector.
- Field producers, packing, and field consumers have correct execution-instance dependencies. The transfer module does not synchronize an independent fields manager.

After packing, `b.commit()` completes reads of the input fields and writes to `b`. After unpacking, `x.commit()` completes reads of `x` and writes to the destination fields. Further kernels on the same instance can also follow an unpack before the eventual commit.

No transfer updates KRONA MPI halos or physical boundary slots. Run the application's halo exchange and boundary procedures when their updated values are needed.

## 9. Solver methods and options

### 9.1 Selecting a method

`Solver<Matrix, PrecScalar>` obtains its solver scalar and execution space from `Matrix`. The same interface accepts CSR and BSR matrices. The main exercised methods are:

| `options.method` | Intended problem | Preconditioning and storage considerations |
| --- | --- | --- |
| `"GMRES"` | General nonsymmetric square systems | Restarted Krylov basis; usually right-preconditioned by the automatic policy |
| `"FGMRES"` / `"FLEXIBLE GMRES"` | General systems with potentially variable or inexact preconditioning | Requires right preconditioning when enabled; stores preconditioned basis vectors too |
| `"BICGSTAB"` | General nonsymmetric systems when a short recurrence is useful | Requires right preconditioning when enabled; no GMRES restart length; breakdown remains possible |
| `"CG"` | Symmetric positive-definite systems | Requires mathematically compatible positive-definite, symmetry-preserving preconditioning |
| `"MINRES"` | Real symmetric systems, including indefinite systems | Requires compatible symmetric positive-definite preconditioning |

Method names are uppercased. `FGMRES` is explicitly translated to Belos' `FLEXIBLE GMRES` name. Other names go to the native Belos factory; availability depends on the build and manager. They are not a promise of compatibility with all wrapper controls or matrix formats. The wrapper requires a manager exposing `Convergence Tolerance` and `Maximum Iterations` and only provides a square, single-RHS solve interface.

Use the canonical names above for the documented behavior. Automatic side selection and symmetry metadata use name checks: a name ending in `CG`, or exactly `MINRES`, is treated as symmetric. Native aliases do not necessarily receive every canonical-name special case.

`problem->setHermitian(true)` sets Belos metadata; it does not verify or symmetrize the assembled matrix. Selecting CG or MINRES is the application's mathematical assertion. Arbitrary ILU, nonsymmetric relaxation, or AMG settings do not automatically become suitable for these methods.

### 9.2 Wrapper options

| `SolverOptions` member | Default | Meaning |
| --- | --- | --- |
| `method` | `"GMRES"` | Native solver name, with the FGMRES alias above |
| `relative_tolerance` | Unset | Chooses `1e-5` for actual matrix scalar `float`, `1e-10` for `double` |
| `absolute_tolerance` | `0` | Absolute true-residual target |
| `maximum_iterations` | `500` | Total iteration limit; zero requests residual evaluation without iterations |
| `restart_length` | `50` | Default native `Num Blocks` where supported |
| `side` | `Automatic` | Left for the symmetric name cases above; right otherwise |
| `solver_parameters` | Empty | Additional native Belos parameters |
| `preconditioner` | Default `PreconditionerOptions` | AMG is enabled by default |

Relative tolerance must be finite, in `[0, 1)`, and absolute tolerance finite and nonnegative. At least one must be positive. The iteration limit must be nonnegative. Restart length must be positive when the selected manager uses `Num Blocks` and that parameter was not explicitly supplied.

`Left` corresponds to a transformed problem involving `M^{-1} A`; `Right` involves `A M^{-1}` while recovering the physical solution. The final wrapper convergence decision always uses the original residual `b - A*x`.

For an SPD problem with simple Jacobi preconditioning:

<!-- example: cg_options -->
```cpp
la::SolverOptions options;
options.method = "CG";
options.relative_tolerance = 1e-10;
options.absolute_tolerance = 1e-14;
options.maximum_iterations = 1000;
options.preconditioner.type = "RELAXATION";
options.preconditioner.parameters.set("relaxation: type", "Jacobi");
options.preconditioner.parameters.set("relaxation: sweeps", 1);
```

Use a tolerance appropriate to the matrix scalar. The stopping tolerance passed to Belos must be positive and representable in that scalar; a tiny double-valued option can underflow when the solve itself uses `float`.

### 9.3 Native parameters and precedence

`solver_parameters` is copied into the native manager at construction. Parameter names, value types, and supported choices are native Belos contracts. `solver.native()->getValidParameters()` provides the selected manager's valid parameter list. Teuchos parameters are typed: an `int`, `double`, and `float` with the same numerical value are not interchangeable for every native setting.

The wrapper subsequently controls these settings on each iterating solve:

| Native setting | Source/precedence |
| --- | --- |
| `Convergence Tolerance` | Derived from wrapper tolerances and the current RHS/initial residual |
| `Maximum Iterations` | `options.maximum_iterations` |
| Supported implicit/explicit/legacy residual-scaling controls | Set to `"None"` by the wrapper |
| `Num Blocks` | Explicit `solver_parameters` value wins; otherwise `restart_length` |
| `Maximum Restarts` | Explicit native value wins; otherwise derived from the iteration limit and `Num Blocks`, if supported |
| Other native parameters | Passed through to the manager |

Set stopping policy through `SolverOptions`; putting a conflicting tolerance or residual-scaling choice into `solver_parameters` will not override it. For managers with an unscaled residual control, the wrapper passes the absolute residual threshold; otherwise it passes the corresponding initial-residual-relative tolerance. The independent final residual check remains authoritative.

For GMRES-specific restart and native logging controls:

<!-- example: belos_parameters -->
```cpp
la::SolverOptions options;
options.method = "GMRES";
options.solver_parameters.set("Num Blocks", 80);
options.solver_parameters.set("Maximum Restarts", 20);
options.solver_parameters.set("Verbosity",
    static_cast<int>(Belos::Errors | Belos::Warnings | Belos::StatusTestDetails));
options.solver_parameters.set("Output Frequency", 10);
```

Options are copied into the solver/preconditioner. Editing the original options variable afterward does not reconfigure an existing solver. Construct a new solver for a new wrapper configuration. The mutable native manager is an interoperability facility; some of its parameters are reapplied by the wrapper at the next solve as described above.

### 9.4 Initial guesses, stopping criteria, and results

`solve(b, x)` accepts the current contents of `x` as the initial guess and overwrites `x` with the resulting iterate. It never resets the guess automatically. Reusing a solution from a previous time step is therefore straightforward; use `x.fill(0, exec)` when a zero guess is intended.

For a nonempty system, the wrapper computes the true initial residual and defines:

```text
r0 = b - A*x_initial
reference = ||b||_2, if ||b||_2 > 0; otherwise ||r0||_2
threshold = max(absolute_tolerance, relative_tolerance * reference)
converged = finite(||b - A*x||_2) && ||b - A*x||_2 <= threshold
```

The zero-RHS rule permits meaningful relative convergence from a nonzero initial guess. If both reference and residual are zero, relative residual is reported as zero. Nonfinite RHS/initial norms and an overflowing threshold are rejected.

| `SolveResult` member | Meaning |
| --- | --- |
| `converged` | Authoritative independently verified true-residual result |
| `krylov_converged` | Native Belos solve returned `Converged` |
| `iterations_started` | Belos iterations were actually invoked |
| `iterations` | Iterations reported by the manager |
| `rhs_norm` | `||b||_2` |
| `initial_residual_norm` | `||b - A*x_initial||_2` |
| `residual_norm` | Final true residual norm |
| `relative_residual` | Final true norm divided by the reference above |
| `residual_threshold` | Absolute norm threshold used for verification |

If the initial guess already meets the target, or `maximum_iterations == 0`, the wrapper returns after the initial residual evaluation, without preparing a preconditioner or calling Belos. In those paths `iterations_started` and `krylov_converged` remain false. A globally empty system returns converged with zero/default result fields.

A native convergence flag and the verified result can differ. Inspect `result.converged`; the wrapper does not automatically retry or perform iterative refinement if the final true residual fails the target. Reported norms use `double` fields, but a float solve still uses the native float-scalar numerical operations; these fields do not imply double-precision norm accumulation.

### 9.5 The auxiliary residual vector

The solver lazily allocates one native vector in the solver scalar on the first nonempty solve. This is workspace for `b - A*x`, not another physical unknown or application field.

It provides the initial true residual to Belos and independently verifies the final result. An iterating solve performs one initial and one final wrapper residual SpMV, plus the corresponding global norms. The initial residual is reused by Belos instead of asking it to recompute the same initial SpMV. This check consumes an additional owned vector and numerical work, but it does not copy the full residual to a host field.

## 10. Preconditioners

### 10.1 Selection and supported routes

`PreconditionerOptions` contains `type` (default `"AMG"`) and a native parameter list. The selector is uppercased; native parameter names and values retain their native case and types.

| Selection | CSR | BSR | Notes |
| --- | --- | --- | --- |
| Empty string / `"NONE"` | Yes | Yes | No preconditioner operator; native Belos runs without one |
| `"RELAXATION"` | Yes | Yes | Defaults to Jacobi unless specified; BSR uses native block-aware handling |
| `"CHEBYSHEV"` | Exercised | Native/build dependent, not an exercised BSR combination | Polynomial preconditioning; spectral assumptions matter |
| `"SCHWARZ"` with `RILUK` | Yes | Use the block route instead | Distributed wrapper around a local scalar incomplete factorization |
| `"SCHWARZ"` with `RBILUK` | Not the documented scalar route | Yes, zero overlap | Block incomplete factorization on local subdomains |
| `"AMG"` / `"MUELU"` | Yes | Yes, with the BSR restrictions below | Algebraic multigrid |
| Custom factory | Yes | Yes | Any compatible native operator in the preconditioner scalar |

Except for `NONE`, AMG, and MueLu, names are delegated to `Ifpack2::Factory`. Other native choices can be available in a particular build; the module does not promise every native preconditioner works with every storage format, map layout, scalar, or backend.

Standalone local factors can expose subdomain maps rather than the matrix's distributed point maps. Such an operator is rejected by the wrapper's map checks. In particular, use `SCHWARZ` for the documented distributed ILU route and for the block `RBILUK` route, instead of attaching a bare local factor directly.

To disable the default AMG:

<!-- example: no_preconditioner -->
```cpp
la::SolverOptions options;
options.preconditioner.type = "NONE";
```

### 10.2 Relaxation and Schwarz

Jacobi applies inexpensive diagonal/block-diagonal corrections. Native relaxation settings select sweeps and other available algorithms. Changing relaxation to a nonsymmetric method can invalidate the mathematical assumptions of CG/MINRES.

The CSR and BSR examples above explicitly configure Schwarz and ILU level zero. For BSR, the wrapper supplies these defaults when omitted:

```text
inner preconditioner name = "RBILUK"
schwarz: overlap level = 0
```

Trilinos 17.2.1's block Schwarz overlap path is unsupported here. **Any nonzero BSR overlap is rejected**, including on a one-rank run. This restriction concerns the preconditioner subdomain overlap; it does not remove off-rank matrix columns or Tpetra's normal SpMV communication.

Increasing ILU fill can improve convergence while increasing setup time and factor memory. Native `fact: iluk level-of-fill` is an integer in the inner preconditioner parameter sublist. The best fill level, overlap for supported scalar paths, and algorithm depend on the assembled operator.

### 10.3 Algebraic multigrid

MueLu builds coarse levels from the operator, applies relaxation on fine/coarse levels, and transfers corrections between them. Matrix values influence quantities such as coupling strength, interpolation, and coarse operators; a fixed graph alone does not make an old hierarchy numerically current.

The wrapper supplies the following defaults only when the corresponding parameter is absent:

| Parameter | CSR default | BSR default |
| --- | --- | --- |
| `multigrid algorithm` | `"sa"` | `"unsmoothed"` |
| `cycle type` | `"V"` | `"V"` |
| `smoother: type` | `"CHEBYSHEV"` | `"RELAXATION"` |
| `number of equations` | Not set by wrapper | Block size `B` |
| `coarse: type` | Not set by wrapper | `"RELAXATION"` |
| `use kokkos refactor` | `true` | `false` |
| `coarse: max size` | `200` | `200` |
| `verbosity` | `"none"` | `"none"` |
| `reuse: type` | `"none"` | `"none"` |

For BSR relaxation defaults, `smoother: params` selects Jacobi. If the coarse type is relaxation, `coarse: params` selects Jacobi with eight sweeps. User-specified values take precedence. These coarse defaults avoid requiring a separate direct coarse solver for the default block route; they are not a universal choice for all coupled physics.

For BSR, `number of equations` must equal the block size and `use kokkos refactor` must remain false. These are checked restrictions of the current Trilinos 17.2.1 integration. An incompatible alternative smoother, coarse solver, or hierarchy configuration can still be rejected by native MueLu.

A concrete AMG configuration suitable for either matrix format's own defaults is:

<!-- example: amg_options -->
```cpp
la::SolverOptions options;
options.method = "GMRES";
options.preconditioner.type = "AMG";
auto& amg = options.preconditioner.parameters;
amg.set("coarse: max size", 100);
amg.set("verbosity", "none");
amg.set("reuse: type", "none");
```

Default BSR AMG aims to preserve block storage in the hierarchy. However, native coarse-operator construction can temporarily expand block operands into CSR; this significant setup cost is detailed in [section 13](#13-native-behavior-and-hidden-costs).

### 10.4 Coefficient revisions and reuse

Every mutable matrix-access acquisition increments `revision()`, even if the application eventually writes no coefficients. Host setters and `mark_modified()` also increment it. Reads do not.

`prepare()` compares the local revision to `prepared_revision_` and collectively reduces a dirty flag. If any rank changed, all ranks participate in the numerical update. If no rank changed and preparation remains valid, setup is skipped.

| Situation | Behavior |
| --- | --- |
| First enabled preparation | Construct the native preconditioner |
| Repeated prepare/solve with unchanged matrix | Reuse the prepared operator |
| Changed coefficients, Ifpack2 | Reuse initialized structure and call `compute()` again |
| Changed coefficients, AMG with default `reuse: type = "none"` | Release the old wrapper-owned hierarchy and create a fresh one |
| Changed coefficients, AMG with explicit native reuse policy | Call native `ReuseTpetraPreconditioner` |
| `invalidate()` | Force the next preparation path; retain the configured native reuse policy |
| Disabled preconditioner or globally empty system | Record preparation state without constructing an operator |

`"none"` means full AMG rebuilding **after a detected change**, not rebuilding on every unchanged solve. For explicit hierarchy reuse:

<!-- example: amg_reuse -->
```cpp
options.preconditioner.parameters.set("reuse: type", "full");
```

The exact components retained by a reuse mode follow native MueLu policy. Reusing a hierarchy is not equivalent to recomputing every hierarchy component from scratch, even though updated matrix data are supplied. Only `none` and `full` are the documented exercised reuse choices here. Other native modes are passed through and remain algorithm/build dependent. Preserve native lowercase `"none"`; selector uppercasing does not apply to this parameter value.

`invalidate()` does not override AMG's configured reuse policy or discard all cached symbolic structure. For a deliberate fresh configuration or topology, construct a new solver/preconditioner. `generation()` counts actual setup/update executions, not solve calls or matrix revisions that have not yet been prepared.

`Solver::prepare()` optionally performs explicit preconditioner preparation; an iterating `solve()` also does it automatically. Early successful guesses and zero-iteration solves do not prepare. A separate `prepare()` does not guarantee that all Belos/native workspace has already been allocated.

### 10.5 Standalone use and native handles

`Preconditioner<Matrix, PrecScalar>` can be used without the Belos wrapper. After preparing it, `native()` returns the solver-scalar operator and `native_inner()` returns the operator in `PrecScalar`.

The following function applies a prepared preconditioner to committed input/output vectors:

<!-- example: standalone_preconditioner -->
```cpp
template<class Matrix, class P>
void apply_preconditioner(la::Preconditioner<Matrix, P>& pc,
                         const la::Vector<typename Matrix::scalar_type,
                                          typename Matrix::execution_space>& input,
                         la::Vector<typename Matrix::scalar_type,
                                    typename Matrix::execution_space>& output)
{
    input.require_committed();
    output.require_committed();
    pc.prepare();
    const auto op = pc.native();
    if (!op.is_null()) {
        op->apply(*input.native(), *output.native());
    } else {
        Tpetra::deep_copy(*output.native(), *input.native()); // identity action
    }
    Kokkos::fence("application: standalone preconditioner complete");
}
```

Use distinct vectors with compatible point maps. These native calls are the application's blocking/ordering responsibility. A disabled or empty preconditioner may have a null native handle. `native()` requires preparation but does not automatically detect a subsequent matrix edit; call `prepare()` again before standalone use after changes.

### 10.6 Custom factories

A custom factory receives:

1. the native matrix in the requested preconditioner scalar and the **same storage format** as the wrapper matrix;
2. the previous inner operator, null on first construction.

It returns a nonnull `Tpetra::Operator<PrecScalar, ...>`. The wrapper checks that its domain and range maps match the system point map using `isSameAs`. Returning an operator does not require returning a matrix: a multilevel, composite, or application-defined native operator can satisfy this interface.

This example implements a reusable Ifpack2 Jacobi factory, illustrating construction versus coefficient refresh:

<!-- example: custom_factory -->
```cpp
template<class Matrix, class P>
typename la::Preconditioner<Matrix, P>::Factory jacobi_factory()
{
    using PC = la::Preconditioner<Matrix, P>;
    using E = typename Matrix::execution_space;
    using Node = typename la::Types<P, E>::node_type;
    using RowMatrix = Tpetra::RowMatrix<P, LO, GO, Node>;
    using NativePC = Ifpack2::Preconditioner<P, LO, GO, Node>;

    return [](const Teuchos::RCP<const typename PC::matrix_type>& matrix,
              const Teuchos::RCP<typename PC::inner_operator_type>& previous)
              -> Teuchos::RCP<typename PC::inner_operator_type> {
        Teuchos::RCP<NativePC> pc;
        if (previous.is_null()) {
            pc = Ifpack2::Factory::create<RowMatrix>("RELAXATION", matrix);
            Teuchos::ParameterList parameters;
            parameters.set("relaxation: type", "Jacobi");
            pc->setParameters(parameters);
            pc->initialize();
        } else {
            pc = Teuchos::rcp_dynamic_cast<NativePC>(previous, true);
        }
        pc->compute(); // current coefficients, including when reusing the object
        return pc;
    };
}
```

Given an assembled matrix `A`, install the callback through the solver constructor:

<!-- example: custom_factory_use -->
```cpp
using Matrix = la::CsrMatrix<S, Exec>;
la::SolverOptions options;
options.method = "GMRES";
la::Solver<Matrix, S> solver(A, options, jacobi_factory<Matrix, S>());
```

A supplied callback overrides type-based selection, even if `options.preconditioner.type` is `NONE`. The factory is called initially and after collectively detected changes. It must refresh any reused numerical data and obey the native collective requirements on every rank. Returning the previous object unchanged is only correct if that object already handles the updated coefficients appropriately.

## 11. Mixed precision

### 11.1 Choosing the two scalars

Matrix and vector storage use the matrix's `Scalar`, normally `SolverReal`. The second solver template argument selects preconditioner arithmetic, normally `PreconditionerReal`:

<!-- example: precision_choices -->
```cpp
la::CsrMatrix<double, Exec> A(pattern);
la::SolverOptions options;
options.method = "FGMRES";
options.preconditioner.type = "RELAXATION";

la::Solver<decltype(A), double> same_precision(A, options);
la::Solver<decltype(A), float> mixed_precision(A, options);
// Solver<decltype(A)> would use krona::PreconditionerReal.
```

These constructors do not by themselves prepare either preconditioner. The example shows alternative choices; an application normally keeps the one it intends to use. BSR uses the same precision selection.

| Solver scalar / preconditioner scalar | Coefficient storage and operator route |
| --- | --- |
| `double / double` | Original native matrix shared with preconditioner; no precision adapter |
| `float / float` | Original native matrix shared with preconditioner; no precision adapter |
| `double / float` | Additional float coefficient storage, float inner operator, conversion adapter |
| `float / double` | Additional double coefficient storage, double inner operator, conversion adapter |

The choice is compile-time automatic: no manual adapter switch is needed. Native graph storage is shared where the native format permits it; coefficients of differing scalar types cannot share the same array.

Preconditioner storage is allocated lazily during actual enabled preparation. Selecting different scalars does not allocate conversion buffers for a disabled preconditioner or a globally empty system. A solve that accepts its initial guess also skips preparation unless the application explicitly requested it earlier.

### 11.2 Matrix conversion during preparation

With equal scalars, `preconditioner_matrix_ = A.native()` shares the original native object through `RCP`. It does not construct another coefficient array.

With different scalars, the preconditioner lazily creates another native matrix in its own precision, on the same sparsity structure and with the same format: CSR remains CSR and BSR remains BSR. After every collectively detected coefficient change, one device-accessible kernel casts all local coefficients into that matrix.

This conversion also checks that the converted values are finite. Overflow from finite double values to float infinity is therefore rejected during preparation. The check does not diagnose loss of significant digits, underflow to zero, poor conditioning, or an unsuitable preconditioner. Its reduction result is a host scalar, making this part of the blocking setup boundary.

The original solver matrix and Krylov arithmetic retain the solver scalar. The original operator is used for `A*x` and the final true residual.

### 11.3 `PrecisionOperator::apply()`

The public preconditioner operator must accept the same scalar as Belos. Internally, `detail::PrecisionOperator<Scalar, InnerScalar, Exec>` adapts a native operator in `InnerScalar`:

1. Convert the input multivector from solver precision to a cached inner-precision buffer.
2. Apply the inner operator in its precision, producing another inner buffer.
3. Convert the result back to solver precision.
4. Apply the requested `alpha`/`beta` combination in solver precision.

For the common `alpha = 1`, `beta = 0` case, the converted result goes directly into the output. A nonzero `beta` requires a cached solver-precision temporary so the old output is preserved for the final update. An `alpha == 0` call skips the inner application and only handles the output scaling.

Inner input/output buffers are allocated lazily and reused. A changed multivector width or transpose orientation can require reallocation. The adapter forwards domain/range maps and transpose capability from the inner operator; it does not introduce transpose support when the inner operator lacks it. It is a general native-operator adapter, not a CSR-only implementation.

Conversions occur on **each preconditioner application**, not on every original-matrix SpMV. They are memory traffic and additional native operations, even when no host staging occurs. Lower-precision factors/hierarchies may save memory or work, but whether that outweighs conversion costs is problem/backend dependent. Equal-precision configurations bypass this adapter entirely.

## 12. Integration into a mesh-based solver

### 12.1 Build the algebraic topology once

The mesh adapter preserves KRONA's original local cell order. It creates rank-contiguous owned numbering using a prefix sum, then obtains ghost-cell GIDs through one mesh-halo exchange on a private duplicated MPI context. No global per-cell allgather table is built.

`CellNumbering` contains:

| Member | Meaning |
| --- | --- |
| `n_owned` | Number of locally owned cells |
| `owned_begin` | First rank-contiguous owned GID |
| `global_cells` | Global owned-cell count |
| `cell_to_global` | GID for each owned and MPI-ghost cell, in original local order |
| `owned_gids()` | Span over the owned prefix only |
| `expanded_gids(B)` | Host cell-major scalar GIDs for **all** stored cells, including MPI ghosts |

`build_cell_numbering(mesh, comm)` is collective and blocking. It uses the mesh's halo plan and checks its relevant consistency. It does not require geometry or device mesh storage.

`make_face_csr(mesh, numbering)` constructs a host graph containing each owned diagonal and real face-neighbor couplings. Internal owned-owned faces contribute both directions. MPI neighbors become remote columns. Physical boundary slots are excluded. The helper sorts/deduplicates row neighbors, so its output satisfies the pattern's no-duplicates contract.

This is a graph choice, not coefficient or boundary-condition assembly. Wider stencils, constraints, additional couplings, or a different unknown placement require an application-constructed `CsrInput`.

For a scalar system, use this graph directly as a `CsrPattern`. For coupled BSR, wrap the same cell graph in `BsrPattern`. For a deliberately scalar-expanded coupled CSR system, construct the expanded point topology yourself; `expanded_gids()` supplies numbering, not a full expanded graph.

**Do not pass the full `expanded_gids(B)` result as owned IDs.** It includes ghost cells. The owned prefix has `n_owned * B` entries. BSR's own `distribution()` already provides the correct owned point map.

### 12.2 A reusable coupled solve loop

The following complete template demonstrates topology reuse with a five-component BSR system. Its two callbacks belong to the application, not to the linalg API:

- `assemble_system(A, exec)` refreshes all matrix coefficients and the residual fields captured by the supplied pack. It submits their producer kernels on `exec`, using the matrix's accessors for coefficient writes.
- `apply_update(exec)` consumes the unpacked increment fields and submits the application update on the same instance.

The supplied residual fields must already represent the intended RHS, including its sign. `increments` describes separate writable destination fields. An increment solve starts from zero on every iteration here; for an absolute-unknown solve, a retained initial guess can be preferable.

<!-- example: coupled_iterations -->
```cpp
template<class ResidualPack, class IncrementPack,
         class AssembleSystem, class ApplyUpdate>
la::SolveResult coupled_iterations(const krona::mesh::MeshPart& mesh,
                                  krona::mpi::CommView comm, int iterations,
                                  ResidualPack residuals, IncrementPack increments,
                                  AssembleSystem assemble_system,
                                  ApplyUpdate apply_update)
{
    static_assert(ResidualPack::components == 5);
    static_assert(IncrementPack::components == 5);
    krona::mpi::require(iterations > 0, "example: positive iteration count required");
    Exec exec;
    const auto numbering = la::build_cell_numbering(mesh, comm);
    la::Distribution<Exec> cells(numbering.owned_gids(), comm);
    const auto input = la::make_face_csr(mesh, numbering);
    la::CsrPattern<Exec> cell_graph(cells, input, exec);
    la::BsrPattern<Exec> pattern(cell_graph, 5);
    la::BsrMatrix<S, Exec> A(pattern);
    la::Vector<S, Exec> rhs(A.distribution()), delta(A.distribution());

    la::SolverOptions options;
    options.method = "FGMRES";
    options.relative_tolerance = 1e-10;
    options.preconditioner.type = "SCHWARZ"; // defaults to block RBILUK, overlap 0
    la::Solver<decltype(A), S> solver(A, options);

    la::SolveResult result;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        assemble_system(A, exec);
        la::copy_from_fields(rhs, residuals, exec);
        delta.fill_async(S{0}, exec);
        A.commit();
        rhs.commit();
        delta.commit();

        result = solver.solve(rhs, delta);
        if (!result.converged) return result;

        la::copy_to_fields(delta, increments, exec);
        apply_update(exec);
        delta.commit(); // same instance: completes unpack and application update
    }
    return result;
}
```

The example takes a fixed iteration count to show the integration sequence. A real nonlinear/time-stepping driver chooses convergence criteria, time-step control, halo exchanges, boundary updates, and coefficient construction. If the update needs neighbor or boundary data, establish those application dependencies at the appropriate point; the linalg loop does not supply them.

The solver automatically observes matrix revisions during these iterations. If only the RHS changes, avoid obtaining a mutable matrix view: then the prepared preconditioner can remain unchanged. A read-only inspection of coefficients does not invalidate preparation, but still creates an access batch requiring commit.

### 12.3 Multiple matrices and topology changes

Reuse a single `CsrPattern` or `BsrPattern` for several matrices with the same topology. Coefficients are independent; maps and graph descriptors are shared. For example, separate Jacobian and auxiliary operators can use the same block pattern, with separate solvers if needed.

Movement that changes coefficients while preserving graph and distribution can be handled as a numerical update. A changed stencil, repartitioning, adaptive topology, ownership, or number of unknowns per cell requires rebuilding the corresponding patterns, matrices, vectors, solver, and application slot lookup. The module does not currently update an existing immutable pattern in place or transfer solutions across a changed distribution.

## 13. Native behavior and hidden costs

This section collects costs that are easy to miss when reading the wrapper interface.

### 13.1 Native mirrors and view acquisition

Tpetra controls native memory and host/device coherence. A device accessor can synchronize host data to the device before returning. A host accessor can synchronize in the opposite direction and may allocate a mirror lazily. Returning a Kokkos view is therefore not an unconditional guarantee of no allocation or no synchronization.

CSR coefficient and vector `OverwriteAll` access tell Tpetra that old host values do not need to be preserved before a full device overwrite. The application must actually overwrite all entries. A read/write accessor retains existing data and may need the corresponding synchronization.

Trilinos 17.2.1 exposes only read/write mutable coefficient access for `BlockCrsMatrix`. Consequently, BSR `overwrite_values()` and `overwrite_blocks()` currently delegate to read/write access. An old, newer host copy can still be transferred even when the application intends to overwrite everything.

Host `set_values()` returns after the host writes; it does not promise that a GPU copy has already been materialized. A later native device operation may perform that transfer. BSR's host setter also uses native read/write access and can synchronize old device data before replacing it. The direct device assembly route avoids deliberately staging coefficients through an application host buffer.

### 13.2 Graph sharing, permutations, and BSR memory

CSR matrices share the native graph through `RCP`. BSR matrices share the device block graph, but Trilinos 17.2.1's native block matrix construction also copies host row-offset/column-index buffers for each native block matrix instance. Mixed-precision BSR storage therefore has another such native host-index allocation.

Retaining the original-input permutation costs approximately one `size_t` per graph entry on the device and a retained host representation, subject to memory-space aliasing. Its construction also uses temporary host topology/mapping data. Set `keep_input_mapping = false` only when all later assembly uses finalized native order.

BSR reduces **index** storage for dense coupled blocks, not the number of coefficients in those blocks. A dense `B x B` block stores `B²` values in either a fully expanded CSR representation or BSR. BSR stores one block-column index where expanded CSR stores `B²` scalar-column indices, and has fewer row offsets. Zeros inside a stored dense block still consume coefficient storage and arithmetic.

For illustration, with `B = 5`, double values, and four-byte local column indices, one fully occupied block contributes roughly `25 * (8 + 4) = 300` bytes of values plus column indices in expanded CSR versus `25 * 8 + 4 = 204` bytes in BSR. This excludes row pointers, maps, host mirrors, alignment, and solver/preconditioner workspace. It is not a factor-of-25 reduction in total matrix memory.

BSR also needs expanded point maps and scalar vectors with `B` entries per owned cell. Pattern creation builds expanded GIDs on the host. Reusing a `BsrPattern` shares this map construction result.

### 13.3 Temporary CSR expansion during BSR AMG setup

The wrapper does not permanently flatten its BSR solver matrix into CSR. The mixed-precision matrix, if needed, also retains BSR format. The exercised default MueLu route retains block matrices in its hierarchy.

However, in Trilinos 17.2.1, native Xpetra block matrix products used in coarse-operator construction can convert block operands to scalar CSR, compute the product, and convert the result back to block storage. The relevant native paths are in `Xpetra_MatrixMatrix_def.hpp` and `Xpetra_TripleMatrixMultiply_def.hpp`; they can emit a block-to-point conversion warning during setup.

This creates temporary expanded indices and matrices, increasing setup time and peak memory. It occurs while constructing/updating hierarchy products, rather than implying that every original BSR SpMV or every preconditioner application expands the original matrix. The extent depends on the selected native hierarchy/product path.

The current block route also requires `use kokkos refactor = false`. Portable execution of the wrapper does not imply that this native setup path has been optimized or benchmarked on every accelerator backend. Full hierarchy rebuilds can pay these costs again after coefficient changes; explicit reuse changes which native components are rebuilt but has its own numerical tradeoffs.

### 13.4 Mixed-precision storage and traffic

When scalars differ, account for:

- another coefficient array in preconditioner precision;
- format-dependent native metadata, including the BSR host-index caveat above;
- one coefficient conversion/check pass per actual preparation update;
- cached inner-precision input/output multivectors;
- two vector conversions per ordinary preconditioner application;
- an additional solver-precision temporary for a nonzero `beta` application;
- all normal factors, hierarchy levels, and native workspace of the chosen preconditioner.

The `ifpack_`, `amg_`, `inner_`, and public `operator_` handles are not themselves four copies of the numerical data. In the active equal-precision route, several handles share an object; with mixed precision, the public operator is an adapter referencing the inner one. Factors and AMG levels are nevertheless real numerical allocations independent of reference-counting overhead.

### 13.5 Blocking native operations and workspace

Custom assembly can use a selected execution instance, but the native solve/preconditioner interfaces here do not accept that instance as an asynchronous submission stream. Commit before crossing that boundary.

Global Kokkos fences are present in pattern completion, matrix construction, matrix `apply()`, actual preconditioner preparation, and the iterating solve completion path. They can wait for unrelated work. `commit()` itself only fences the recorded instance. Empty, already-converged, and zero-iteration solve paths do not perform the same final global fence; their native norm results still require the relevant completed computations.

Belos allocates its own Krylov workspace. GMRES memory grows with restart length; FGMRES additionally retains preconditioned basis vectors. Ifpack2/MueLu allocate their own work arrays, factors, and hierarchy levels. Tpetra imports and the precision adapter can allocate lazily. `prepare()` is therefore not a guarantee of allocation-free subsequent solves.

The wrapper's true-residual vector and extra SpMVs/norms are intentional verification costs, described in section 9. A larger restart, stronger factorization, or mixed-precision setup should be assessed together with its iteration count and memory footprint.

## 14. Contracts, diagnostics, and extension points

### 14.1 Collective and lifetime contracts

- All ranks in the distribution communicator participate in collective construction, preparation, matrix application, and solving, including ranks with no owned rows.
- Use consistent solver/preconditioner selections and compatible options across ranks. Inconsistent branches can lead to mismatched native collectives.
- Keep the MPI communicator, runtime, matrix, and application view storage alive for their required use. Solvers/preconditioners must be destroyed before their referenced matrix, and all native objects before runtime finalization.
- Commit all wrappers whose custom reads/writes must finish before native work. RHS and solution must be distinct native vectors; in-place matrix application is also rejected.
- Matrix/vector map identity must match within the wrapper API. A custom preconditioner must instead pass the native `isSameAs` point-map checks.
- Solver, preconditioner, and mixed-precision adapter workspace are not reentrant. Concurrent access to the same instance is unsupported. MPI calls must also respect the application's provided MPI thread level.

### 14.2 Current numerical and interface limits

The wrapper supports real `float`/`double`, fixed immutable graphs, uniform square BSR blocks, square systems with one shared row/domain/range ownership distribution, and a single RHS per `solve()` call. Native multivector types exist for interoperability and preconditioner application; there is no wrapper multi-RHS solve method.

There is no wrapper transpose solve or transpose matrix application. Trilinos 17.2.1's native block matrix does not provide transpose apply for this route; a transpose-dependent native algorithm cannot be assumed to work with BSR merely because Belos can create its manager.

The application remains responsible for correct discretization, boundary contributions, conservation, constraint treatment, pressure gauges/nullspaces, matrix symmetry/definiteness, and a mathematically suitable preconditioner. The mesh adapter supplies connectivity, not those decisions. No wrapper-side coordinates/nullspace construction or physics-specific AMG setup is supplied.

The mixed-precision coefficient conversion checks finiteness after casting. This is not a general matrix validator: equal-precision preparation does not run that conversion check, and packing/copy helpers do not diagnose every nonfinite or lossy scalar conversion.

### 14.3 Common diagnostics

| Symptom/diagnostic | Meaning and action |
| --- | --- |
| `pending reads or writes` | A tracked batch remains open; commit the affected wrapper, including after reads/extraction |
| `commit before switching execution instances` | Another instance was supplied while a batch was pending |
| Vectors must share matrix distribution | Construct vectors from `A.distribution()` or a copied common descriptor; avoid independent equivalent maps/expansions |
| Duplicate column | Consolidate topology to one slot per row/column; accumulate contributions into that coefficient |
| Column has no owner | Check numbering, remote GIDs, communicator, and ownership; physical ghost slots are not algebraic unknowns by default |
| Original input mapping disabled | Use native-order assembly or rebuild the pattern with retained mapping |
| Preconditioner maps do not match | Use a distributed wrapper such as Schwarz for local factors, or correct a custom operator's point maps |
| BSR Schwarz overlap rejection | Use overlap zero or choose a different supported representation/preconditioner |
| BSR AMG equation-count/refactor rejection | Match the block size and keep `use kokkos refactor = false` |
| Nonfinite in preconditioner precision | A converted matrix value is NaN/Inf, possibly due to narrowing overflow |
| Stopping tolerance not representable | Choose a meaningful target in the actual solver scalar |
| `krylov_converged` differs from `converged` | Inspect the independently recomputed true residual and conditioning/tolerance settings |
| Extra setup after an apparent no-op | Acquiring a mutable matrix view increments its revision; use a read-only view for inspection |

Invalid contracts and native exceptions are handled through the project's `mpi::require` / `mpi::guarded` error path, rather than a recoverable wrapper error object. An ordinary completed solve that does not meet the residual target returns `converged == false`; native failures such as invalid settings or some numerical breakdowns can instead enter the error path.

### 14.4 What can be extended without changing the current contracts

The current factory interface allows additional native preconditioner operators, while the matrix-template solver separates algorithm selection from CSR/BSR storage. The field-pack interface already accepts different component organizations without changing algebraic ordering.

Potential future work includes native BSR setup paths that avoid temporary CSR products, genuine block `OverwriteAll` access when available upstream, additional validated native solver/preconditioner combinations and backends, application-specific AMG data, and explicit support for changing graph/distribution lifetimes. These are extension opportunities, not capabilities promised by the present implementation.

### 14.5 Example and backend verification

The C++ examples in this document were compiled against the current module and Trilinos 17.2.1 with the Serial backend. The example harness was run on one, two, and three MPI ranks, including settings, factories, mixed precision, field transfers, and the integration loop. The complete CSR/BSR examples were checked against their all-ones exact solution. This validates the documented interfaces and these numerical examples; it is not an accelerator-performance claim or a validation of every native parameter combination.


