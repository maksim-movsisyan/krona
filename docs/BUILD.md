# KRONA Build Foundation

The root `CMakeLists.txt` replaces the reference build from scratch. It contains
no partitioner/solver implementation libraries or application executables.
The `INTERFACE` targets only carry configuration and dependency requirements;
they do not compile libraries. No `cmake/` directory or external header template
is required.

Requirements: CMake 3.28 or newer, a C++20-capable dependency-compatible compiler,
and the installed MPI/HDF5/Kokkos stack. The default configuration also requires
parallel CGNS, distributed KaMinPar/TBB, and Trilinos. Check `cmake --version`:
the stock CMake package on Ubuntu 22.04 is older than this minimum.

## Configure against the installed dependencies

Run from the KRONA repository root. The following command uses the unified
Trilinos installation from the installation guide, including its bundled Kokkos.
Even when KRONA selects Serial, that GPU-enabled dependency installation needs
a compatible CUDA compiler arrangement. Compiler selection must be supplied
before CMake's first `project()` call, not changed inside `CMakeLists.txt`.

```bash
OMPI_CXX="$PWD/deps/src/trilinos-17.2.1/packages/kokkos/bin/nvcc_wrapper" \
cmake -S . -B build/serial-fp64 -G Ninja \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$PWD/deps/install/trilinos-17.2.1;$PWD/deps/install/cgns-4.5.2-i64;$PWD/deps/install/kaminpar-3.7.3" \
    -DKRONA_EXECUTION_SPACE=SERIAL \
    -DKRONA_PRECISION_PROFILE=FP64 \
    -DKRONA_LOCAL_INDEX_BITS=32 \
    -DKRONA_GLOBAL_INDEX_BITS=64
```

The baseline selects static parallel HDF5 to match the CGNS installation in the
setup document. It does not replace or rebuild any installed dependency.
If discovery selects the wrong HDF5, supply `HDF5_ROOT` and/or
`HDF5_C_COMPILER_EXECUTABLE` for the intended parallel installation in a fresh
build directory.

When implementation targets are added, build using the same compiler override:

```bash
OMPI_CXX="$PWD/deps/src/trilinos-17.2.1/packages/kokkos/bin/nvcc_wrapper" \
    cmake --build build/serial-fp64 --parallel 2
```

Today, configuration performs dependency compile/link probes, generates the
typed configuration header, and creates metadata targets. The subsequent build
has no application source to compile. `BUILD_TESTING` is available, but no KRONA
tests have been registered. Install/export rules are deferred until there is
an actual public library API to install.

## Configuration parameters

| Parameter | Values | Default |
|---|---|---|
| `KRONA_EXECUTION_SPACE` | `SERIAL`, `OPENMP`, `CUDA`, `HIP` | `SERIAL` |
| `KRONA_LOCAL_INDEX_BITS` | `32`, `64` | `32` |
| `KRONA_GLOBAL_INDEX_BITS` | `32`, `64` | `64` |
| `KRONA_PRECISION_PROFILE` | `FP32`, `FP64`, `MIXED` | `FP64` |
| `KRONA_ENABLE_CGNS` | `ON`, `OFF` | `ON` |
| `KRONA_ENABLE_PARTITIONING` | `ON`, `OFF` | `ON` |
| `KRONA_ENABLE_TRILINOS` | `ON`, `OFF` | `ON` |
| `KRONA_WARNINGS_AS_ERRORS` | `ON`, `OFF` | `OFF` |
| `BUILD_TESTING` | `ON`, `OFF` | CTest default: `ON` |

Disabling an optional dependency boundary skips its discovery and validation.
For example, `KRONA_ENABLE_TRILINOS=OFF` still requires Kokkos, MPI, and parallel
HDF5, but does not require the linear algebra stack. These switches are not
claims that unimplemented physics or partitioning modules are available.

The selected execution space is checked against the **installed** Kokkos and
Tpetra capabilities. Setting a KRONA parameter does not enable a backend inside
an already-built dependency. Architecture settings such as `AMPERE86` belong to
that dependency build, not to a second set of conflicting KRONA flags.

For OpenMP or CUDA, keep the same dependency prefix and select a fresh KRONA
build directory with `KRONA_EXECUTION_SPACE=OPENMP` or `CUDA`. This foundation
selects one KRONA execution type at compile time; runtime dispatch can be added
at the application boundary later.

For HIP, use the HIP-enabled Trilinos installation, set
`KRONA_EXECUTION_SPACE=HIP`, and select the compatible compiler before configuring
the fresh build directory. With OpenMPI and a typical ROCm location, the
command-local override is `OMPI_CXX=/opt/rocm/bin/hipcc`. Use the actual path.
Do not reuse a CUDA compiler cache for HIP.

## Numerical precision roles

`FP32` sets every role to `float`; `FP64` sets every role to `double`.
The initial reference profile is FP64. The MIXED profile exposes six independent
cache parameters:

| MIXED parameter | Generated alias | Default |
|---|---|---|
| `KRONA_MIXED_STORAGE_PRECISION` | `krona::StorageReal` | `FP32` |
| `KRONA_MIXED_COMPUTE_PRECISION` | `krona::ComputeReal` | `FP64` |
| `KRONA_MIXED_ACCUMULATION_PRECISION` | `krona::AccumulationReal` | `FP64` |
| `KRONA_MIXED_GEOMETRY_PRECISION` | `krona::GeometryReal` | `FP64` |
| `KRONA_MIXED_SOLVER_PRECISION` | `krona::SolverReal` | `FP64` |
| `KRONA_MIXED_PRECONDITIONER_PRECISION` | `krona::PreconditionerReal` | `FP32` |

Each MIXED parameter accepts `FP32` or `FP64`. Its cached value is inactive
while a uniform profile is selected. Returning to MIXED reuses the saved role
choices. The configuration summary prints the effective types.

For a GPU mixed-precision build, change these arguments in the full configure
command above:

```text
-B build/cuda-mixed
-DKRONA_EXECUTION_SPACE=CUDA
-DKRONA_PRECISION_PROFILE=MIXED
-DKRONA_MIXED_COMPUTE_PRECISION=FP32
```

The default MIXED profile is deliberately conservative about arithmetic and
reductions. This is a build-time type contract, not an implemented or numerically
validated mixed-precision algorithm. A different-precision preconditioner needs
an explicit operator adapter; Belos/Ifpack2 do not automatically convert it.

Future kernels must use the role aliases explicitly. Reduction variables and
expression operands must be cast to their intended arithmetic type; storing a
result in `double` does not retroactively evaluate a `float` expression in double
precision. Future wire formats and checkpoint schemas must record their actual
types rather than assume that `sizeof(Real)` defines a file format.

There is intentionally no universal `Real` alias, FP16/BF16 switch, or implicit
fast-math setting. A profile named MIXED can be tuned to use one scalar type for
every role; `krona::build::uses_multiple_real_types` reports the actual result.

## Index contracts and ABI boundaries

The generated aliases `krona::LocalIndex` and `krona::GlobalIndex` are independent
signed `std::int32_t`/`std::int64_t` choices. CSR offsets use the local-width
contract; validate total connectivity lengths as well as entity counts.

Changing either width changes KRONA's type/ABI contract. Rebuild all dependent
KRONA modules together. Separate build directories are recommended for backend,
index, or precision profiles. Do not link implementation libraries from different
profile builds just because their names match.

These settings do **not** change `cgsize_t`, KaMinPar identifiers, Tpetra ordinals,
MPI counts, or the mesh file schema. Their adapters must check conversions and
handle their own bounds independently.

The first Tpetra adapter will use `int` local ordinals and `long long` global
ordinals, matching the setup document's ETI. A 64-bit KRONA LocalIndex remains
legal, but a matrix whose local indexing exceeds Tpetra's `int` range cannot be
represented by that adapter. A 32-bit KRONA GlobalIndex can be widened at this
boundary. `std::int64_t` and `long long` are not assumed to be the same C++ type.

## Target boundaries

| Target | Intended dependency boundary |
|---|---|
| `krona::build_config` | C++20 and generated fundamental type aliases |
| `krona::warnings` | KRONA-only compiler diagnostics |
| `krona::execution` | Build configuration and Kokkos |
| `krona::mpi` | MPI C API accessed from C++ |
| `krona::hdf5` | Parallel HDF5 and MPI |
| `krona::cgns` | Parallel CGNS; available when enabled |
| `krona::partitioning` | Distributed KaMinPar, TBB, and MPI; available when enabled |
| `krona::linear_algebra` | Selected Trilinos packages; available when enabled |

Link dependencies where they are actually needed. A geometry or topology module
does not automatically acquire the full Trilinos or partitioning stack.

When adding a real module whose public headers use KRONA types, link
`krona::build_config` publicly. Link `krona::warnings` privately so diagnostics
are not imposed on consumers. Kokkos in public headers similarly requires public
propagation of `krona::execution`. No global include directories, link directories,
compiler definitions, or blanket optimization flags are used.

The generated header lives at `build/<configuration>/generated/include/krona/config.hpp`.
Do not copy it into the source tree or edit it manually. It records effective
types and profile metadata, not mutable mesh state.

## Dependency validation and its limits

Configuration compiles and links probes for the selected Kokkos execution type,
parallel HDF5 API, parallel CGNS API when enabled, and required Tpetra scalar/Node
combinations when enabled. Probe results are refreshed on reconfiguration.
No probe executable is run: configuration does not launch MPI ranks or require
executing a GPU kernel on the configure host.

CGNS export metadata is preserved. Missing native HDF5 static-target names are
bridged to the selected static HDF5 target. For installations without a usable
CGNS config file, the upstream targets/version files are loaded explicitly;
dependency lists are never erased to make a link succeed.

These probes do not establish MPI runtime ABI compatibility, exercise KaMinPar's
API, or validate numerical accuracy/performance. Real multi-rank and GPU tests
must accompany the corresponding implementation modules.

Mesh movement, adaptation, migration, and cache invalidation are future runtime
contracts. They will be defined with the mesh architecture, not represented by
build switches that pretend those features already exist.

## Upstream references

- https://kokkos.org/kokkos-core-wiki/get-started/integrating-kokkos-into-your-cmake-project.html
- https://github.com/trilinos/Trilinos/tree/master/demos/simpleBuildAgainstTrilinos
- https://cmake.org/cmake/help/latest/module/CheckCXXSourceCompiles.html
- https://github.com/CGNS/CGNS
- https://github.com/KaHIP/KaMinPar
