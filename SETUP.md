# KRONA Installation and Environment Setup

This document describes the dependency environment for KRONA.

Target platform: Linux, with Ubuntu/Debian package commands.

Target toolchain:

- C++20 and GCC;
- OpenMPI;
- parallel HDF5;
- CGNS with parallel I/O;
- distributed KaMinPar;
- Kokkos and selected Trilinos packages;
- CMake and Ninja.

## 1. Directory Layout and Build Conventions

Unless stated otherwise, run all commands from the `deps` directory,
which initially contains:

- `src`: dependency source trees;
- `install`: installed headers, libraries, executables, and CMake packages.

Initialize these directories if necessary:

```bash
mkdir -p src install
```

Build directories are created inside the corresponding source trees,
for example:

- `src/cgns-4.5.2/build-i32`;
- `src/kaminpar-3.7.3/build`;
- `src/trilinos-17.2.1/build`;

Commands use `$PWD` to form an absolute installation path from the
current directory. No persistent dependency environment variables are
required.

Do not switch directories halfway through a command sequence: `$PWD`
must still refer to `deps`.

## 2. Version Policy

The following versions are pinned for a reproducible initial setup:

| Dependency | Version or source |
|---|---|
| OpenMPI | Distribution package |
| Parallel HDF5 | Distribution package |
| CGNS | 4.5.2 |
| KaMinPar | 3.7.3 |
| Trilinos | 17.2.1 |
| Kokkos | 5.2.1, bundled with Trilinos |
| Kokkos Kernels | 5.2.1, bundled with Trilinos |
| toml++ | 3.4.0, vendored header |

These versions were selected using upstream information available on
2026-10-01. They are not a promise that independently updating each
dependency will preserve compatibility.

## 3. Base Packages

```bash
sudo apt update && sudo apt upgrade

sudo apt install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    pkg-config \
    curl \
    ca-certificates \
    openmpi-bin \
    libopenmpi-dev \
    libhdf5-openmpi-dev \
    hdf5-tools \
    libtbb-dev \
    libopenblas-dev \
    liblapack-dev \
    libgtest-dev \
    zlib1g-dev \
    libaec-dev \
    clangd \
    clang-format \
    libsparsehash-dev
```

Package roles:

- `build-essential` provides the core compilation tools (such as `gcc`, `g++`, and `make`) required to build C/C++ software from source.
- `cmake` is a cross-platform build system generator used to manage and automate the software compilation process.
- `git` is a distributed version control system for tracking changes and managing source code during development.
- `ninja-build` is a small, fast build system designed to speed up compilation, often used as a backend generator for CMake.
- `pkg-config` is a helper tool used during compilation to provide the correct compiler and linker flags for installed libraries.
- `curl` is a command-line tool for transferring data with URLs, used here for the optional `toml++` download.
- `ca-certificates` provides a standard set of Certificate Authority (CA) certificates required for secure HTTPS connections (e.g., when `curl` downloads files).
- `openmpi-bin` provides the core executables and runtime binaries for Open MPI, enabling the parallel execution of MPI applications.
- `libopenmpi-dev` provides the development headers and libraries needed to compile and link software against the Open MPI implementation.
- `libhdf5-openmpi-dev` provides HDF5 built with parallel I/O capabilities using OpenMPI for high-performance parallel data access.
- `hdf5-tools` provides command-line utilities such as `h5dump` and `h5ls` for inspecting, manipulating, and converting HDF5 files.
- `libtbb-dev` provides the Intel Threading Building Blocks library, which offers shared-memory parallel tasking used by components like KaMinPar.
- `libopenblas-dev` provides an optimized implementation of Basic Linear Algebra Subprograms (BLAS), supplying high-performance CPU numerical routines.
- `liblapack-dev` provides the Linear Algebra PACKage for solving systems of linear equations and eigenvalue problems, used alongside OpenBLAS by selected Trilinos packages.
- `libgtest-dev` provides the Google Test framework, making it available for compiling and running unit and dependency tests.
- `zlib1g-dev` provides the development files for the zlib compression library, a dependency required when linking HDF5 and CGNS statically.
- `libaec-dev` provides the development files for the Adaptive Entropy Coding library, another compression dependency required for static linking with HDF5 and CGNS.
- `clangd` is a language server that provides IDE features like code completion, go-to-definition, and real-time diagnostics for C/C++ code.
- `clang-format` is a tool to automatically format C/C++ code according to specified style guidelines, ensuring consistent code formatting across the project.
- `libsparsehash-dev` provides an extremely memory-efficient hash map implementation for C++, optimized to store large amounts of data with minimal memory overhead.

## 4. CGNS with Parallel I/O

A package named `libcgns-dev` does not by itself guarantee that parallel
I/O is enabled. Building CGNS from source provides explicit control over
this configuration.

The following setup preserves the original partitioner's 32-bit
`cgsize_t` choice. The installation name includes `i32` to make that
choice visible.

### 4.1 Download, configure, build, and install

Run from `deps`:

```bash
git clone --depth 1 --branch v4.5.2 \
    https://github.com/CGNS/CGNS.git \
    src/cgns-4.5.2

cmake -S src/cgns-4.5.2 \
    -B src/cgns-4.5.2/build-i32 \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX="$PWD/install/cgns-4.5.2-i32" \
    -DCGNS_ENABLE_HDF5=ON \
    -DCGNS_ENABLE_PARALLEL=ON \
    -DHDF5_NEED_MPI=ON \
    -DHDF5_PREFER_PARALLEL=ON \
    -DHDF5_C_COMPILER_EXECUTABLE="$(command -v h5pcc)" \
    -DCMAKE_PREFIX_PATH="/usr/lib/$(gcc -print-multiarch)/hdf5/openmpi" \
    -DCGNS_ENABLE_LFS=ON \
    -DCGNS_BUILD_SHARED=OFF \
    -DCGNS_ENABLE_FORTRAN=OFF \
    -DCGNS_ENABLE_TESTS=OFF \
    -DCGNS_ENABLE_64BIT=OFF

cmake --build src/cgns-4.5.2/build-i32 --parallel 2

cmake --install src/cgns-4.5.2/build-i32
```

The HDF5 prefix above follows the Ubuntu/Debian multiarch layout.
Replace it when using a custom HDF5 installation or a cluster module.

### 4.2 Explanation of the important options

| Option | Purpose |
|---|---|
| `CGNS_ENABLE_HDF5=ON` | Enables the HDF5 storage backend |
| `CGNS_ENABLE_PARALLEL=ON` | Enables the parallel CGNS API |
| `HDF5_NEED_MPI=ON` | Requests an MPI-enabled HDF5 dependency |
| `HDF5_PREFER_PARALLEL=ON` | Prefers parallel HDF5 during CMake module discovery |
| `HDF5_C_COMPILER_EXECUTABLE` | Identifies the intended HDF5 compiler wrapper |
| `CMAKE_PREFIX_PATH` | Adds the intended HDF5 installation to package searches |
| `CMAKE_C_COMPILER=mpicc` | Uses the selected MPI compiler wrapper |
| `CGNS_ENABLE_LFS=ON` | Enables large-file support |
| `CGNS_ENABLE_64BIT=OFF` | Selects 32-bit `cgsize_t` |
| `CGNS_BUILD_SHARED=OFF` | Builds a static CGNS library |
| `CMAKE_POSITION_INDEPENDENT_CODE=ON` | Allows use of static objects in shared-library contexts |

`HDF5_PREFER_PARALLEL` is a search preference, not a universal guarantee.
A previously cached result or an HDF5 package configuration can affect
discovery.

Inspect the configuration output and the selected HDF5 paths. Confirm
that parallel CGNS is actually enabled.

When changing HDF5 installations or index widths, configure a new build
directory instead of reusing a cache containing the old selection.

### 4.3 Index width and large meshes

`CGNS_ENABLE_64BIT` controls the CGNS API's `cgsize_t` type. It does not
control floating-point precision.

Use a 64-bit CGNS build when mesh sizes, connectivity lengths, offsets,
or other CGNS quantities cannot be represented by the 32-bit API.

For a 64-bit installation:

- use `CGNS_ENABLE_64BIT=ON`;
- use a separate build directory, such as `build-i64`;
- use a separate installation prefix, such as `cgns-4.5.2-i64`;
- rebuild the partitioner against that installation.

Do not mix headers from an i32 installation with an i64 library.

KRONA's `LocalIndex`, `GlobalIndex`, the CGNS API type, and the on-disk
integer representation are separate choices. The reader must handle
conversions and range checks explicitly.

A 32-bit API cannot safely represent an arbitrarily large mesh merely
because a slower reading path exists.

### 4.4 MIXED sections and the existing partitioner reader

`WARNING-1` : IF YOUR MESH CONTAINS MIXED SECTIONS, THEN -DCGNS_ENABLE_64BIT=OFF/ON - MUST BE CONSISTENT WITH FILE INTEGER PRECSISION, OTHERWISE THE CGNS-READER FALL BACK TO RAM HEAVY READ (IT WILL WORK BUT IT IS SLOWER AND CONTRIBUTE MORE RAM).

`WARNING-2` : IF YOUR MESH CONTAINS MIXED SECTIONS, THEN YOU FILE VERISION (FILE, NOT CGNS) MUST BE MORE OR EQUAL 4.*, OTHERWISE THE CGNS-READER FALL BACK TO RAM HEAVY READ (IT WILL WORK BUT IT IS SLOWER AND CONTRIBUTE MORE RAM).

### 4.5 Convert and update older files

The following example uses separate output files to preserve the input.
Adjust the mesh paths before running it.

Run from `deps`:

```bash
./install/cgns-4.5.2-i32/bin/cgnsconvert -f -h \
    ../mesh/sphere.cgns \
    ../mesh/sphere_hdf5.cgns

./install/cgns-4.5.2-i32/bin/cgnscheck -u \
    ../mesh/sphere_hdf5.cgns \
    ../mesh/sphere_updated.cgns

./install/cgns-4.5.2-i32/bin/cgnscheck \
    ../mesh/sphere_updated.cgns
```

Explanation:

- `cgnsconvert -h` selects HDF5 output.
- `-f` forces conversion even when the input already uses the requested
  storage format.
- `cgnscheck -u` updates and checks the file.
- The final command checks the resulting file again without requesting
  an update.

CGNS also provides the `cgnsupdate` utility.

Do not manually change the CGNS version field as a substitute for
updating the file structure.

After conversion, verify that the partitioner can read the file and
that the intended scalable MIXED reading path is selected.

## 5. Distributed KaMinPar

KaMinPar supports shared-memory and distributed-memory partitioning.

The distributed build must be enabled explicitly. MPI compiler wrappers
alone do not enable it.

### 5.1 Download, configure, build, and install

Run from `deps`:

```bash
git clone --recursive --depth 1 --branch v3.7.3 \
    https://github.com/KaHIP/KaMinPar.git \
    src/kaminpar-3.7.3

cmake -S src/kaminpar-3.7.3 \
    -B src/kaminpar-3.7.3/build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_INSTALL_PREFIX="$PWD/install/kaminpar-3.7.3" \
    -DKAMINPAR_BUILD_DISTRIBUTED=ON \
    -DINSTALL_KAMINPAR=ON \
    -DKAMINPAR_BUILD_WITH_MTUNE_NATIVE=OFF \
    -DBUILD_TESTING=OFF \
    -DKAMINPAR_BUILD_TESTS=OFF

cmake --build src/kaminpar-3.7.3/build --parallel 2

cmake --install src/kaminpar-3.7.3/build
```

### 5.2 Explanation

- `--recursive` initializes the repository's submodules.
- Additional dependencies may still be downloaded during configuration.
- `KAMINPAR_BUILD_DISTRIBUTED=ON` enables distributed partitioning.
- `INSTALL_KAMINPAR=ON` enables installation support.
- `KAMINPAR_BUILD_WITH_MTUNE_NATIVE=OFF` avoids specializing the
  installation for the build machine's CPU instruction set.

The native option is disabled deliberately for a portable baseline.
A separate machine-specific build can enable it after benchmarking.

The distributed CMake target is `KaMinPar::dKaMinPar`.

TBB remains a dependency even when the partitioner is configured to use
one thread per MPI rank. Set the thread count through the partitioner's
or KaMinPar's runtime configuration.

Do not assume that disabling OpenMP in Trilinos controls KaMinPar's TBB
threads.

KaMinPar index-width options must match the adapter used by the
partitioner. Distributed global identifiers and local identifiers are
not necessarily the same type.

Before adopting a different KaMinPar version, compile the existing
adapter and run a representative multi-rank partitioning case.

## 6. Kokkos and Trilinos: Installation Strategy

KRONA uses the Kokkos and Kokkos Kernels versions bundled with
Trilinos 17.2.1:

- Kokkos 5.2.1;
- Kokkos Kernels 5.2.1.

This keeps the dependency versions aligned with the selected Trilinos
release. Do not introduce a second, independently installed Kokkos into
this configuration.

KRONA can use Kokkos directly without making every computational module
depend on Tpetra or other Trilinos solver packages.

The following packages are enabled:

| Package | Intended role |
|---|---|
| Kokkos | Execution and memory portability |
| Kokkos Kernels | Portable numerical kernels |
| Tpetra | Distributed sparse matrices and vectors |
| Belos | Iterative linear solvers |
| Ifpack2 | Preconditioning |
| MueLu | Algebraic multigrid |

Required supporting packages are enabled automatically through Trilinos
dependencies.

Additional packages can be introduced when a concrete application needs
them.

### 6.1 Backend configuration

The initial workstation installation includes three execution backends:

| Backend | Hardware and execution model |
|---|---|
| Serial | Sequential CPU execution inside each MPI rank |
| OpenMP | Multithreaded CPU execution inside each MPI rank |
| CUDA | NVIDIA GPU execution |

MPI provides distributed execution between ranks and can be used with
any of these backends.

Kokkos supports Serial together with one host parallel backend and one
device backend. Therefore, Serial + OpenMP + CUDA is a valid combination.
Serial + OpenMP + HIP is also valid, but CUDA and HIP cannot be enabled
together in the same Kokkos build.

One installation can provide Serial, OpenMP, and CUDA implementations.
Separate CPU and CUDA installations are not required for this
workstation configuration.

A HIP installation must use a different configuration because it changes
the device backend and compiler toolchain.

### 6.2 Download Trilinos

Run from `deps`:

```bash
git clone --depth 1 --branch trilinos-release-17-2-1 \
    https://github.com/trilinos/Trilinos.git \
    src/trilinos-17.2.1
```

The initial configuration uses:

- build directory: `src/trilinos-17.2.1/build`;
- installation prefix: `install/trilinos-17.2.1`.

Both CPU and CUDA implementations belong to this installation.

The build directory is inside the downloaded repository but remains
separate from its source files. Do not configure directly into the
repository root.

## 7. Unified Configuration: MPI + Serial + OpenMP + CUDA

This configuration builds the selected Trilinos packages with:

- MPI support;
- Serial, OpenMP, and CUDA execution;
- float and double scalar support;
- Tpetra local ordinals of type `int`;
- Tpetra global ordinals of type `long long`.

### 7.1 Prerequisites

The toolchain must satisfy the requirements of the selected dependencies:

- CMake 3.27 or newer for Trilinos 17;
- C++20;
- a compatible GCC host compiler;
- CUDA Toolkit 12.2 or newer for Kokkos 5.x with NVCC;
- an NVIDIA driver compatible with the installed CUDA Toolkit.

Check the CUDA release's supported host compiler versions before
building. A compiler suitable for a CPU build may be too new for a
particular CUDA Toolkit.

Verify:

```bash
cmake --version
g++ --version
mpirun --version
mpicxx --showme:command
nvcc --version
nvidia-smi
```

`nvcc` must be available to Kokkos's compiler wrapper.

The CUDA version displayed by `nvidia-smi` describes driver
compatibility. Use `nvcc --version` to inspect the installed toolkit.

The configuration below targets the RTX 3060 using
`Kokkos_ARCH_AMPERE86=ON`. Change this option for another GPU architecture.

### 7.2 Configure

Run from `deps`:

```bash
OMPI_CXX="$PWD/src/trilinos-17.2.1/packages/kokkos/bin/nvcc_wrapper" \
cmake -S src/trilinos-17.2.1 \
    -B src/trilinos-17.2.1/build \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$PWD/install/trilinos-17.2.1" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DTPL_ENABLE_MPI=ON \
    -DTPL_ENABLE_BLAS=ON \
    -DTPL_ENABLE_LAPACK=ON \
    -DTPL_ENABLE_CUDA=ON \
    -DTrilinos_ENABLE_Fortran=OFF \
    -DTrilinos_ENABLE_ALL_PACKAGES=OFF \
    -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
    -DTrilinos_ENABLE_Kokkos=ON \
    -DTrilinos_ENABLE_KokkosKernels=ON \
    -DTrilinos_ENABLE_Tpetra=ON \
    -DTrilinos_ENABLE_Belos=ON \
    -DTrilinos_ENABLE_Ifpack2=ON \
    -DTrilinos_ENABLE_MueLu=ON \
    -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
    -DTrilinos_ENABLE_FLOAT=ON \
    -DTeuchos_ENABLE_FLOAT=ON \
    -DTpetra_INST_FLOAT=ON \
    -DTpetra_INST_DOUBLE=ON \
    -DTrilinos_ENABLE_COMPLEX_FLOAT=OFF \
    -DTrilinos_ENABLE_COMPLEX_DOUBLE=OFF \
    -DTrilinos_ENABLE_LONG_DOUBLE=OFF \
    -DTpetra_INST_INT_LONG_LONG=ON \
    -DTpetra_INST_INT_INT=OFF \
    -DTpetra_INST_INT_LONG=OFF \
    -DTrilinos_ENABLE_OpenMP=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_HIP=OFF \
    -DKokkos_ARCH_AMPERE86=ON \
    -DTpetra_INST_SERIAL=ON \
    -DTpetra_INST_OPENMP=ON \
    -DTpetra_INST_CUDA=ON \
    -DTpetra_INST_HIP=OFF \
    -DTpetra_ASSUME_GPU_AWARE_MPI=OFF \
    -DTrilinos_ENABLE_TESTS=OFF \
    -DTrilinos_ENABLE_EXAMPLES=OFF \
    -DTpetra_ENABLE_TESTS=ON \
    -DBelos_ENABLE_TESTS=ON \
    -DTrilinos_TEST_CATEGORIES=BASIC \
    -DMPI_EXEC_MAX_NUMPROCS=1
```

`OMPI_CXX` is set only for this command. No permanent shell environment
change is made.

For OpenMPI, this setting makes `mpicxx` invoke Kokkos's `nvcc_wrapper`
as its underlying C++ compiler. The wrapper coordinates CUDA and host
compilation.

`nvcc_wrapper` uses `g++` as its default host compiler. Ensure that this
compiler is compatible with the CUDA Toolkit and the MPI installation.

This mechanism is specific to OpenMPI. Other MPI implementations may
require a different compiler-selection mechanism.

### 7.3 Build, test, and install

Run from `deps`, proceeding only after each command succeeds:

```bash
OMPI_CXX="$PWD/src/trilinos-17.2.1/packages/kokkos/bin/nvcc_wrapper" \
    cmake --build src/trilinos-17.2.1/build --parallel 2

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    ctest --test-dir src/trilinos-17.2.1/build \
    --output-on-failure \
    --no-tests=error \
    --parallel 1

cmake --install src/trilinos-17.2.1/build
```

The compiler override is repeated during compilation because CMake
stores the `mpicxx` path, not a permanent `OMPI_CXX` environment setting.
Use the same override for subsequent rebuilds.

Two compilation jobs provide a conservative starting point for a
workstation. Increase the job count if sufficient RAM is available.

The test command limits OpenBLAS and OpenMP to one thread for this
baseline run. These settings apply only to that command.

`MPI_EXEC_MAX_NUMPROCS=1` limits the initial MPI test configurations to
one rank, avoiding an implicit multi-rank assignment to the workstation's
single GPU.

This baseline does not validate multi-rank communication or multithreaded
scaling. Add those tests with explicit thread counts and rank-to-device
assignment when the corresponding execution modes are introduced.

### 7.4 Build and dependency flags

| Flag | Explanation |
|---|---|
| `-S`, `-B` | Select the source and build directories |
| `-G Ninja` | Generates a Ninja build |
| `CMAKE_INSTALL_PREFIX` | Selects the installation directory |
| `CMAKE_BUILD_TYPE=Release` | Enables the toolchain's release optimization settings |
| `CMAKE_C_COMPILER=mpicc` | Uses the selected MPI C wrapper |
| `CMAKE_CXX_COMPILER=mpicxx` | Uses the selected MPI C++ wrapper |
| `CMAKE_CXX_STANDARD=20` | Requests C++20 |
| `CMAKE_CXX_STANDARD_REQUIRED=ON` | Prevents fallback to an older C++ standard |
| `CMAKE_CXX_EXTENSIONS=OFF` | Requests standard C++ rather than GNU language extensions |
| `CMAKE_POSITION_INDEPENDENT_CODE=ON` | Produces position-independent objects where supported |
| `BUILD_SHARED_LIBS=OFF` | Builds static libraries for packages honoring this setting |
| `TPL_ENABLE_MPI=ON` | Enables the MPI dependency |
| `TPL_ENABLE_BLAS=ON` | Enables BLAS integration |
| `TPL_ENABLE_LAPACK=ON` | Enables LAPACK integration |
| `TPL_ENABLE_CUDA=ON` | Enables Trilinos's CUDA dependency |
| `Trilinos_ENABLE_Fortran=OFF` | Disables Fortran language support in this build |
| `Trilinos_ENABLE_ALL_PACKAGES=OFF` | Avoids enabling every Trilinos package |
| `Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF` | Avoids enabling all optional package dependencies |

Disabling Fortran language support does not remove the BLAS/LAPACK
dependency. Prebuilt BLAS/LAPACK libraries may still require their own
runtime libraries.

The individual `Trilinos_ENABLE_<Package>=ON` options enable the packages
listed in Section 6 and their required dependencies.

Static linking still requires transitive dependencies at the final link
step. Use dependency-provided CMake targets and package information.

### 7.5 Backend flags

| Flag | Explanation |
|---|---|
| `Trilinos_ENABLE_OpenMP=ON` | Enables OpenMP support at the Trilinos level |
| `Kokkos_ENABLE_SERIAL=ON` | Builds Kokkos's sequential CPU backend |
| `Kokkos_ENABLE_OPENMP=ON` | Builds Kokkos's threaded CPU backend |
| `Kokkos_ENABLE_CUDA=ON` | Builds Kokkos's NVIDIA GPU backend |
| `Kokkos_ENABLE_HIP=OFF` | Keeps the alternative HIP device backend disabled |
| `Kokkos_ARCH_AMPERE86=ON` | Selects NVIDIA compute capability 8.6 |
| `Tpetra_INST_SERIAL=ON` | Builds Tpetra instantiations for the Serial Node |
| `Tpetra_INST_OPENMP=ON` | Builds Tpetra instantiations for the OpenMP Node |
| `Tpetra_INST_CUDA=ON` | Builds Tpetra instantiations for the CUDA Node |
| `Tpetra_INST_HIP=OFF` | Excludes Tpetra HIP instantiations |

Kokkos backend availability and Tpetra instantiation are separate
settings. Both must be enabled when KRONA needs Tpetra on a particular
backend.

The installation makes these implementations available. KRONA chooses
which implementation to use through its execution-space and Tpetra
Node types.

With Serial, OpenMP, and CUDA enabled:

- `Kokkos::DefaultExecutionSpace` is CUDA;
- `Kokkos::DefaultHostExecutionSpace` is OpenMP.

Consequently, use explicit execution types when KRONA needs Serial
execution rather than relying on a default type.

Selecting among already-built backends does not require rebuilding
Trilinos. KRONA can select a backend during compilation, or provide
runtime dispatch between implementations that it has compiled.

Enabling OpenMP does not convert Serial kernels into threaded kernels.
However, other OpenMP-enabled library operations may use threads.
Control thread counts when running the pure-MPI configuration.

A CUDA-enabled installation retains CUDA toolchain and runtime
dependencies. Prepare a CPU-only configuration when deploying to an
environment without the required CUDA support.

### 7.6 Precision and index flags

| Flag | Explanation |
|---|---|
| `Trilinos_ENABLE_EXPLICIT_INSTANTIATION=ON` | Precompiles selected template combinations into the libraries |
| `Trilinos_ENABLE_FLOAT=ON` | Requests float support in packages using this global setting |
| `Teuchos_ENABLE_FLOAT=ON` | Enables float support in Teuchos |
| `Tpetra_INST_FLOAT=ON` | Includes float Tpetra instantiations |
| `Tpetra_INST_DOUBLE=ON` | Includes double Tpetra instantiations |
| `Trilinos_ENABLE_COMPLEX_FLOAT=OFF` | Disables complex single-precision support through the global setting |
| `Trilinos_ENABLE_COMPLEX_DOUBLE=OFF` | Disables complex double-precision support through the global setting |
| `Trilinos_ENABLE_LONG_DOUBLE=OFF` | Disables extended-precision support through the global setting |
| `Tpetra_INST_INT_LONG_LONG=ON` | Includes local ordinal `int` with global ordinal `long long` |
| `Tpetra_INST_INT_INT=OFF` | Excludes the `int`/`int` ordinal combination |
| `Tpetra_INST_INT_LONG=OFF` | Excludes the `int`/`long` ordinal combination |

The intended real-valued Tpetra interface supports float and double
across all three enabled Nodes, with the selected ordinal pair.

More scalar/backend combinations increase compilation work and library
size. They do not automatically increase the cost of executing a
particular selected implementation.

Use the exact supported C++ types at the Tpetra interface.
`std::int64_t` can alias `long` rather than `long long`, and these are
different template types even when their widths match.

The local ordinal remains subject to the representable range of `int`.
A 64-bit global ordinal does not remove local indexing limits.

These settings do not prescribe KRONA's mesh index representation.
Adapters may convert between representations with explicit range checks.

Use the ordinary LP64 BLAS interface unless an ILP64 configuration is
deliberately supported throughout the stack. A 64-bit Tpetra global
ordinal does not require ILP64 BLAS.

### 7.7 Mixed precision in KRONA

This installation supplies float and double building blocks.
KRONA defines the numerical precision policy.

Kokkos Views and kernels use the types chosen by the application;
there is no global Kokkos flag that changes every field to FP32 or FP64.

Precision roles should be independent where the algorithms permit it:

- geometry storage;
- solution storage;
- flux evaluation;
- residual accumulation;
- matrix storage;
- preconditioner storage;
- convergence checks;
- halo communication.

For example, a mixed-precision linear algorithm can use float for an
inner correction solve and double for residual evaluation and solution
updates. Its convergence criteria, conversions, and fallback behavior
must be implemented and validated.

The options above do not enable FP16, BF16, or automatic Tensor Core
execution.

Keep aggressive floating-point transformations such as `-ffast-math`
out of the initial numerical reference configuration.

### 7.8 MPI and test flags

| Flag | Explanation |
|---|---|
| `Tpetra_ASSUME_GPU_AWARE_MPI=OFF` | Allows Tpetra to stage communication through host memory instead of assuming MPI accepts device buffers |
| `Trilinos_ENABLE_TESTS=OFF` | Keeps tests disabled globally unless a package is explicitly enabled for testing |
| `Trilinos_ENABLE_EXAMPLES=OFF` | Avoids building examples globally |
| `Tpetra_ENABLE_TESTS=ON` | Enables Tpetra tests explicitly |
| `Belos_ENABLE_TESTS=ON` | Enables Belos tests explicitly |
| `Trilinos_TEST_CATEGORIES=BASIC` | Selects the BASIC test category |
| `MPI_EXEC_MAX_NUMPROCS=1` | Limits MPI test configurations to one rank |

`Tpetra_ASSUME_GPU_AWARE_MPI=OFF` does not disable GPU computation.

Enable GPU-aware communication only after validating that the actual
MPI installation and transport support the intended device buffers.

This setting controls Tpetra communication. KRONA's own halo exchange
needs an independent host-staging or device-buffer communication policy.

The test selection covers the explicitly enabled package tests, not the
entire Trilinos test suite.

### 7.9 Switching from CUDA to HIP

For an AMD GPU configuration, replace CUDA with HIP while retaining
Serial and OpenMP.

Install a compatible ROCm toolchain and driver first. Kokkos 5.x lists
ROCm 6.2.0 as its minimum supported version; also check support for the
actual GPU and operating system.

Apply the following changes to the configure command:

| CUDA configuration | HIP replacement |
|---|---|
| `OMPI_CXX=.../nvcc_wrapper` | `OMPI_CXX=/opt/rocm/bin/hipcc` |
| `TPL_ENABLE_CUDA=ON` | `TPL_ENABLE_CUDA=OFF` |
| `Kokkos_ENABLE_CUDA=ON` | `Kokkos_ENABLE_CUDA=OFF` |
| `Kokkos_ENABLE_HIP=OFF` | `Kokkos_ENABLE_HIP=ON` |
| `Kokkos_ARCH_AMPERE86=ON` | Remove and select the actual AMD GPU architecture |
| `Tpetra_INST_CUDA=ON` | `Tpetra_INST_CUDA=OFF` |
| `Tpetra_INST_HIP=OFF` | `Tpetra_INST_HIP=ON` |

`/opt/rocm/bin/hipcc` is an example installation path. Replace it with
the actual compiler path.

For example, an AMD MI250 configuration uses:

```bash
-DKokkos_ARCH_AMD_GFX90A=ON
```

This architecture is an example for the MI200 family, not a generic
setting for every AMD GPU.

Keep the following settings enabled:

```bash
-DTrilinos_ENABLE_OpenMP=ON \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_OPENMP=ON \
-DTpetra_INST_SERIAL=ON \
-DTpetra_INST_OPENMP=ON
```

Keep `Tpetra_ASSUME_GPU_AWARE_MPI=OFF` for the initial HIP setup as well.

The ROCm compiler must provide working OpenMP support and be compatible
with the selected MPI and C++ dependencies. The HIP configuration is
not purely a backend-flag change: it changes the underlying compiler.

Use the HIP compiler override during both configuration and compilation:

```bash
OMPI_CXX=/opt/rocm/bin/hipcc \
    cmake --build src/trilinos-17.2.1/build-hip --parallel 2
```

Configure `build-hip` first using the modified configuration command.

Use a fresh build directory when changing from CUDA to HIP.
Do not reuse the CUDA CMake cache.

If retaining both installations, change:

- the build directory to `src/trilinos-17.2.1/build-hip`;
- the installation prefix to `$PWD/install/trilinos-17.2.1-hip`;
- the test and install commands to reference `build-hip`.

Both builds can use the same downloaded Trilinos source tree.

## 8. What Backend Portability Means for KRONA

Installing Kokkos does not automatically move ordinary C++ loops onto
a GPU.

Portable computational code must use the Kokkos programming model:

- Kokkos execution policies and parallel dispatch;
- views allocated in suitable memory spaces;
- device-callable kernel code;
- explicit data movement where required;
- synchronization before data is consumed by another execution context.

Mesh loading, configuration parsing, and most filesystem operations can
remain host-side activities.

Switching between Serial, OpenMP, and CUDA implementations already
included in this installation does not require rebuilding Trilinos.
KRONA must select the appropriate execution-space and Tpetra Node
types, either during compilation or through explicitly implemented
runtime dispatch.

Introducing another device backend, such as HIP, requires a new
dependency build with the corresponding compiler and architecture.

Keep backend-specific choices behind narrow interfaces. Data layout,
work decomposition, temporary storage, and communication still require
validation and may need tuning for each target.

## 9. toml++ for Solver Configuration

The existing solver layout uses a vendored toml++ header at:

```text
external/tomlplusplus/include/toml++/toml.hpp
```

The intended pinned version is 3.4.0.

toml++ is a header-only TOML parser requiring C++17 or newer, so it is
compatible with the C++20 toolchain described here.

No separate compilation or installation is required.

### 9.1 Verify the vendored header

Run this section from the KRONA repository root, not from `deps`:

```bash
test -f external/tomlplusplus/include/toml++/toml.hpp
```

The previous solver documentation names `cfd_solver` as the consuming
target. Confirm the actual target name during integration into KRONA.

If TOML parsing is confined to implementation files, its include path
can remain private to that target. If public headers expose toml++
types, the dependency must also be available to consumers.

### 9.2 Download or replace the header

Run from the repository root:

```bash
(
    set -e

    mkdir -p external/tomlplusplus/include/toml++

    curl --fail --location --retry 3 \
        --output external/tomlplusplus/include/toml++/toml.hpp.tmp \
        https://raw.githubusercontent.com/marzer/tomlplusplus/v3.4.0/toml.hpp

    mv external/tomlplusplus/include/toml++/toml.hpp.tmp \
        external/tomlplusplus/include/toml++/toml.hpp
)
```

The temporary download avoids replacing an existing header with a
partial download if the transfer fails.

Keep the upstream license and attribution with the vendored dependency.

## 10. Connecting the Dependencies to KRONA

Keep KRONA's top-level `CMakeLists.txt` in the repository root.
A separate `cmake` directory is not required by this dependency layout.

Use separate KRONA build directories for CPU and CUDA configurations.

Each configuration must discover exactly one intended Trilinos
installation and the Kokkos installation that belongs to it.

For example, a CPU configuration can receive these installation
prefixes through `CMAKE_PREFIX_PATH`:

```text
/path/to/KRONA/deps/install/cgns-4.5.2-i32;
/path/to/KRONA/deps/install/kaminpar-3.7.3;
/path/to/KRONA/deps/install/trilinos-17.2.1
```

CPU and CUDA KRONA build configurations can use the same unified
Trilinos installation. Separate KRONA build directories are useful
when selecting the execution backend at compile time.

A HIP configuration must discover its corresponding HIP-enabled
Trilinos installation.

Prefer dependency-provided CMake targets and package configuration files
over manually listing headers and libraries. Static libraries still
have transitive dependencies that must reach the final link command.

The final KRONA configure commands depend on its root CMakeLists.txt
and should be added when dependency integration is implemented.

## 11. Validation Before Adopting the Environment

Treat the environment as validated only after checking:

1. Parallel HDF5 and parallel CGNS are actually selected.
2. The partitioner builds against the selected KaMinPar and CGNS APIs.
3. A representative mesh can be partitioned with multiple MPI ranks.
4. The resulting HDF5 file has the expected topology, geometry,
   partition offsets, and communication metadata.
5. The enabled Trilinos tests pass.
6. KRONA can use the intended float and double Tpetra instantiations.
7. The CUDA build runs on the intended GPU when that configuration is
   introduced.

Dependency tests do not replace KRONA's numerical verification or
partitioner integration tests.

Keep the successful configure commands, source revisions, compiler
versions, and test results with the project's reproducibility records.

## 12. Moving to an HPC Cluster

Use the cluster's supported compiler, MPI, CUDA/HIP, and numerical
library modules when available.

A cluster does not inherently require rebuilding HDF5 from source.
Use a compatible parallel HDF5 module if the site provides one.

Rebuild dependencies when changing an incompatible compiler/MPI stack
or GPU architecture. Do not assume that a workstation installation can
be copied directly to a cluster.

Configure the hardware-specific options for the target compute nodes,
which may differ from the login node.

Run MPI and GPU tests through the site's scheduler and follow its
rank-binding and GPU-allocation conventions.

## References

- Trilinos releases:
  https://github.com/trilinos/Trilinos/releases
- Trilinos build guidance:
  https://trilinos.github.io/download.html
- Kokkos requirements:
  https://kokkos.org/kokkos-core-wiki/get-started/requirements.html
- Kokkos documentation:
  https://kokkos.org/kokkos-core-wiki/
- CGNS source and build documentation:
  https://github.com/CGNS/CGNS
- CGNS tools:
  https://cgns.org/cgns-archives/CGNS_docs_current/cgnstools/tools.html
- CMake FindHDF5:
  https://cmake.org/cmake/help/latest/module/FindHDF5.html
- KaMinPar:
  https://github.com/KaHIP/KaMinPar
- toml++:
  https://github.com/marzer/tomlplusplus
- NVIDIA CUDA installation guidance:
  https://docs.nvidia.com/cuda/cuda-installation-guide-linux/