#==================================================================
#==== REQUIRED CMAKE VERSION ======================================
#==================================================================
cmake_minimum_required(
    VERSION 3.28
)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build/<configuration>")
endif()

#==================================================================
#==== PROJECT NAME AND LANGS ======================================
#==================================================================
project(
    KRONA
    VERSION 0.1.0
    DESCRIPTION "Portable distributed scientific computing" 
    LANGUAGES C CXX
)

#==================================================================
#==== CXX MINIMUM VERSION REQUIRED ================================
#==================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#==================================================================
#==== DEFAULT COMPILATION MODE (RelWithDebInfo) =================== 
#==================================================================
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    get_property(_krona_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(NOT _krona_multi_config AND NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build configuration" FORCE)
    endif()
endif()

#==================================================================
#==== SYSTEM MODULES & TESTING CAPABILITIES =======================
#==================================================================
include(CTest)
include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

#==================================================================
#==== CORE ARCHITECTURE & DATA TYPE CONFIGURATION =================
#==================================================================
function(krona_cache_choice name default description)
    set(${name} "${default}" CACHE STRING "${description}")
    set_property(CACHE ${name} PROPERTY STRINGS ${ARGN})
    if(NOT "${${name}}" IN_LIST ARGN)
        message(FATAL_ERROR "${name}='${${name}}' is invalid. Supported values: ${ARGN}")
    endif()
endfunction()

krona_cache_choice(KRONA_EXECUTION_SPACE SERIAL "KRONA execution space" SERIAL OPENMP CUDA HIP)
krona_cache_choice(KRONA_GLOBAL_INDEX_BITS 64 "Signed global mesh index width" 32 64)
krona_cache_choice(KRONA_LOCAL_INDEX_BITS 32 "Signed local mesh index and CSR offset width" 32 64)
krona_cache_choice(KRONA_PRECISION_PROFILE FP64 "Numerical precision profile" FP32 FP64 MIXED)

#==================================================================
#==== THIRD-PARTY COMPONENT & OPTIONAL FEATURE SWITCHES ===========
#==================================================================
option(KRONA_ENABLE_CGNS "Discover and validate parallel CGNS support" ON)
option(KRONA_ENABLE_PARTITIONING "Discover distributed KaMinPar and TBB" ON)
option(KRONA_ENABLE_TRILINOS "Discover and validate the Trilinos linear algebra stack" ON)
option(KRONA_WARNINGS_AS_ERRORS "Treat KRONA compiler warnings as errors" OFF)

#==================================================================
#==== HIGH-PERFORMANCE MIXED PRECISION & LAYER TARGETS ============
#==================================================================
set(_krona_real_roles STORAGE COMPUTE ACCUMULATION GEOMETRY SOLVER PRECONDITIONER)
if(KRONA_PRECISION_PROFILE STREQUAL "MIXED")
    foreach(_role IN LISTS _krona_real_roles)
        if(_role STREQUAL "STORAGE" OR _role STREQUAL "PRECONDITIONER")
            set(_default FP32)
        else()
            set(_default FP64)
        endif()
        krona_cache_choice(KRONA_MIXED_${_role}_PRECISION "${_default}"
            "MIXED profile: ${_role} precision (inactive in uniform profiles)" FP32 FP64)
        set(_krona_${_role}_precision "${KRONA_MIXED_${_role}_PRECISION}")
    endforeach()
else()
    foreach(_role IN LISTS _krona_real_roles)
        set(_krona_${_role}_precision "${KRONA_PRECISION_PROFILE}")
    endforeach()
endif()

#==================================================================
#==== DYNAMIC TEMPLATE INSTANTIATION & TYPE ANALYSIS ==============
#==================================================================
set(_krona_real_types "")
foreach(_role IN LISTS _krona_real_roles)
    if(_krona_${_role}_precision STREQUAL "FP32")
        set(_krona_${_role}_type float)
    else()
        set(_krona_${_role}_type double)
    endif()
    list(APPEND _krona_real_types "${_krona_${_role}_type}")
endforeach()
list(REMOVE_DUPLICATES _krona_real_types)
list(LENGTH _krona_real_types _krona_real_type_count)
set(_krona_multiple_real_types false)
if(_krona_real_type_count GREATER 1)
    set(_krona_multiple_real_types true)
endif()

#==================================================================
#==== CONFIGURING NATIVE C++ INDEX TYPES & KOKKOS BACKENDS ========
#==================================================================
set(_krona_local_index_type "std::int${KRONA_LOCAL_INDEX_BITS}_t")
set(_krona_global_index_type "std::int${KRONA_GLOBAL_INDEX_BITS}_t")
set(_krona_execution_type "Kokkos::Serial")
if(KRONA_EXECUTION_SPACE STREQUAL "OPENMP")
    set(_krona_execution_type "Kokkos::OpenMP")
elseif(KRONA_EXECUTION_SPACE STREQUAL "CUDA")
    set(_krona_execution_type "Kokkos::Cuda")
elseif(KRONA_EXECUTION_SPACE STREQUAL "HIP")
    set(_krona_execution_type "Kokkos::HIP")
endif()

#==================================================================
#==== PREPARE BUILD TREE FOR GENERATED CONFIGURATION HEADERS ======
#==================================================================
set(_krona_generated_include "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY "${_krona_generated_include}/krona")
file(CONFIGURE OUTPUT "${_krona_generated_include}/krona/config.hpp" CONTENT [=[
#pragma once
// Generated by KRONA's root CMakeLists.txt. Do not edit.
#include <climits>
#include <cstdint>
#include <limits>
#include <string_view>

#define KRONA_EXECUTION_SPACE_@KRONA_EXECUTION_SPACE@ 1

namespace krona {
using LocalIndex = @_krona_local_index_type@;
using GlobalIndex = @_krona_global_index_type@;
using StorageReal = @_krona_STORAGE_type@;
using ComputeReal = @_krona_COMPUTE_type@;
using AccumulationReal = @_krona_ACCUMULATION_type@;
using GeometryReal = @_krona_GEOMETRY_type@;
using SolverReal = @_krona_SOLVER_type@;
using PreconditionerReal = @_krona_PRECONDITIONER_type@;

namespace build {
inline constexpr std::string_view version = "@PROJECT_VERSION@";
inline constexpr std::string_view execution_space = "@KRONA_EXECUTION_SPACE@";
inline constexpr std::string_view precision_profile = "@KRONA_PRECISION_PROFILE@";
inline constexpr unsigned local_index_bits = @KRONA_LOCAL_INDEX_BITS@;
inline constexpr unsigned global_index_bits = @KRONA_GLOBAL_INDEX_BITS@;
inline constexpr bool uses_multiple_real_types = @_krona_multiple_real_types@;
} // namespace build

static_assert(CHAR_BIT == 8);
static_assert(sizeof(LocalIndex) * CHAR_BIT == build::local_index_bits);
static_assert(sizeof(GlobalIndex) * CHAR_BIT == build::global_index_bits);
static_assert(std::numeric_limits<LocalIndex>::is_signed);
static_assert(std::numeric_limits<GlobalIndex>::is_signed);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
} // namespace krona
]=] @ONLY NEWLINE_STYLE UNIX)

#==================================================================
#==== GLOBAL BUILD CONFIGURATION INTERFACE & EXPORT CONTRACT ======
#==================================================================
add_library(krona_build_config INTERFACE)
add_library(krona::build_config ALIAS krona_build_config)
target_compile_features(krona_build_config INTERFACE cxx_std_20)
target_include_directories(krona_build_config INTERFACE
    "$<BUILD_INTERFACE:${_krona_generated_include}>")

#==================================================================
#==== COMPILER WARNINGS & CODE QUALITY CONTROLS ===================
#==================================================================
add_library(krona_warnings INTERFACE)
add_library(krona::warnings ALIAS krona_warnings)
target_compile_options(krona_warnings INTERFACE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>"
    "$<$<COMPILE_LANG_AND_ID:CXX,NVIDIA>:-Xcompiler=-Wall,-Wextra,-Wpedantic,-Wconversion,-Wshadow>"
    "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/permissive->")
if(KRONA_WARNINGS_AS_ERRORS)
    target_compile_options(krona_warnings INTERFACE
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Werror>"
        "$<$<COMPILE_LANG_AND_ID:CXX,NVIDIA>:-Werror=all-warnings;-Xcompiler=-Werror>"
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX>")
endif()

#==================================================================
#==== ACTIVE DEPENDENCY VERIFICATION & SANITY CHECKS ==============
#==================================================================
function(krona_check_cxx label source)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES ${ARGN})
    set(CMAKE_REQUIRED_INCLUDES "${_krona_generated_include}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
    unset(_krona_cxx_check CACHE)
    check_cxx_source_compiles("${source}" _krona_cxx_check)
    if(NOT _krona_cxx_check)
        message(FATAL_ERROR "${label}. See CMakeFiles/CMakeConfigureLog.yaml for compiler/linker output.")
    endif()
    cmake_pop_check_state()
endfunction()

function(krona_require_target name)
    if(NOT TARGET "${name}")
        message(FATAL_ERROR "Required dependency target '${name}' was not exported by the selected installation.")
    endif()
endfunction()

#==================================================================
#==== DISTRIBUTED COMPUTING: MPI DETECT & CONFIGURATION ===========
#==================================================================
set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI 3.1 REQUIRED COMPONENTS C CXX)

add_library(
    krona_mpi 
    STATIC
        src/mpi/error.cpp
        src/mpi/runtime.cpp
        src/mpi/communicator.cpp
        src/mpi/collective_status.cpp
)
add_library(krona::mpi ALIAS krona_mpi)
target_include_directories(
    krona_mpi 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
    krona_mpi
    PUBLIC
        MPI::MPI_CXX
        krona::build_config
    PRIVATE
        krona::warnings
)

#==================================================================
#==== LINEAR ALGEBRA & CODESIGN PARALLELISM: TRILINOS / KOKKOS ====
#==================================================================
if(KRONA_ENABLE_TRILINOS)
    find_package(Trilinos 17.2.1 CONFIG REQUIRED
        COMPONENTS Tpetra Belos Ifpack2 MueLu)
    krona_require_target(Trilinos::all_selected_libs)
    krona_require_target(Kokkos::kokkos)
else()
    find_package(Kokkos 5.2 CONFIG REQUIRED)
    krona_require_target(Kokkos::kokkos)
endif()

add_library(krona_execution INTERFACE)
add_library(krona::execution ALIAS krona_execution)
target_link_libraries(krona_execution INTERFACE krona::build_config Kokkos::kokkos)

#==================================================================
#==== COMPILE-TIME SANITY CHECK: KOKKOS BACKEND VERIFICATION ======
#==================================================================
set(_krona_kokkos_probe "
#include <Kokkos_Core.hpp>
#include <krona/config.hpp>
#ifndef KOKKOS_ENABLE_${KRONA_EXECUTION_SPACE}
#error Requested KRONA execution space is not enabled in the installed Kokkos
#endif
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        using Exec = ${_krona_execution_type};
        Kokkos::View<krona::StorageReal*, typename Exec::memory_space> values(\"probe\", 1);
        Kokkos::parallel_for(\"probe\", Kokkos::RangePolicy<Exec>(0, 1),
            KOKKOS_LAMBDA(const int i) { values(i) = krona::StorageReal{1}; });
        Exec{}.fence();
    }
    Kokkos::finalize();
    return 0;
}")
krona_check_cxx(
    "The selected Kokkos backend cannot compile/link with this compiler. Check backend availability and use the compatible nvcc_wrapper/hipcc toolchain for a GPU-enabled installation"
    "${_krona_kokkos_probe}" Kokkos::kokkos)

#==================================================================
#==== PARALLEL HDF5 DETECTION & LINKAGE VERIFICATION ==============
#==================================================================
set(HDF5_PREFER_PARALLEL TRUE)
set(HDF5_USE_STATIC_LIBRARIES TRUE)
set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE TRUE)
find_package(HDF5 1.10 REQUIRED MODULE COMPONENTS C)
if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "Parallel HDF5 is required. Select its prefix/wrapper and configure a fresh build directory.")
endif()
krona_require_target(HDF5::HDF5)

cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES HDF5::HDF5 MPI::MPI_C)
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
unset(_krona_parallel_hdf5_check CACHE)
check_c_source_compiles([[
#include <mpi.h>
#include <hdf5.h>
#ifndef H5_HAVE_PARALLEL
#error HDF5 headers do not provide parallel I/O
#endif
int main(void) {
    return H5Pset_fapl_mpio(H5P_DEFAULT, MPI_COMM_WORLD, MPI_INFO_NULL) < 0;
}
]] _krona_parallel_hdf5_check)
if(NOT _krona_parallel_hdf5_check)
    message(FATAL_ERROR "Parallel HDF5 headers/libraries cannot link with the selected MPI. See CMakeConfigureLog.yaml.")
endif()
cmake_pop_check_state()

add_library(krona_hdf5 INTERFACE)
add_library(krona::hdf5 ALIAS krona_hdf5)
target_link_libraries(krona_hdf5 INTERFACE HDF5::HDF5 MPI::MPI_C)

#==================================================================
#==== CFD MESH I/O: PARALLEL CGNS SEARCH & VALIDATION =============
#==================================================================
if(KRONA_ENABLE_CGNS)
    # CGNS exports can refer to HDF5's native target names even when the
    # distribution's HDF5 only provides CMake FindHDF5 targets. Supply those
    # missing names for the selected STATIC HDF5; preserve CGNS link metadata.
    foreach(_hdf5_name hdf5::hdf5-static hdf5-static)
        if(NOT TARGET "${_hdf5_name}")
            add_library("${_hdf5_name}" INTERFACE IMPORTED)
            set_property(TARGET "${_hdf5_name}" PROPERTY INTERFACE_LINK_LIBRARIES HDF5::HDF5)
        endif()
    endforeach()
    find_package(cgns 4.5.2 CONFIG QUIET)
    if(NOT cgns_FOUND)
        # Some CGNS distributions install an export and a version file without
        # a usable package config. Load that upstream export, not guessed libs.
        find_path(KRONA_CGNS_CMAKE_DIR NAMES cgns-targets.cmake
            HINTS "${cgns_ROOT}" "${CGNS_ROOT}"
            PATH_SUFFIXES lib/cmake/cgns lib64/cmake/cgns
                "lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake/cgns"
            REQUIRED)
        if(NOT EXISTS "${KRONA_CGNS_CMAKE_DIR}/cgns-config-version.cmake")
            message(FATAL_ERROR "The selected CGNS export has no version metadata. Use CGNS 4.5.2 or newer.")
        endif()
        include("${KRONA_CGNS_CMAKE_DIR}/cgns-config-version.cmake")
        if(NOT DEFINED PACKAGE_VERSION OR PACKAGE_VERSION VERSION_LESS 4.5)
            message(FATAL_ERROR "CGNS 4.5.2 or newer is required; export reports '${PACKAGE_VERSION}'.")
        endif()
        include("${KRONA_CGNS_CMAKE_DIR}/cgns-targets.cmake")
    endif()
    krona_require_target(CGNS::cgns_static)
    krona_check_cxx("The selected CGNS cannot link its parallel API with the selected MPI/HDF5"
        "#include <pcgnslib.h>\nint main() { int file = 0; return cgp_open(\"configure-probe.cgns\", CG_MODE_READ, &file); }"
        CGNS::cgns_static HDF5::HDF5 MPI::MPI_CXX)
    add_library(krona_cgns INTERFACE)
    add_library(krona::cgns ALIAS krona_cgns)
    target_link_libraries(krona_cgns INTERFACE CGNS::cgns_static krona::hdf5)
endif()

#==================================================================
#==== GRAPH PARTITIONING & MESH BALANCING: KAMINPAR & TBB =========
#==================================================================
if(KRONA_ENABLE_PARTITIONING)
    find_package(TBB CONFIG REQUIRED COMPONENTS tbb)
    find_package(KaMinPar 3.7.3 CONFIG REQUIRED)
    krona_require_target(KaMinPar::dKaMinPar)
    krona_require_target(TBB::tbb)
    add_library(krona_partitioning INTERFACE)
    add_library(krona::partitioning ALIAS krona_partitioning)
    target_link_libraries(krona_partitioning INTERFACE
        KaMinPar::dKaMinPar TBB::tbb MPI::MPI_CXX)
endif()

#==================================================================
#==== DISTRIBUTED ALGEBRA: TPETRA EXPLICIT INSTANTIATION CHECK ====
#==================================================================
if(KRONA_ENABLE_TRILINOS)
    # The adapter intentionally uses the installed Tpetra int/long long pair,
    # independently of mesh index width. Narrowing must be checked by the
    # future adapter; changing LocalIndex does not rebuild Tpetra's ABI.
    set(_krona_tpetra_node "Tpetra::KokkosCompat::Kokkos${KRONA_EXECUTION_SPACE}WrapperNode")
    if(KRONA_EXECUTION_SPACE STREQUAL "SERIAL")
        set(_krona_tpetra_node "Tpetra::KokkosCompat::KokkosSerialWrapperNode")
    elseif(KRONA_EXECUTION_SPACE STREQUAL "OPENMP")
        set(_krona_tpetra_node "Tpetra::KokkosCompat::KokkosOpenMPWrapperNode")
    elseif(KRONA_EXECUTION_SPACE STREQUAL "CUDA")
        set(_krona_tpetra_node "Tpetra::KokkosCompat::KokkosCudaWrapperNode")
    endif()
    set(_krona_linear_types "${_krona_SOLVER_type}" "${_krona_PRECONDITIONER_type}")
    list(REMOVE_DUPLICATES _krona_linear_types)
    foreach(_scalar IN LISTS _krona_linear_types)
        set(_krona_tpetra_probe "
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <TpetraCore_config.h>
#ifndef HAVE_TPETRA_INST_${KRONA_EXECUTION_SPACE}
#error The selected Tpetra Node was not instantiated
#endif
#ifndef HAVE_TPETRA_INST_INT_LONG_LONG
#error Tpetra int/long long ordinals are required by the KRONA adapter
#endif
int main(int argc, char** argv) {
    Tpetra::ScopeGuard guard(&argc, &argv);
    using Node = ${_krona_tpetra_node};
    using Map = Tpetra::Map<int, long long, Node>;
    auto map = Teuchos::rcp(new Map(Tpetra::global_size_t{1}, 0LL, Tpetra::getDefaultComm()));
    Tpetra::CrsMatrix<${_scalar}, int, long long, Node> matrix(map, std::size_t{1});
    return 0;
}")
        krona_check_cxx("Tpetra cannot compile/link ${_scalar}, int, long long, ${KRONA_EXECUTION_SPACE}; check its ETI options and compiler/MPI compatibility"
            "${_krona_tpetra_probe}" Trilinos::all_selected_libs MPI::MPI_CXX)
    endforeach()
    add_library(krona_linear_algebra INTERFACE)
    add_library(krona::linear_algebra ALIAS krona_linear_algebra)
    target_link_libraries(krona_linear_algebra INTERFACE
        krona::build_config Trilinos::all_selected_libs MPI::MPI_CXX)
endif()

#==================================================================
#==== CONFIGURATION REPORT & APPLICATION ENTRYS ===================
#==================================================================
message(STATUS "")
message(STATUS "KRONA ${PROJECT_VERSION}: configuration foundation (no application targets yet)")
message(STATUS "  Backend:           ${KRONA_EXECUTION_SPACE}")
message(STATUS "  Mesh indices:      local=${KRONA_LOCAL_INDEX_BITS}, global=${KRONA_GLOBAL_INDEX_BITS}")
message(STATUS "  Precision profile: ${KRONA_PRECISION_PROFILE}")
foreach(_role IN LISTS _krona_real_roles)
    message(STATUS "  ${_role}: ${_krona_${_role}_type}")
endforeach()
message(STATUS "  Parallel HDF5:     ${HDF5_VERSION} (static baseline)")
message(STATUS "  CGNS / partitioning / Trilinos: ${KRONA_ENABLE_CGNS} / ${KRONA_ENABLE_PARTITIONING} / ${KRONA_ENABLE_TRILINOS}")
message(STATUS "  BUILD_TESTING:     ${BUILD_TESTING}")
message(STATUS "  Generated types:   ${_krona_generated_include}/krona/config.hpp")








#==================================================================
#==== ADDIITONAL LIBRARIES  ======================================= 
#==================================================================
    #=======================
    #=== MESH GENERATION ===
    #=======================
if(KRONA_ENABLE_CGNS)
    add_library(
        krona_mesh_generation
        STATIC
            src/mesh_generation/config.cpp   
            src/mesh_generation/topology.cpp
            src/mesh_generation/writer.cpp
    )
    add_library(krona::mesh_generation ALIAS krona_mesh_generation)
    target_include_directories(
        krona_mesh_generation
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_include_directories(
        krona_mesh_generation
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/external/tomlplusplus/include
    )
    target_link_libraries(
        krona_mesh_generation
        PUBLIC
            krona::build_config
            krona::mpi
            krona::cgns
        PRIVATE
            krona::warnings
    )
endif()

    #=================
    #=== MESH CORE ===
    #=================
add_library(
    krona_mesh_core 
    STATIC
        src/mesh/extrusion.cpp
        src/mesh/faces.cpp
        src/mesh/localmesh.cpp
        src/mesh/geometry.cpp
        src/mesh/reorder.cpp
        src/mesh/bc_config.cpp
        src/mesh/validate.cpp
        src/mesh/aux_connectivity.cpp
        src/mesh/aux_geometry.cpp
        src/mesh/wall_distance.cpp
)
add_library(krona::mesh_core ALIAS krona_mesh_core)
target_include_directories(
    krona_mesh_core 
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
    krona_mesh_core 
    PUBLIC 
        krona::build_config
        krona::mpi
    PRIVATE
        krona::warnings
)

    #===================
    #=== MESH OUTPUT ===
    #===================
add_library(
    krona_mesh_output 
    STATIC
        src/io/solver_mesh/hdf5_reader.cpp
        src/io/solver_mesh/hdf5_writer.cpp
        src/io/vtu/vtu.cpp
)
add_library(krona::mesh_output ALIAS krona_mesh_output)
target_include_directories(
    krona_mesh_output 
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
    krona_mesh_output 
    PUBLIC 
        krona::build_config
        krona::mpi
    PRIVATE
        krona::warnings
        krona::hdf5
)

    #===============
    #=== CGNS IO ===
    #===============
if(KRONA_ENABLE_CGNS)
    add_library(
        krona_cgns_io
        STATIC
            src/io/cgns/reader.cpp
    )
    add_library(krona::cgns_io ALIAS krona_cgns_io)
    target_include_directories(
        krona_cgns_io 
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(
        krona_cgns_io
        PUBLIC 
            krona::mesh_core 
            krona::mpi
        PRIVATE 
            krona::cgns 
            krona::warnings
    )
endif()

    #======================
    #=== MESH PARTITION ===
    #======================
if(KRONA_ENABLE_PARTITIONING)
    add_library(
        krona_mesh_partition
        STATIC
            src/mesh_partition/partition.cpp
    )

    # Use nvcc_wrapper's host compiler for this CPU-only library.
    target_compile_options(
        krona_mesh_partition
        PRIVATE
            --host-only
    )

    add_library(krona::mesh_partition ALIAS krona_mesh_partition)
    target_include_directories(
        krona_mesh_partition
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(
        krona_mesh_partition
        PUBLIC
            krona::mesh_core
            krona::mpi
        PRIVATE
            krona::partitioning
            krona::warnings
    )
endif()

#==================================================================
#==== EXECUTABLES  ================================================
#==================================================================
    #======================
    #=== MESH GENERATOR ===
    #======================
if(KRONA_ENABLE_CGNS)
    add_executable(krona_mesh_generator apps/mesh_generator.cpp)
    target_link_libraries(
        krona_mesh_generator
        PRIVATE
            krona::build_config
            krona::warnings
            krona::mpi

            krona::mesh_generation
    )
endif()

    #========================
    #=== MESH PARTITIONER ===
    #========================
if(KRONA_ENABLE_CGNS)
    add_executable(krona_mesh_partitioner apps/mesh_partitioner.cpp)
    target_link_libraries(
        krona_mesh_partitioner
        PRIVATE
            krona::build_config
            krona::warnings
            krona::mpi
            krona::cgns_io
            krona::mesh_core
            krona::mesh_partition
            krona::mesh_output
    )
endif()


#==================================================================
#==== ADD BUILDING TESTING ========================================
#==================================================================
#if(BUILD_TESTING)
#    add_subdirectory(tests/mesh)
#
#    if(KRONA_ENABLE_CGNS)
#        add_subdirectory(tests/cgns)
#    endif()
#endif()