#==================================================================
#==== REQUIRED CMAKE VERSION ======================================
#==================================================================
cmake_minimum_required(VERSION 3.28)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build/<configuration>")
endif()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND NOT DEFINED ENV{CMAKE_EXPORT_COMPILE_COMMANDS})
    if(NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS OR "${CMAKE_EXPORT_COMPILE_COMMANDS}" STREQUAL "")
        set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json" FORCE)
    endif()
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
    get_property(_krona_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(NOT _krona_multi_config AND NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build configuration" FORCE)
    endif()
endif()

#==================================================================
#==== SYSTEM MODULES & TESTING CAPABILITIES =======================
#==================================================================
option(BUILD_TESTING "Build KRONA tests" OFF)
option(KRONA_BUILD_APPLICATIONS "Build KRONA applications" ${PROJECT_IS_TOP_LEVEL})
include(CTest)
include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
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
krona_cache_choice(KRONA_LOCAL_INDEX_BITS 32 "Signed local mesh index width" 32 64)
krona_cache_choice(KRONA_PRECISION_PROFILE FP64 "Numerical precision profile" FP32 FP64 MIXED)

#==================================================================
#==== THIRD-PARTY COMPONENT & OPTIONAL FEATURE SWITCHES ===========
#==================================================================
option(KRONA_ENABLE_CGNS "Build parallel CGNS I/O and mesh generation" ON)
option(KRONA_ENABLE_PARTITIONING "Build distributed mesh partitioning with KaMinPar/TBB" ON)
option(KRONA_ENABLE_TRILINOS "Build KRONA linear algebra with Trilinos" ON)
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
set(_krona_source_include "${CMAKE_CURRENT_SOURCE_DIR}/include")
file(SHA256 "${_krona_generated_include}/krona/config.hpp" _krona_config_hash)

add_library(krona_build_config INTERFACE)
add_library(krona::build_config ALIAS krona_build_config)
target_compile_features(krona_build_config INTERFACE cxx_std_20)
target_include_directories(krona_build_config INTERFACE
    "$<BUILD_INTERFACE:${_krona_generated_include}>")
    
#==================================================================
#==== USFUL HELPERS ===============================================
#==================================================================
function(krona_public_headers target scope)
    target_include_directories(${target} ${scope}
        "$<BUILD_INTERFACE:${_krona_source_include}>"
        "$<INSTALL_INTERFACE:include>")
endfunction()

function(krona_interface name)
    add_library(krona_${name} INTERFACE)
    add_library(krona::${name} ALIAS krona_${name})
    krona_public_headers(krona_${name} INTERFACE)
    target_link_libraries(krona_${name} INTERFACE ${ARGN})
endfunction()

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

function(krona_require_target name)
    if(NOT TARGET "${name}")
        message(FATAL_ERROR
            "Required dependency target '${name}' was not exported by the selected installation.")
    endif()
endfunction()

#==================================================================
#==== ACTIVE DEPENDENCY VERIFICATION & SANITY CHECKS ==============
#==================================================================
function(krona_prepare_probe id language source)
    set(_inputs "${source};${_krona_config_hash};${CMAKE_PREFIX_PATH};${CMAKE_TOOLCHAIN_FILE}")
    string(APPEND _inputs
        ";${CMAKE_${language}_COMPILER};${CMAKE_${language}_COMPILER_ID};${CMAKE_${language}_COMPILER_VERSION}"
        ";${CMAKE_${language}_COMPILER_ARG1};${CMAKE_${language}_COMPILER_LAUNCHER}"
        ";${CMAKE_${language}_FLAGS};${CMAKE_${language}_FLAGS_DEBUG};${CMAKE_${language}_FLAGS_RELEASE}"
        ";${CMAKE_${language}_FLAGS_RELWITHDEBINFO};${CMAKE_${language}_FLAGS_MINSIZEREL}"
        ";${CMAKE_BUILD_TYPE};${CMAKE_TRY_COMPILE_CONFIGURATION};${CMAKE_EXE_LINKER_FLAGS}"
        ";${CMAKE_EXE_LINKER_FLAGS_DEBUG};${CMAKE_EXE_LINKER_FLAGS_RELEASE}"
        ";${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO};${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL}"
        ";${CMAKE_SYSROOT};${CMAKE_SYSROOT_COMPILE};${CMAKE_SYSROOT_LINK}"
        ";${CMAKE_REQUIRED_FLAGS};${CMAKE_REQUIRED_INCLUDES};${CMAKE_REQUIRED_LIBRARIES}"
        ";${MPI_C_COMPILER};${MPI_CXX_COMPILER};${MPI_C_LIBRARIES};${MPI_CXX_LIBRARIES}"
        ";$ENV{OMPI_CXX};$ENV{MPICH_CXX};$ENV{NVCC_WRAPPER_DEFAULT_COMPILER}")
    if(id STREQUAL "KOKKOS" OR id MATCHES "^TPETRA_")
        string(APPEND _inputs ";${Kokkos_DIR};${Trilinos_DIR}")
    elseif(id STREQUAL "HDF5")
        string(APPEND _inputs ";${HDF5_C_LIBRARIES}")
    elseif(id STREQUAL "CGNS")
        string(APPEND _inputs ";${cgns_DIR};${KRONA_CGNS_CMAKE_DIR};${HDF5_C_LIBRARIES}")
    endif()
    foreach(_target IN LISTS CMAKE_REQUIRED_LIBRARIES)
        if(TARGET "${_target}")
            foreach(_property INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS
                    INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_OPTIONS INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION
                    IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_DEBUG)
                get_target_property(_value "${_target}" "${_property}")
                string(APPEND _inputs ";${_target}.${_property}=${_value}")
            endforeach()
        endif()
    endforeach()
    string(SHA256 _signature "${_inputs}")
    if(NOT "${_signature}" STREQUAL "${_KRONA_CHECK_${id}_SIGNATURE}")
        unset(_KRONA_CHECK_${id} CACHE)
    endif()
    set(_KRONA_CHECK_${id}_SIGNATURE "${_signature}" CACHE INTERNAL "KRONA probe inputs" FORCE)
endfunction()

function(krona_check_source id language label source)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES ${ARGN})
    set(CMAKE_REQUIRED_INCLUDES "${_krona_generated_include}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
    krona_prepare_probe("${id}" "${language}" "${source}")
    if(language STREQUAL "C")
        check_c_source_compiles("${source}" _KRONA_CHECK_${id})
    else()
        check_cxx_source_compiles("${source}" _KRONA_CHECK_${id})
    endif()
    if(NOT _KRONA_CHECK_${id})
        message(FATAL_ERROR
            "${label}. See ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CMakeConfigureLog.yaml.")
    endif()
    cmake_pop_check_state()
endfunction()

#==================================================================
#==== DISTRIBUTED COMPUTING: MPI DETECT & CONFIGURATION ===========
#==================================================================
set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI 3.1 REQUIRED COMPONENTS C CXX)
add_library(krona_mpi STATIC
    src/mpi/error.cpp
    src/mpi/runtime.cpp
    src/mpi/communicator.cpp
    src/mpi/collective_status.cpp)
add_library(krona::mpi ALIAS krona_mpi)
krona_public_headers(krona_mpi PUBLIC)
target_link_libraries(krona_mpi
    PUBLIC MPI::MPI_CXX krona::build_config
    PRIVATE krona::warnings)

#==================================================================
#==== LINEAR ALGEBRA & CODESIGN PARALLELISM: TRILINOS / KOKKOS ====
#==================================================================
if(KRONA_ENABLE_TRILINOS)
    find_package(Trilinos 17.2.1 CONFIG REQUIRED COMPONENTS Tpetra Belos Ifpack2 MueLu)
    krona_require_target(Trilinos::all_selected_libs)
else()
    find_package(Kokkos 5.2 CONFIG REQUIRED)
endif()
krona_require_target(Kokkos::kokkos)
krona_interface(execution krona::build_config Kokkos::kokkos)

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
    Kokkos::ScopeGuard guard(argc, argv);
    using Exec = ${_krona_execution_type};
    Kokkos::View<krona::StorageReal*, typename Exec::memory_space> values(\"probe\", 1);
    Kokkos::parallel_for(\"probe\", Kokkos::RangePolicy<Exec>(0, 1),
        KOKKOS_LAMBDA(const int i) { values(i) = krona::StorageReal{1}; });
    Exec{}.fence();
    return 0;
}")
krona_check_source(KOKKOS CXX
    "The selected Kokkos backend cannot compile/link; check backend availability and the nvcc_wrapper/hipcc toolchain for GPU-enabled installations"
    "${_krona_kokkos_probe}" Kokkos::kokkos)

#==================================================================
#==== PARALLEL HDF5 DETECTION & LINKAGE VERIFICATION ==============
#==================================================================
set(HDF5_PREFER_PARALLEL TRUE)
set(HDF5_USE_STATIC_LIBRARIES TRUE)
set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE TRUE)
find_package(HDF5 1.10 REQUIRED MODULE COMPONENTS C)
if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "Parallel HDF5 is required; select its prefix/wrapper in a fresh build directory.")
endif()
krona_require_target(HDF5::HDF5)
krona_check_source(HDF5 C "Parallel HDF5 cannot compile/link with the selected MPI" [[
#include <mpi.h>
#include <hdf5.h>
#ifndef H5_HAVE_PARALLEL
#error HDF5 headers do not provide parallel I/O
#endif
int main(void) {
    return H5Pset_fapl_mpio(H5P_DEFAULT, MPI_COMM_WORLD, MPI_INFO_NULL) < 0;
}
]] HDF5::HDF5 MPI::MPI_C)
krona_interface(hdf5 HDF5::HDF5 MPI::MPI_C)

#==================================================================
#==== CFD MESH I/O: PARALLEL CGNS SEARCH & VALIDATION =============
#==================================================================
if(KRONA_ENABLE_CGNS)
    # Some exports use native HDF5 static target names absent from FindHDF5.
    # Never present a selected shared HDF5 as a static target.
    set(_krona_hdf5_static FALSE)
    foreach(_library IN LISTS HDF5_C_LIBRARIES)
        if(_library MATCHES "(^|[/\\\\])(lib)?hdf5(-static)?\\.a$")
            set(_krona_hdf5_static TRUE)
        endif()
    endforeach()
    foreach(_hdf5_name hdf5::hdf5-static hdf5-static)
        if(NOT TARGET "${_hdf5_name}")
            if(NOT _krona_hdf5_static)
                message(FATAL_ERROR
                    "CGNS needs native HDF5 static target names, but FindHDF5 did not select a static libhdf5 archive. Select the parallel static HDF5 installation.")
            endif()
            add_library("${_hdf5_name}" INTERFACE IMPORTED)
            set_property(TARGET "${_hdf5_name}" PROPERTY INTERFACE_LINK_LIBRARIES HDF5::HDF5)
        endif()
    endforeach()

    find_package(cgns 4.5.2 CONFIG QUIET)
    if(NOT cgns_FOUND)
        find_path(KRONA_CGNS_CMAKE_DIR NAMES cgns-targets.cmake
            HINTS "${cgns_DIR}" "${cgns_ROOT}" "${CGNS_ROOT}"
            PATH_SUFFIXES lib/cmake/cgns lib64/cmake/cgns
                "lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake/cgns"
            REQUIRED)
        # Isolate PACKAGE_* variables used by the upstream version file.
        function(krona_load_cgns_export directory)
            if(NOT EXISTS "${directory}/cgns-config-version.cmake")
                message(FATAL_ERROR "The CGNS export has no version metadata; use CGNS >= 4.5.2.")
            endif()
            set(PACKAGE_FIND_VERSION 4.5.2)
            set(PACKAGE_FIND_VERSION_MAJOR 4)
            set(PACKAGE_FIND_VERSION_MINOR 5)
            set(PACKAGE_FIND_VERSION_PATCH 2)
            set(PACKAGE_FIND_VERSION_TWEAK 0)
            set(PACKAGE_FIND_VERSION_COUNT 3)
            unset(PACKAGE_VERSION)
            unset(PACKAGE_VERSION_COMPATIBLE)
            unset(PACKAGE_VERSION_EXACT)
            unset(PACKAGE_VERSION_UNSUITABLE)
            include("${directory}/cgns-config-version.cmake")
            if(NOT DEFINED PACKAGE_VERSION OR PACKAGE_VERSION VERSION_LESS 4.5
                    OR PACKAGE_VERSION_UNSUITABLE)
                message(FATAL_ERROR "CGNS >= 4.5.2 is required; export reports '${PACKAGE_VERSION}'.")
            endif()
            include("${directory}/cgns-targets.cmake")
            set(cgns_VERSION "${PACKAGE_VERSION}" PARENT_SCOPE)
        endfunction()
        krona_load_cgns_export("${KRONA_CGNS_CMAKE_DIR}")
    endif()
    krona_require_target(CGNS::cgns_static)
    krona_check_source(CGNS CXX "CGNS requires 64-bit cgsize_t and a linkable parallel MPI/HDF5 API" [[
#include <climits>
#include <pcgnslib.h>
static_assert(sizeof(cgsize_t) * CHAR_BIT == 64, "KRONA requires 64-bit CGNS indices");
int main() {
    int file = 0;
    return cgp_open("configure-probe.cgns", CG_MODE_READ, &file);
}
]] CGNS::cgns_static HDF5::HDF5 MPI::MPI_CXX)
    krona_interface(cgns CGNS::cgns_static krona::hdf5)
endif()

#==================================================================
#==== GRAPH PARTITIONING & MESH BALANCING: KAMINPAR & TBB =========
#==================================================================
if(KRONA_ENABLE_PARTITIONING)
    find_package(TBB CONFIG REQUIRED COMPONENTS tbb)
    find_package(KaMinPar 3.7.3 CONFIG REQUIRED)
    krona_require_target(KaMinPar::dKaMinPar)
    krona_require_target(TBB::tbb)
    krona_interface(partitioning KaMinPar::dKaMinPar TBB::tbb MPI::MPI_CXX)

    # --host-only belongs to nvcc_wrapper; plain GCC/Clang must not receive it.
    function(krona_check_host_only)
        cmake_push_check_state(RESET)
        set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
        krona_prepare_probe(HOST_ONLY CXX "--host-only")
        check_cxx_compiler_flag("--host-only" _KRONA_CHECK_HOST_ONLY)
        cmake_pop_check_state()
    endfunction()
    krona_check_host_only()
endif()

#==================================================================
#==== TOML++ (AUX PARSING LIBRARY) ================================
#==================================================================
if(NOT TARGET tomlplusplus::tomlplusplus)
    find_package(tomlplusplus 3 CONFIG QUIET)
endif()
add_library(krona_tomlplusplus INTERFACE)
add_library(krona::tomlplusplus ALIAS krona_tomlplusplus)
if(TARGET tomlplusplus::tomlplusplus)
    target_link_libraries(krona_tomlplusplus INTERFACE tomlplusplus::tomlplusplus)
else()
    find_path(KRONA_TOMLPLUSPLUS_INCLUDE_DIR
        NAMES toml++/toml.hpp toml++/toml.h toml.hpp
        HINTS "${PROJECT_SOURCE_DIR}/external/tomlplusplus/include")
        
    if(NOT KRONA_TOMLPLUSPLUS_INCLUDE_DIR)
        message(FATAL_ERROR "toml++ not found: install it, keep external/tomlplusplus/include, or set KRONA_TOMLPLUSPLUS_INCLUDE_DIR")
    endif()
    target_include_directories(krona_tomlplusplus SYSTEM INTERFACE "${KRONA_TOMLPLUSPLUS_INCLUDE_DIR}")
endif()

#==================================================================
#==== DISTRIBUTED ALGEBRA: TPETRA EXPLICIT INSTANTIATION CHECK ====
#==================================================================
if(KRONA_ENABLE_TRILINOS)
    set(_krona_tpetra_node "Tpetra::KokkosCompat::KokkosDeviceWrapperNode<${_krona_execution_type}>")
    set(_krona_linear_types "${_krona_SOLVER_type}" "${_krona_PRECONDITIONER_type}")
    list(REMOVE_DUPLICATES _krona_linear_types)
    foreach(_scalar IN LISTS _krona_linear_types)
        string(TOUPPER "${_scalar}" _scalar_macro)
        set(_krona_tpetra_probe "
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <TpetraCore_config.h>
#include <BelosTpetraAdapter.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <Ifpack2_Factory.hpp>
#ifndef HAVE_TPETRA_MPI
#error Distributed KRONA algebra requires an MPI-enabled Tpetra
#endif
#ifndef HAVE_TPETRA_INST_${KRONA_EXECUTION_SPACE}
#error The selected Tpetra execution space was not instantiated
#endif
#ifndef HAVE_TPETRA_INST_INT_LONG_LONG
#error KRONA algebra requires int/long long Tpetra ordinals
#endif
#ifndef HAVE_TPETRA_INST_${_scalar_macro}
#error The requested scalar was not instantiated in Tpetra
#endif
int main(int argc, char** argv) {
    Tpetra::ScopeGuard guard(&argc, &argv);
    using Node = ${_krona_tpetra_node};
    using Map = Tpetra::Map<int, long long, Node>;
    auto map = Teuchos::rcp(new Map(Tpetra::global_size_t{1}, 0LL, Tpetra::getDefaultComm()));
    using Matrix = Tpetra::CrsMatrix<${_scalar}, int, long long, Node>;
    using RowMatrix = Tpetra::RowMatrix<${_scalar}, int, long long, Node>;
    using MV = Tpetra::MultiVector<${_scalar}, int, long long, Node>;
    using OP = Tpetra::Operator<${_scalar}, int, long long, Node>;
    auto matrix = Teuchos::rcp(new Matrix(map, std::size_t{1}));
    auto prec = Ifpack2::Factory::create<RowMatrix>(\"RELAXATION\", matrix);
    Belos::PseudoBlockGmresSolMgr<${_scalar}, MV, OP> solver;
    return 0;
}")
        krona_check_source(TPETRA_${_scalar_macro} CXX
            "Trilinos cannot compile/link Tpetra/Belos/Ifpack2 for ${_scalar}, int, long long, ${KRONA_EXECUTION_SPACE}; check ETI, memory space and MPI/compiler compatibility"
            "${_krona_tpetra_probe}" Trilinos::all_selected_libs MPI::MPI_CXX)
    endforeach()
    krona_interface(linear_algebra krona::execution Trilinos::all_selected_libs MPI::MPI_CXX)
endif()








#==================================================================
#==== ADDIITONAL LIBRARIES  ======================================= 
#==================================================================
    #=======================
    #=== MESH GENERATION ===
    #=======================
if(KRONA_ENABLE_CGNS)
    add_library(krona_mesh_generation STATIC
        src/mesh_generation/config.cpp
        src/mesh_generation/topology.cpp
        src/mesh_generation/writer.cpp)
    add_library(krona::mesh_generation ALIAS krona_mesh_generation)
    krona_public_headers(krona_mesh_generation PUBLIC)
    target_link_libraries(krona_mesh_generation
        PUBLIC krona::build_config krona::mpi krona::cgns
        PRIVATE krona::warnings krona::tomlplusplus)
endif()


    #=================
    #=== MESH CORE ===
    #=================
add_library(krona_mesh_core STATIC
    src/mesh/extrusion.cpp
    src/mesh/faces.cpp
    src/mesh/localmesh.cpp
    src/mesh/geometry.cpp
    src/mesh/reorder.cpp
    src/mesh/bc_config.cpp
    src/mesh/validate.cpp
    src/mesh/aux_connectivity.cpp
    src/mesh/aux_geometry.cpp
    src/mesh/wall_distance.cpp)
add_library(krona::mesh_core ALIAS krona_mesh_core)
krona_public_headers(krona_mesh_core PUBLIC)
target_link_libraries(krona_mesh_core
    PUBLIC krona::build_config krona::mpi
    PRIVATE krona::warnings)


    #===============
    #=== MESH IO ===
    #===============
add_library(krona_mesh_io STATIC
    src/io/solver_mesh/hdf5_reader.cpp
    src/io/solver_mesh/hdf5_writer.cpp
    src/io/vtu/vtu.cpp)
add_library(krona::mesh_io ALIAS krona_mesh_io)
krona_public_headers(krona_mesh_io PUBLIC)
target_link_libraries(krona_mesh_io
    PUBLIC krona::build_config krona::mpi krona::mesh_core
    PRIVATE krona::warnings krona::hdf5)


    #======================
    #=== MESH EXECUTION ===
    #======================
krona_interface(mesh_compute krona::execution krona::mpi)


    #==============
    #=== FIELDS ===
    #==============
krona_interface(fields krona::execution krona::mpi)


    #=====================
    #=== HALO EXHANGER ===
    #=====================
krona_interface(halo krona::execution krona::mpi krona::mesh_compute krona::fields)


    #===================================================
    #=== GRADIENT OPERATOR (DIFF OPERATOR IN FUTURE) ===
    #===================================================
krona_interface(gradient krona::mesh_compute krona::fields)


    #=======================
    #=== RECONSTRUCTIONs ===
    #=======================
krona_interface(reconstruction krona::mesh_compute krona::fields)


    #======================
    #=== LINEAR ALGEBRA ===
    #======================
if(KRONA_ENABLE_TRILINOS)
    add_library(krona_linalg STATIC src/linalg/mesh_adapter.cpp)
    add_library(krona::linalg ALIAS krona_linalg)
    krona_public_headers(krona_linalg PUBLIC)
    target_link_libraries(krona_linalg
        PUBLIC krona::execution krona::mpi krona::linear_algebra
        PRIVATE krona::warnings)
endif()

    #========================
    #=== SCALAR TRANSPORT ===
    #========================   
krona_interface(thermodynamics krona::execution krona::mpi)

    #==================================
    #=== SCALAR BOUNDARY CONDITIONS ===
    #==================================
add_library(krona_boundary_scalar STATIC src/boundary/scalar/config.cpp)
add_library(krona::boundary_scalar ALIAS krona_boundary_scalar)
target_compile_features(krona_boundary_scalar PUBLIC cxx_std_20)
krona_public_headers(krona_boundary_scalar PUBLIC)
target_link_libraries(krona_boundary_scalar
    PUBLIC krona::mesh_compute krona::fields krona::mpi
    PRIVATE krona::warnings krona::tomlplusplus)

    
    #====================================
    #=== PHYSICAL BOUNDARY CONDITIONS ===
    #====================================
add_library(krona_boundary_physical STATIC src/boundary/physical/config.cpp)
add_library(krona::boundary_physical ALIAS krona_boundary_physical)
target_compile_features(krona_boundary_physical PUBLIC cxx_std_20)
krona_public_headers(krona_boundary_physical PUBLIC)
target_link_libraries(krona_boundary_physical
    PUBLIC krona::boundary_scalar
    PRIVATE krona::warnings krona::tomlplusplus)


    #========================
    #=== SCALAR TRANSPORT ===
    #========================
krona_interface(transport_scalar krona::mesh_compute krona::fields krona::boundary_scalar krona::linalg krona::halo krona::mpi)


    #===============
    #=== CGNS IO ===
    #===============
if(KRONA_ENABLE_CGNS)
    add_library(krona_cgns_io STATIC src/io/cgns/reader.cpp)
    add_library(krona::cgns_io ALIAS krona_cgns_io)
    krona_public_headers(krona_cgns_io PUBLIC)
    target_link_libraries(krona_cgns_io
        PUBLIC krona::mesh_core krona::mpi
        PRIVATE krona::cgns krona::warnings)
endif()


    #======================
    #=== MESH PARTITION ===
    #======================
if(KRONA_ENABLE_PARTITIONING)
    add_library(krona_mesh_partition STATIC src/mesh_partition/partition.cpp)
    add_library(krona::mesh_partition ALIAS krona_mesh_partition)
    krona_public_headers(krona_mesh_partition PUBLIC)
    if(_KRONA_CHECK_HOST_ONLY)
        target_compile_options(krona_mesh_partition PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:--host-only>")
    endif()
    target_link_libraries(krona_mesh_partition
        PUBLIC krona::mesh_core krona::mpi
        PRIVATE krona::partitioning krona::warnings)
endif()


#==================================================================
#==== EXECUTABLES  ================================================
#==================================================================
if(KRONA_BUILD_APPLICATIONS)
    #======================
    #=== MESH GENERATOR ===
    #======================
    if(KRONA_ENABLE_CGNS)
        add_executable(krona_mesh_generator apps/mesh_generator.cpp)
        target_link_libraries(krona_mesh_generator PRIVATE
            krona::mesh_generation krona::warnings) #krona::build_config krona::mpi
    endif()

    #========================
    #=== MESH PARTITIONER ===
    #========================
    if(KRONA_ENABLE_CGNS AND KRONA_ENABLE_PARTITIONING)
        add_executable(krona_mesh_partitioner apps/mesh_partitioner.cpp)
        target_link_libraries(krona_mesh_partitioner PRIVATE
            krona::cgns_io krona::mesh_partition krona::mesh_io krona::warnings)
    endif()

    #============================
    #=== KOKKOS PLUG (DELETE) ===
    #============================
    add_executable(krona_plug apps/plug.cpp)
    target_link_libraries(krona_plug PRIVATE
            krona::mesh_io krona::mesh_compute krona::warnings krona::fields)

    #================================
    #=== SCALAR TRANSPORT EXAMPLE ===
    #================================
    add_executable(krona_scalar_diffusion apps/scalar_diffusion.cpp)
    target_link_libraries(krona_scalar_diffusion PRIVATE
        krona::transport_scalar krona::mesh_io krona::warnings)

    #============================================
    #=== STEADY CONVECTION DIFFUSION EQUATION ===
    #============================================
    add_executable(krona_scalar_convection_diffusion apps/scalar_convection_diffusion.cpp)
    target_link_libraries(krona_scalar_convection_diffusion PRIVATE
        krona::transport_scalar krona::mesh_io krona::warnings)
    
    #============================================
    #=== TRANSIENT CONVECTION DIFFUSION EQUATION ===
    #============================================
    add_executable(krona_scalar_transient_convection_diffusion apps/scalar_transient_convection_diffusion.cpp)
    target_link_libraries(krona_scalar_transient_convection_diffusion PRIVATE
        krona::transport_scalar krona::mesh_io krona::warnings)

    #==================================================
    #=== 3D TRANSIENT CONVECTION DIFFUSION EQUATION ===
    #==================================================
    add_executable(krona_scalar_transient_gaussian_3d apps/scalar_transient_gaussian_3d.cpp)
    target_link_libraries(krona_scalar_transient_gaussian_3d PRIVATE
        krona::transport_scalar krona::mesh_io krona::warnings)



endif()
    
#==================================================================
#==== ADD BUILDING TESTING ========================================
#==================================================================
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()


#==================================================================
#==== CONFIGURATION REPORT & APPLICATION ENTRYS ===================
#==================================================================
message(STATUS "")
message(STATUS "KRONA ${PROJECT_VERSION}")
message(STATUS "  Backend:           ${KRONA_EXECUTION_SPACE}")
message(STATUS "  Mesh indices:      local=${KRONA_LOCAL_INDEX_BITS}, global=${KRONA_GLOBAL_INDEX_BITS}")
message(STATUS "  Precision profile: ${KRONA_PRECISION_PROFILE}")
foreach(_role IN LISTS _krona_real_roles)
    message(STATUS "  ${_role}: ${_krona_${_role}_type}")
endforeach()
message(STATUS "  Parallel HDF5:     ${HDF5_VERSION} (static linkage preferred)")
message(STATUS "  CGNS / partitioning / Trilinos: ${KRONA_ENABLE_CGNS} / ${KRONA_ENABLE_PARTITIONING} / ${KRONA_ENABLE_TRILINOS}")
message(STATUS "  Applications: ${KRONA_BUILD_APPLICATIONS}")
message(STATUS "  BUILD_TESTING:     ${BUILD_TESTING}")
message(STATUS "  Generated types:   ${_krona_generated_include}/krona/config.hpp")
