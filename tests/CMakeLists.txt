# Included from the KRONA root after all library targets are defined.
add_library(krona_test_support INTERFACE)
target_include_directories(krona_test_support INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(krona_test_support INTERFACE cxx_std_20)
target_link_libraries(krona_test_support INTERFACE krona::execution krona::mpi krona::warnings)

# Host/MPI suites share the same test conventions without a Kokkos dependency.
add_library(krona_test_support_host_only INTERFACE)
target_include_directories(krona_test_support_host_only INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(krona_test_support_host_only INTERFACE cxx_std_20)
target_link_libraries(krona_test_support_host_only INTERFACE krona::mpi krona::warnings)

add_subdirectory(math)
if(KRONA_ENABLE_CGNS)
    add_subdirectory(mesh/generation)
endif()
add_subdirectory(mesh)
