add_executable(krona_test_boundary_physical test_physical_boundary.cpp)
target_link_libraries(krona_test_boundary_physical PRIVATE krona_test_support krona::boundary_physical krona::boundary_scalar)
target_compile_definitions(krona_test_boundary_physical PRIVATE
    "KRONA_PHYSICAL_BOUNDARY_TEST_DATA=\"${CMAKE_CURRENT_SOURCE_DIR}/data/bc.toml\"")

if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Physical boundary tests require an MPI launcher")
endif()

foreach(case IN ITEMS thermodynamics generic_eos thermal_correction assembly physics config fill layouts dispatch profile statuses incompressible unlabelled empty mixed_storage large_offset_geometry distributed)
    set(ranks 1)
    if(case STREQUAL "distributed")
        set(ranks 2)
    endif()
    add_test(NAME "boundary.physical.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_test_boundary_physical>
                ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("boundary.physical.${case}" PROPERTIES
        LABELS "boundary;physical;numerical" PROCESSORS "${ranks}" TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()

foreach(case IN ITEMS missing wrong_name duplicate invalid_geometry invalid_model extent temperature_without_energy unknown_type parser_duplicate unlabelled typo malformed_toml)
    add_test(NAME "boundary.physical.reject_${case}"
        COMMAND "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:krona_test_boundary_physical>"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
            "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
            "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DCASE=${case}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
    set_tests_properties("boundary.physical.reject_${case}" PROPERTIES
        LABELS "boundary;physical;contract" TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
