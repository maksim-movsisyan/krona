add_executable(krona_test_boundary_scalar test_boundary_scalar.cpp)
target_link_libraries(krona_test_boundary_scalar PRIVATE krona_test_support krona::boundary_scalar)
target_compile_definitions(krona_test_boundary_scalar PRIVATE
    "KRONA_BOUNDARY_TEST_DATA=\"${CMAKE_CURRENT_SOURCE_DIR}/data/bc.toml\"")

if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Scalar boundary tests require an MPI launcher")
endif()

foreach(case IN ITEMS relations config fill layouts affine assembly empty distributed)
    set(ranks 1)
    if(case STREQUAL "distributed")
        set(ranks 2)
    endif()
    add_test(NAME "boundary.scalar.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_test_boundary_scalar>
                ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("boundary.scalar.${case}" PROPERTIES
        LABELS "boundary;scalar;numerical" PROCESSORS "${ranks}" TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()

foreach(case IN ITEMS missing wrong_name scattered unlabelled zero_diffusivity unknown_type typo duplicate invalid_geometry)
    add_test(NAME "boundary.scalar.reject_${case}"
        COMMAND "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:krona_test_boundary_scalar>"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
            "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
            "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DCASE=${case}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
    set_tests_properties("boundary.scalar.reject_${case}" PROPERTIES
        LABELS "boundary;scalar;contract" TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
