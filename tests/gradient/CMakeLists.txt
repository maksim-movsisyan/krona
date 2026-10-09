add_executable(krona_test_gradient test_gradient.cpp)
target_link_libraries(krona_test_gradient PRIVATE krona_test_support krona::gradient krona::halo)

if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Gradient tests require an MPI launcher")
endif()

foreach(case IN ITEMS constant affine weighted layouts range scale precision degree empty distributed)
    set(ranks 1)
    if(case STREQUAL "distributed")
        set(ranks 2)
    endif()
    add_test(NAME "gradient.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_test_gradient>
                ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("gradient.${case}" PROPERTIES
        LABELS "gradient;numerical"
        PROCESSORS "${ranks}"
        TIMEOUT 120
        RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()

add_test(NAME gradient.singular
    COMMAND "${CMAKE_COMMAND}"
        "-DTEST_EXECUTABLE=$<TARGET_FILE:krona_test_gradient>"
        "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
        "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
set_tests_properties(gradient.singular PROPERTIES
    LABELS "gradient;numerical" TIMEOUT 120 RUN_SERIAL TRUE
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
