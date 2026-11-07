# Flux invariants, exact discontinuities, device kernels and Sod evolution are module tests.
add_executable(krona_test_riemann test_riemann.cpp)
target_link_libraries(krona_test_riemann PRIVATE
    krona_test_support krona::riemann)

if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Riemann tests require an MPI launcher")
endif()

foreach(case IN ITEMS consistency contacts fallback eos device sod)
    add_test(NAME "numerics.riemann.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_riemann>"
            ${MPIEXEC_POSTFLAGS} "${case}")

    set_tests_properties("numerics.riemann.${case}" PROPERTIES
        TIMEOUT 120
        RUN_SERIAL TRUE
        LABELS "numerics;riemann"
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
 
# mpi::require terminates the process; validate the diagnostic in a child run.
add_test(NAME numerics.riemann.reject_gamma
    COMMAND "${CMAKE_COMMAND}"
        "-DPROGRAM=$<TARGET_FILE:krona_test_riemann>"
        "-DCASE=reject_gamma"
        "-DEXPECT=Roe: gamma must be finite and greater than one"
        "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
        "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_LIST_DIR}/expect_failure.cmake")
set_tests_properties(numerics.riemann.reject_gamma PROPERTIES
    TIMEOUT 120
    RUN_SERIAL TRUE
    LABELS "numerics;riemann;validation"
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")