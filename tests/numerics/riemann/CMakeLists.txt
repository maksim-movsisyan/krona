add_executable(krona_test_riemann test_riemann.cpp)
target_link_libraries(krona_test_riemann PRIVATE
    krona_test_support krona::thermodynamics)

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