if(NOT KRONA_ENABLE_TRILINOS)
    return()
endif()

add_executable(krona_test_scalar_transient test_transient.cpp)
target_link_libraries(krona_test_scalar_transient PRIVATE krona_test_support krona::transport_scalar)

if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Scalar transient tests require an MPI launcher")
endif()

foreach(case IN ITEMS coefficients additive conservation strided order empty native)
    add_test(NAME "transport.scalar.transient.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1 ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_scalar_transient>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("transport.scalar.transient.${case}" PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
endforeach() 

foreach(ranks IN ITEMS 2 3)
    add_test(NAME "transport.scalar.transient.mpi${ranks}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}" ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_scalar_transient>" ${MPIEXEC_POSTFLAGS})
    set_tests_properties("transport.scalar.transient.mpi${ranks}" PROPERTIES
        TIMEOUT 180 PROCESSORS "${ranks}" RUN_SERIAL TRUE)
endforeach()
 
set(cases bad_mass bad_state bad_entry overflow not_started
    zero_step negative_previous_step coefficient_overflow)
set(diagnostics "cell mass must be finite and positive" "time-layer values must be finite"
    "nonfinite or unrepresentable" "nonfinite or unrepresentable" "call begin_async"
    "Time derivative: time step must be finite and positive"
    "Time derivative: time step must be finite and positive"
    "Time derivative: unrepresentable coefficient")
list(LENGTH cases count)
math(EXPR last "${count}-1")
foreach(i RANGE "${last}")
    list(GET cases "${i}" case)
    list(GET diagnostics "${i}" diagnostic)
    add_test(NAME "transport.scalar.transient.reject_${case}"
        COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:krona_test_scalar_transient>"
            "-DCASE=${case}" "-DEXPECT=${diagnostic}" "-DRANKS=1"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../expect_failure.cmake")
    set_tests_properties("transport.scalar.transient.reject_${case}" PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)
endforeach()
