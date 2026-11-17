add_executable(krona_test_flow test_conservative.cpp)
target_link_libraries(krona_test_flow PRIVATE krona_test_support krona::flow)

add_test(NAME flow.conservative
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
        ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_flow>" ${MPIEXEC_POSTFLAGS})
set_tests_properties(flow.conservative PROPERTIES
    LABELS "flow;unit" TIMEOUT 120
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
