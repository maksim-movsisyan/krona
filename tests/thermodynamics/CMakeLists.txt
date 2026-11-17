add_executable(krona_test_acoustic_state test_acoustic_state.cpp)
target_link_libraries(krona_test_acoustic_state PRIVATE krona_test_support krona::thermodynamics)

add_test(NAME thermodynamics.acoustic_state
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
        ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_acoustic_state>" ${MPIEXEC_POSTFLAGS})
set_tests_properties(thermodynamics.acoustic_state PROPERTIES
    LABELS "thermodynamics;unit" TIMEOUT 120
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
