add_executable(krona_test_reconstruction test_reconstruction.cpp)
target_link_libraries(krona_test_reconstruction PRIVATE krona_test_support krona::reconstruction krona::execution krona::halo)
if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Reconstruction tests require an MPI launcher")
endif()
foreach(case IN ITEMS first_order affine tvd darwish bounds venkat coupling ranges quick quadratic convergence mesh misuse)
    add_test(NAME "numerics.reconstruction.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1 ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_reconstruction>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("numerics.reconstruction.${case}" PROPERTIES
        TIMEOUT 120 RUN_SERIAL TRUE LABELS "numerics;reconstruction"
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
add_test(NAME numerics.reconstruction.mpi2
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2 ${MPIEXEC_PREFLAGS}
        "$<TARGET_FILE:krona_test_reconstruction>" ${MPIEXEC_POSTFLAGS} distributed)
set_tests_properties(numerics.reconstruction.mpi2 PROPERTIES
    TIMEOUT 120 RUN_SERIAL TRUE PROCESSORS 2 LABELS "numerics;reconstruction;mpi"
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
