add_executable(krona_test_viscous test_viscous.cpp)
target_link_libraries(krona_test_viscous PRIVATE krona_test_support krona::execution krona::mpi)
if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Viscous flux tests require an MPI launcher")
endif()
foreach(case IN ITEMS shear rigid_rotation dilation fourier energy covariance nonthermal properties invalid device)
    add_test(NAME "numerics.viscous.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1 ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_viscous>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("numerics.viscous.${case}" PROPERTIES
        TIMEOUT 120 RUN_SERIAL TRUE LABELS "numerics;viscous"
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
