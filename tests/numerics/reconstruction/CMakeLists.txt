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

set(cases range output gradient)
set(diagnostics
    "Reconstruction: invalid range or input/output extents"
    "Reconstruction: invalid range or input/output extents"
    "Reconstruction: gradient extent is too short")
list(LENGTH cases count)
math(EXPR last "${count}-1")
foreach(i RANGE "${last}")
    list(GET cases "${i}" case)
    list(GET diagnostics "${i}" diagnostic)
    add_test(NAME "numerics.reconstruction.reject_${case}"
        COMMAND "${CMAKE_COMMAND}"
            "-DPROGRAM=$<TARGET_FILE:krona_test_reconstruction>"
            "-DCASE=reject_${case}" "-DEXPECT=${diagnostic}"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
            "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
            "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
    set_tests_properties("numerics.reconstruction.reject_${case}" PROPERTIES
        TIMEOUT 60 RUN_SERIAL TRUE LABELS "numerics;reconstruction;contract"
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
