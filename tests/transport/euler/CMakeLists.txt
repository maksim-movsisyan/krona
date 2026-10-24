add_executable(krona_test_euler test_euler.cpp)
target_link_libraries(krona_test_euler PRIVATE krona_test_support krona::execution krona::mpi)
if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Euler tests require an MPI launcher")
endif()
foreach(case IN ITEMS uniform split_additive affine reconstruction_fallback walls layouts partition_algebra custom_time empty prepared_profile unlabelled eos_domain geometry)
    add_test(NAME "transport.euler.${case}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
        ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_euler>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("transport.euler.${case}" PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE LABELS "transport;euler"
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
foreach(ranks IN ITEMS 2 3)
    add_test(NAME "transport.euler.mpi${ranks}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
        ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_euler>" ${MPIEXEC_POSTFLAGS} distributed)
    set_tests_properties("transport.euler.mpi${ranks}" PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE PROCESSORS "${ranks}" LABELS "transport;euler;mpi"
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
set(cases alias state boundary)
set(messages "input/output alias is unsupported" "invalid cell thermodynamic state" "invalid exterior boundary state")
foreach(i RANGE 0 2)
    list(GET cases ${i} case)
    list(GET messages ${i} expected)
    add_test(NAME "transport.euler.reject_${case}" COMMAND "${CMAKE_COMMAND}"
        "-DPROGRAM=$<TARGET_FILE:krona_test_euler>" "-DCASE=reject_${case}" "-DEXPECT=${expected}"
        "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
        "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
        -P "${CMAKE_CURRENT_LIST_DIR}/expect_failure.cmake")
    set_tests_properties("transport.euler.reject_${case}" PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE LABELS "transport;euler;validation"
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
