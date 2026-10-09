add_executable(krona_test_linalg test_linalg.cpp)
target_link_libraries(krona_test_linalg PRIVATE krona_test_support krona::linalg krona::fields)
if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Linalg tests require an MPI launcher")
endif()

foreach(case IN ITEMS distribution numbering numbering_empty subcomm csr device update empty precision fields shared native nonconvergence zero_rhs all_empty)
    set(ranks 2)
    if(case STREQUAL "numbering_empty")
        set(ranks 3)
    endif()
    add_test(NAME "linalg.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_test_linalg> ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("linalg.${case}" PROPERTIES
        LABELS "linalg;numerical" PROCESSORS "${ranks}" TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()

foreach(case IN ITEMS duplicate pending missing offsets)
    add_test(NAME "linalg.${case}"
        COMMAND "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:krona_test_linalg>"
            "-DTEST_CASE=${case}"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
            "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
            "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
    set_tests_properties("linalg.${case}" PROPERTIES LABELS "linalg;contract"
        PROCESSORS 2 TIMEOUT 120 RUN_SERIAL TRUE ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
