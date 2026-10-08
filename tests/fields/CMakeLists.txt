add_executable(krona_test_fields test_fields.cpp)
target_link_libraries(krona_test_fields PRIVATE krona_test_support krona::fields)

foreach(case IN ITEMS registry soa transfers tensor empty)
    add_test(
    NAME "fields.${case}"
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:krona_test_fields>
            ${MPIEXEC_POSTFLAGS}
            "${case}")
    set_tests_properties("fields.${case}" PROPERTIES
        LABELS "fields;unit"
        TIMEOUT 120
        RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
 