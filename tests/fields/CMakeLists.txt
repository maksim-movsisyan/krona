add_executable(krona_test_fields test_fields.cpp)
target_link_libraries(krona_test_fields PRIVATE krona_test_support krona::fields)

foreach(case IN ITEMS registry soa transfers tensor empty fill_types aos extended storage)
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

add_executable(krona_test_field_pack test_pack.cpp)
target_link_libraries(krona_test_field_pack PRIVATE krona_test_support krona::fields)

foreach(case IN ITEMS fields mixed const extents strided nested)
    add_test(NAME "fields.pack.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_field_pack>"
            ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("fields.pack.${case}" PROPERTIES
        LABELS "fields;unit"
        TIMEOUT 120
        RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
 