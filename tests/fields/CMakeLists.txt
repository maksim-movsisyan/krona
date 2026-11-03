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

add_executable(krona_test_field_transfer test_transfer.cpp)
target_link_libraries(krona_test_field_transfer PRIVATE krona_test_support krona::fields)
foreach(case IN ITEMS left right types empty)
    add_test(NAME "fields.transfer.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_field_transfer>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("fields.transfer.${case}" PROPERTIES LABELS "fields;unit" TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
foreach(case IN ITEMS mismatch short_input short_output)
    if(case STREQUAL "mismatch")
        set(diagnostic "field pack/DOF size mismatch")
    else()
        set(diagnostic "field is shorter")
    endif()
    add_test(NAME "fields.transfer.reject_${case}"
        COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:krona_test_field_transfer>"
            "-DCASE=${case}" "-DEXPECT=${diagnostic}"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_transfer_failure.cmake")
    set_tests_properties("fields.transfer.reject_${case}" PROPERTIES LABELS "fields;contract" TIMEOUT 60 RUN_SERIAL TRUE
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
 
