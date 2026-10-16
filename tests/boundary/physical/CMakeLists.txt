add_executable(krona_test_physical_boundary test_physical.cpp)
target_link_libraries(krona_test_physical_boundary PRIVATE
    krona_test_support krona::boundary_physical)
target_compile_definitions(krona_test_physical_boundary PRIVATE
    "KRONA_PHYSICAL_TEST_DATA=\"${CMAKE_CURRENT_SOURCE_DIR}/data/bc.toml\"")

if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Physical boundary tests require an MPI launcher")
endif()
foreach(case IN ITEMS thermodynamics walls evaluate incompressible thermal_liquid extension characteristics relations
                      visitation configuration empty custom distributed)
    set(ranks 1)
    if(case STREQUAL "distributed")
        set(ranks 2)
    endif()
    add_test(NAME "boundary.physical.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_test_physical_boundary>
                ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("boundary.physical.${case}" PROPERTIES
        LABELS "boundary;physical;numerical" PROCESSORS "${ranks}" TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()

set(expected_reverse "reverse flow requires explicit backflow data")
set(expected_supersonic "not supersonic in the inward normal direction")
set(expected_direction "inlet direction must point into the domain")
set(expected_choking "total inlet choked")
set(expected_no_temperature "isothermal wall requires a temperature field")
set(expected_no_characteristics "pressure outlet requires a characteristic policy")
set(expected_missing_law "heat-flux wall requires an explicit flux law")
set(expected_missing_binding "missing condition on a labelled patch")
set(expected_conflict "scalar.temperature conflicts")
set(expected_unknown "unsupported physical type")
set(expected_typo "unexpected key")
set(expected_duplicate "duplicate patch_id")
set(expected_name "patch name does not match")
set(expected_short_field "invalid or undersized primitive component")
set(expected_short_correction "invalid or undersized primitive component")
foreach(case IN ITEMS reverse supersonic direction choking no_temperature no_characteristics
                      missing_law missing_binding conflict unknown typo duplicate name short_field short_correction)
    add_test(NAME "boundary.physical.reject_${case}"
        COMMAND "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:krona_test_physical_boundary>"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
            "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
            "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DTEST_CASE=reject_${case}" "-DEXPECTED_DIAGNOSTIC=${expected_${case}}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
    set_tests_properties("boundary.physical.reject_${case}" PROPERTIES
        LABELS "boundary;physical;contract" PROCESSORS 1 TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
