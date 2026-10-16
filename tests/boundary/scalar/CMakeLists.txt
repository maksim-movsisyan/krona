add_executable(krona_test_scalar_boundary test_scalar_boundary.cpp)
target_link_libraries(krona_test_scalar_boundary PRIVATE
    krona_test_support krona::boundary_scalar)
target_compile_definitions(krona_test_scalar_boundary PRIVATE
    "KRONA_BOUNDARY_TEST_DATA=\"${CMAKE_CURRENT_SOURCE_DIR}/data/bc.toml\"")

if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Scalar boundary tests require an MPI launcher")
endif()

foreach(case IN ITEMS reconstruction visitation custom relations empty independent_value
                      boundary_gradients config file_fill layouts affine assembly empty_labels distributed)
    set(ranks 1)
    if(case STREQUAL "distributed")
        set(ranks 2)
    endif()
    add_test(NAME "boundary.scalar.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_test_scalar_boundary>
                ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("boundary.scalar.${case}" PROPERTIES
        LABELS "boundary;scalar;numerical" PROCESSORS "${ranks}" TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()

set(expected_missing_law "FIXED_FLUX requires a flux law")
set(expected_missing_binding "missing condition on a labelled patch")
set(expected_wrong_name "patch name does not match")
set(expected_scattered "contiguous face interval")
set(expected_unlabelled "unlabelled faces need an explicit condition")
set(expected_unknown_type "unsupported scalar type")
set(expected_typo "unexpected key")
set(expected_duplicate "duplicate patch_id")
set(expected_invalid_geometry "invalid owner, unit normal, area or positive normal distance")
set(expected_missing_scalar "missing scalar condition table")
set(expected_short_field "extent is too small")
set(expected_short_correction "extent is too small")
set(expected_missing_law_empty "FIXED_FLUX requires a flux law")
set(expected_zero_coefficient "value > Real{0}")

foreach(case IN ITEMS missing_law missing_binding wrong_name scattered unlabelled unknown_type typo
                      duplicate invalid_geometry missing_scalar short_field short_correction
                      missing_law_empty zero_coefficient)
    set(skip_marker "")
    if(case STREQUAL "zero_coefficient")
        set(skip_marker "KRONA_TEST_SKIP_NO_KOKKOS_ASSERT")
    endif()
    add_test(NAME "boundary.scalar.reject_${case}"
        COMMAND "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:krona_test_scalar_boundary>"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
            "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
            "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DTEST_CASE=${case}" "-DEXPECTED_DIAGNOSTIC=${expected_${case}}"
            "-DSKIP_MARKER=${skip_marker}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
    set_tests_properties("boundary.scalar.reject_${case}" PROPERTIES
        LABELS "boundary;scalar;contract" PROCESSORS 1 TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
    if(NOT skip_marker STREQUAL "")
        set_tests_properties("boundary.scalar.reject_${case}" PROPERTIES
            SKIP_REGULAR_EXPRESSION "${skip_marker}")
    endif()
endforeach()
