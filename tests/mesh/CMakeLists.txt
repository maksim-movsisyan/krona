add_executable(krona_test_mesh test_mesh.cpp)
target_link_libraries(krona_test_mesh PRIVATE krona_test_support krona::mesh_device)

foreach(case IN ITEMS selection geometry ownership halo_plan auxiliary empty
                      transfer auxiliary_contracts validation halo_validation)
    add_test(
    NAME "mesh.${case}"
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:krona_test_mesh>
            ${MPIEXEC_POSTFLAGS}
            "${case}")

    set_tests_properties("mesh.${case}" PROPERTIES
        LABELS "mesh;unit"
        TIMEOUT 120
        RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
 