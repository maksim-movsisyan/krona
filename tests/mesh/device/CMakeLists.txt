add_executable(krona_test_mesh_device test_mesh.cpp)
target_link_libraries(krona_test_mesh_device PRIVATE krona_test_support krona::mesh_device)


# This suite exercises selected views and kernels in the configured execution space.
foreach(case IN ITEMS selection geometry ownership halo_plan auxiliary empty
                      transfer auxiliary_contracts validation halo_validation)
    add_test(NAME "mesh.device.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_mesh_device>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("mesh.device.${case}" PROPERTIES
        LABELS "mesh;device;unit" PROCESSORS 1 TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
