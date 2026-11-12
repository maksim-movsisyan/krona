add_executable(krona_test_mesh_host test_mesh.cpp)
target_link_libraries(krona_test_mesh_host PRIVATE krona_test_support_host_only krona::mesh_host)

# Four ranks exceed the three-cube and two-cell extrusion fixtures
foreach(ranks IN ITEMS 1 2 4)
    foreach(case IN ITEMS topology batching unlabelled migration unequal_ownership geometry
                          geometry_scales reordering auxiliary_connectivity auxiliary_geometry
                          wall_distance extrusion empty)
        set(name "mesh.host.${case}.np${ranks}")
        add_test(NAME "${name}"
            COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
                ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_mesh_host>" ${MPIEXEC_POSTFLAGS} "${case}")
        set_tests_properties("${name}" PROPERTIES
            LABELS "mesh;host;mpi" PROCESSORS "${ranks}" RUN_SERIAL TRUE TIMEOUT 180)
    endforeach()
endforeach()

set(expected_nonmanifold "Non-manifold face")
set(expected_internal_boundary "Boundary element matches an interior face")
set(expected_inverted "Nonpositive or nonfinite cell geometry")
set(expected_permutation "Owned-cell permutation is not a bijection")
set(expected_halo "Halo receive references a non-ghost cell")
foreach(case IN ITEMS nonmanifold internal_boundary inverted permutation halo)
    add_test(NAME "mesh.host.reject_${case}"
        COMMAND "${CMAKE_COMMAND}"
            "-DPROGRAM=$<TARGET_FILE:krona_test_mesh_host>" "-DCASE=reject_${case}"
            "-DEXPECT=${expected_${case}}" "-DRANKS=2"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../expect_failure.cmake")
    set_tests_properties("mesh.host.reject_${case}" PROPERTIES
        LABELS "mesh;host;contract" PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60)
endforeach()
