add_executable(krona_test_viscous_transport test_faces.cpp test_operator.cpp)
target_link_libraries(krona_test_viscous_transport PRIVATE krona_test_support krona::transport_viscous
    krona::mesh_device krona::thermodynamics)

foreach(case IN ITEMS skew_face inner_jacobian walls rows)
    add_test(NAME "transport.viscous.${case}" COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
        ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_viscous_transport>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("transport.viscous.${case}" PROPERTIES TIMEOUT 120 LABELS "transport;viscous"
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()

add_test(NAME transport.viscous.mpi2 COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
    ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_viscous_transport>" ${MPIEXEC_POSTFLAGS} distributed)
set_tests_properties(transport.viscous.mpi2 PROPERTIES TIMEOUT 120 PROCESSORS 2 LABELS "transport;viscous;mpi"
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
