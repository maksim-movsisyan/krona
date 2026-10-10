add_executable(krona_test_diffusion test_diffusion.cpp)
target_link_libraries(krona_test_diffusion PRIVATE krona_test_support krona::transport_scalar)
if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Scalar diffusion tests require an MPI launcher")
endif()

foreach(case IN ITEMS assembly duplicate_faces affine_3d mixed variable refinement strided nonconvergence all_empty publication subcomm)
    foreach(ranks IN ITEMS 1 2)
        add_test(NAME "transport.scalar.${case}.mpi${ranks}"
            COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_test_diffusion> ${MPIEXEC_POSTFLAGS} "${case}")
        set_tests_properties("transport.scalar.${case}.mpi${ranks}" PROPERTIES
            LABELS "transport;scalar;numerical" PROCESSORS "${ranks}" TIMEOUT 120 RUN_SERIAL TRUE
            ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
    endforeach()
endforeach()
add_test(NAME "transport.scalar.empty_rank.mpi3"
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 3
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_test_diffusion> ${MPIEXEC_POSTFLAGS} empty_rank)
set_tests_properties("transport.scalar.empty_rank.mpi3" PROPERTIES
    LABELS "transport;scalar;numerical" PROCESSORS 3 TIMEOUT 120 RUN_SERIAL TRUE
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

foreach(case IN ITEMS pure_neumann negative_diffusion bad_source negative_boundary source_extent pending nonorthogonal)
    add_test(NAME "transport.scalar.${case}"
        COMMAND "${CMAKE_COMMAND}"
            "-DTEST_EXECUTABLE=$<TARGET_FILE:krona_test_diffusion>" "-DTEST_CASE=${case}"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
    set_tests_properties("transport.scalar.${case}" PROPERTIES
        LABELS "transport;scalar;contract" PROCESSORS 2 TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
