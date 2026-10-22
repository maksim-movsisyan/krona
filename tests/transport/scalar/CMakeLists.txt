if(NOT KRONA_ENABLE_TRILINOS)
    return()
endif() 
add_executable(krona_test_scalar_diffusion test_diffusion.cpp test_nonorthogonal.hpp)
target_link_libraries(krona_test_scalar_diffusion PRIVATE krona_test_support krona::transport_scalar)
if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Scalar diffusion tests require an MPI launcher")
endif()
foreach(case IN ITEMS nonorthogonal_linear nonorthogonal_mixed nonorthogonal_float nonorthogonal_orthogonal nonorthogonal_iterations nonorthogonal_weights nonorthogonal_empty matrix_free additive residual bsr_slice linear normal_gradient fixed_flux robin source variable shared shared_assembly nonconvergence duplicate_faces strided_float empty float_solver)
    add_test(NAME "transport.scalar.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1 ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_scalar_diffusion>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("transport.scalar.${case}" PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
endforeach()
foreach(ranks IN ITEMS 2 3)
    add_test(NAME "transport.scalar.mpi${ranks}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}" ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_scalar_diffusion>" ${MPIEXEC_POSTFLAGS})
    set_tests_properties("transport.scalar.mpi${ranks}" PROPERTIES TIMEOUT 180 PROCESSORS "${ranks}" RUN_SERIAL TRUE)
endforeach()
set(cases correction_no_vectors correction_no_weights correction_nan correction_stale correction_short correction_slope pending pure_neumann bad_coefficient bad_source bad_metric missing_weights nonorthogonal bad_boundary bad_batch in_place wrong_graph)
set(expected "correction requires attached FaceNonOrthogonalityVector" "correction requires attached FaceInterpolationWeights"
    "invalid correction data" "normal metrics" "invalid component extent or storage" "correction-dependent boundary slope" "assembly pending" "pure Neumann" "coefficient must be finite and positive" "source density must be finite"
    "normal metrics" "require attached FaceInterpolationWeights" "nonorthogonal geometry" "invalid affine boundary flux"
    "coefficient must be finite and positive" "in-place residual" "must share the face assembly graph")
list(LENGTH cases count)
math(EXPR last "${count}-1")
foreach(i RANGE "${last}")
    list(GET cases "${i}" name)
    list(GET expected "${i}" diagnostic)
    add_test(NAME "transport.scalar.reject_${name}"
        COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:krona_test_scalar_diffusion>"
            "-DCASE=${name}" "-DEXPECT=${diagnostic}" "-DRANKS=1"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
    set_tests_properties("transport.scalar.reject_${name}" PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)
endforeach()
 