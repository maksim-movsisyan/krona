# Native-entry diffusion tests compare corrected assembly with direct residuals. 
add_executable(krona_test_diffusion_core test_operator.cpp fixture.hpp)
target_link_libraries(krona_test_diffusion_core PRIVATE krona_test_support krona::transport_scalar_diffusion krona::halo)
if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Diffusion core tests require an MPI launcher")
endif()
foreach(case IN ITEMS orthogonal nonorthogonal mixed_profiles float_storage empty duplicate custom_correction uncached weights split_ranges)
    add_test(NAME "transport.scalar.diffusion.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1 ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_diffusion_core>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("transport.scalar.diffusion.${case}" PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE LABELS "transport;diffusion")
endforeach()
foreach(ranks IN ITEMS 2 3)
    add_test(NAME "transport.scalar.diffusion.mpi${ranks}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}" ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_diffusion_core>" ${MPIEXEC_POSTFLAGS})
    set_tests_properties("transport.scalar.diffusion.mpi${ranks}" PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE PROCESSORS "${ranks}" LABELS "transport;diffusion;mpi")
endforeach()
# Pair each invalid metric, descriptor or coefficient with the diagnostic it must trigger.
set(cases stale_metric stale_weight null_comm short_gradient nan_gradient bad_coefficient bad_profile bad_face_range)
set(messages "normal metrics" "normal metrics" "null communicator"
    "gradient extent is too short" "invalid correction data" "coefficient must be finite and positive" "invalid affine boundary flux"
    "invalid inner face range")
list(LENGTH cases count)
math(EXPR last "${count}-1") 
foreach(i RANGE "${last}") 
    list(GET cases "${i}" case)
    list(GET messages "${i}" expected)
    add_test(NAME "transport.scalar.diffusion.reject_${case}"
        COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:krona_test_diffusion_core>"
            "-DCASE=${case}" "-DEXPECT=${expected}" "-DRANKS=1"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../expect_failure.cmake")
    set_tests_properties("transport.scalar.diffusion.reject_${case}" PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE LABELS "transport;diffusion;contract")
endforeach()
