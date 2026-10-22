add_executable(krona_test_scalar_convection test_convection.cpp)
target_link_libraries(krona_test_scalar_convection PRIVATE krona_test_support krona::transport_scalar)
if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Scalar convection tests require an MPI launcher")
endif()

foreach(case IN ITEMS upwind linear tvd muscl strided_float limiting zero_flux inflow face_inputs divergent bsr additive quick quadratic frozen_system empty)
    add_test(NAME "transport.scalar.convection.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1 ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_scalar_convection>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("transport.scalar.convection.${case}" PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
endforeach()
foreach(ranks IN ITEMS 2 3)
    add_test(NAME "transport.scalar.convection.mpi${ranks}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}" ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_scalar_convection>" ${MPIEXEC_POSTFLAGS} distributed)
    set_tests_properties("transport.scalar.convection.mpi${ranks}" PROPERTIES TIMEOUT 180 PROCESSORS "${ranks}" RUN_SERIAL TRUE)
endforeach() 

set(cases mass_flux inflow short_flux short_inflow short_gradient short_factors component reconstruction in_place pending overflow underflow batch time)
set(expected "mass flux must be finite" "missing or nonfinite inflow value"
    "invalid component extent or storage" "invalid component extent or storage"
    "reconstruction input extent is too short" "reconstruction input extent is too short"
    "invalid reconstruction component" "invalid reconstructed increment" "in-place residual"
    "call begin_async" "nonfinite or unrepresentable" "nonfinite or unrepresentable"
    "source density must be finite" "time must be finite")
list(LENGTH cases count)
math(EXPR last "${count}-1")
foreach(i RANGE "${last}")
    list(GET cases "${i}" name)
    list(GET expected "${i}" diagnostic)
    add_test(NAME "transport.scalar.convection.reject_${name}"
        COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:krona_test_scalar_convection>"
            "-DCASE=reject_${name}" "-DEXPECT=${diagnostic}" "-DRANKS=1"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/../expect_failure.cmake")
    set_tests_properties("transport.scalar.convection.reject_${name}" PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)
endforeach()
