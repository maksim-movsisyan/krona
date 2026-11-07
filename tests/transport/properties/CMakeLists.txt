# Property-law tests compare analytic coefficients, scalar types and device evaluation. 
add_executable(krona_test_transport_properties test_properties.cpp)
target_link_libraries(krona_test_transport_properties PRIVATE
    krona_test_support krona::transport_properties krona::thermodynamics)

if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Transport property tests require an MPI launcher")
endif()

foreach(case IN ITEMS constants sutherland prandtl custom precision invalid_states eos device)
    add_test(NAME "transport.properties.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1 ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:krona_test_transport_properties>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("transport.properties.${case}" PROPERTIES
        TIMEOUT 120 RUN_SERIAL TRUE LABELS "transport;properties"
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach() 
  
# Configuration errors and exceptions in user validators must produce diagnosed fatal exits.
set(cases mu k cp pr sutherland_mu sutherland_temperature sutherland_shift custom_viscosity custom_cp)
set(messages
    "constant dynamic viscosity must be finite and nonnegative"
    "constant thermal conductivity must be finite and nonnegative"
    "constant cp must be finite and positive"
    "Prandtl number must be finite and positive"
    "Sutherland requires finite mu_ref>0, T_ref>0 and S>=0"
    "Sutherland requires finite mu_ref>0, T_ref>0 and S>=0"
    "Sutherland requires finite mu_ref>0, T_ref>0 and S>=0"
    "custom viscosity validation failed"
    "custom heat capacity validation failed")
list(LENGTH cases count)
math(EXPR last "${count}-1") 
foreach(i RANGE "${last}")
    list(GET cases "${i}" case)
    list(GET messages "${i}" expected)
    add_test(NAME "transport.properties.reject_${case}"
        COMMAND "${CMAKE_COMMAND}"
            "-DPROGRAM=$<TARGET_FILE:krona_test_transport_properties>"
            "-DCASE=reject_${case}" "-DEXPECT=${expected}"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
            "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
            "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_LIST_DIR}/expect_failure.cmake")
    set_tests_properties("transport.properties.reject_${case}" PROPERTIES
        TIMEOUT 120 RUN_SERIAL TRUE LABELS "transport;properties;validation"
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
