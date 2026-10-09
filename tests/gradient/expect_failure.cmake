# Require the intended diagnostic as well as failure; unrelated crashes do not pass.
execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS} "${TEST_EXECUTABLE}" ${MPIEXEC_POSTFLAGS} singular
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 60)
if("${result}" STREQUAL "0" OR NOT "${output}\n${error}" MATCHES "Gradient: singular/ill-conditioned")
    message(FATAL_ERROR "Expected a diagnosed singular stencil; result=${result}\n${output}\n${error}")
endif()
