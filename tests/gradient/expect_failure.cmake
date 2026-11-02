# Require the intended diagnostic as well as failure; unrelated crashes do not pass.
if(NOT DEFINED CASE)
    set(CASE singular)
endif()
if(NOT DEFINED EXPECT)
    set(EXPECT "Gradient: singular/ill-conditioned")
endif()
execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS} "${TEST_EXECUTABLE}" ${MPIEXEC_POSTFLAGS} "${CASE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 60)
if("${result}" STREQUAL "0" OR NOT "${result}" MATCHES "^-?[0-9]+$")
    message(FATAL_ERROR "Expected a diagnosed failure; result=${result}\n${output}\n${error}")
endif()
string(FIND "${output}\n${error}" "${EXPECT}" match)
if(match EQUAL -1)
    message(FATAL_ERROR "Expected '${EXPECT}'; result=${result}\n${output}\n${error}")
endif()
 