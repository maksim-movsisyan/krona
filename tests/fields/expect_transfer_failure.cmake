# Isolate the fatal contract violation in a separate MPI process.
execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
        ${MPIEXEC_PREFLAGS} "${PROGRAM}" ${MPIEXEC_POSTFLAGS} "${CASE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
# Timeouts, successful exits and unrelated failures must not count as a passing test.
if(NOT result MATCHES "^[0-9]+$" OR result EQUAL 0)
    message(FATAL_ERROR "Expected a diagnosed nonzero exit, got ${result}: ${output}${error}")
endif()
string(FIND "${output}${error}" "${EXPECT}" position)
if(position EQUAL -1)
    message(FATAL_ERROR "Missing diagnostic '${EXPECT}': ${output}${error}")
endif()
