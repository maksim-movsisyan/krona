# A rejected contract must terminate MPI and report the intended invariant.
execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${RANKS}"
        ${MPIEXEC_PREFLAGS} "${PROGRAM}" ${MPIEXEC_POSTFLAGS} "${CASE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 40)
if(NOT result MATCHES "^-?[0-9]+$" OR result EQUAL 0)
    message(FATAL_ERROR "Expected a failed MPI job, got '${result}'\n${out}\n${err}")
endif()
string(FIND "${out}\n${err}" "${EXPECT}" found)
if(found LESS 0)
    message(FATAL_ERROR "Missing diagnostic '${EXPECT}'\n${out}\n${err}")
endif()
