execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS} "${TEST_EXECUTABLE}" ${MPIEXEC_POSTFLAGS} "${TEST_CASE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 60)

if("${result}" MATCHES "[Tt]imeout|[Tt]erminated")
    message(FATAL_ERROR "Expected an immediate contract failure, got ${result}\n${output}\n${error}")
endif()

if(NOT "${SKIP_MARKER}" STREQUAL "" AND "${result}" STREQUAL "0")
    string(FIND "${output}\n${error}" "${SKIP_MARKER}" skipped)
    if(NOT skipped EQUAL -1)
        message("${SKIP_MARKER}: coefficient assertion test is unavailable in this Kokkos build")
        return()
    endif()
endif()

string(FIND "${output}\n${error}" "${EXPECTED_DIAGNOSTIC}" found)
if("${result}" STREQUAL "0" OR found EQUAL -1)
    message(FATAL_ERROR "Expected failure containing '${EXPECTED_DIAGNOSTIC}'; result=${result}\n${output}\n${error}")
endif()
