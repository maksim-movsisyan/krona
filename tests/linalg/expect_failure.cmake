execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
            ${MPIEXEC_PREFLAGS} "${TEST_EXECUTABLE}" ${MPIEXEC_POSTFLAGS} "${TEST_CASE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 90)
if(TEST_CASE STREQUAL "duplicate")
    set(expected "Linalg: duplicate column in a CSR row")
elseif(TEST_CASE STREQUAL "pending")
    set(expected "Linalg: matrix has pending writes")
elseif(TEST_CASE STREQUAL "missing")
    set(expected "Linalg: column has no owner in the domain map")
elseif(TEST_CASE STREQUAL "offsets")
    set(expected "Linalg: nonmonotone/out-of-range CSR offsets")
else()
    message(FATAL_ERROR "Unknown expected failure")
endif()
string(FIND "${output}\n${error}" "${expected}" found)
if(result STREQUAL "0" OR result MATCHES "[Tt]imeout" OR found EQUAL -1)
    message(FATAL_ERROR "Expected diagnosed failure (${expected}); result=${result}\n${output}\n${error}")
endif()
