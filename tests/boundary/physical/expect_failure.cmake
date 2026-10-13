# A failure counts only when it reports the intended contract diagnostic.
set(expected_missing "Physical boundary: missing labelled condition")
set(expected_wrong_name "Physical boundary: duplicate patch ID or mismatched name")
set(expected_duplicate "Physical boundary: duplicate patch ID or mismatched name")
set(expected_invalid_geometry "Physical boundary: invalid owner, normal, area or normal distance")
set(expected_invalid_model "Physical boundary: invalid EOS parameters")
set(expected_extent "Physical boundary: field extent or storage is invalid")
set(expected_temperature_without_energy "Physical boundary: this flow has no temperature")
set(expected_unknown_type "Physical boundary TOML: unsupported physical type")
set(expected_parser_duplicate "Physical boundary TOML: duplicate patch_id")
set(expected_unlabelled "Physical boundary: unlabelled faces need an explicit condition")
set(expected_typo "Physical boundary TOML: unexpected key or inapplicable physical parameter")
# toml++ reports this in both exception and parse-result modes.
set(expected_malformed_toml "table header")
if(NOT DEFINED expected_${CASE} OR "${expected_${CASE}}" STREQUAL "")
    message(FATAL_ERROR "Unknown rejection case: ${CASE}")
endif()

execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS} "${TEST_EXECUTABLE}" ${MPIEXEC_POSTFLAGS} "reject_${CASE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
if(result MATCHES "timeout|Timeout|terminated|Terminated")
    message(FATAL_ERROR "Expected an immediate contract failure, got ${result}")
endif()
if(result STREQUAL "0")
    message(FATAL_ERROR "Invalid input was accepted")
endif()
string(FIND "${output}\n${error}" "${expected_${CASE}}" found)
if(found EQUAL -1)
    message(FATAL_ERROR "Unexpected failure for ${CASE}: ${result}\n${output}\n${error}")
endif()
