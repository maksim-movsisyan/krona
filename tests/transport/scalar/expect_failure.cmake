foreach(required IN ITEMS PROGRAM CASE EXPECT MPIEXEC_EXECUTABLE MPIEXEC_NUMPROC_FLAG)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

if(NOT DEFINED RANKS)
    set(RANKS 1)
endif() 
execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${RANKS}"
        ${MPIEXEC_PREFLAGS} "${PROGRAM}" ${MPIEXEC_POSTFLAGS} "${CASE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 45)

# Keep the child's progress/diagnostic visible in ctest -V, including failures.
message("${output}${error}")
if("${result}" STREQUAL "0")
    message(FATAL_ERROR "${CASE} unexpectedly succeeded")
endif()
if("${result}" MATCHES "[Tt]imeout")
    message(FATAL_ERROR "${CASE} did not terminate within 45 seconds: ${result}")
endif()
if(NOT "${output}\n${error}" MATCHES "${EXPECT}")
    message(FATAL_ERROR "${CASE} failed for the wrong reason: ${result}; expected '${EXPECT}'")
endif()
