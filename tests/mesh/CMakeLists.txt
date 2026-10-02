if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Mesh MPI tests require MPIEXEC_EXECUTABLE")
endif()
add_executable(krona_mesh_extrusion_probe extrusion.cpp)
target_link_libraries(krona_mesh_extrusion_probe PRIVATE krona::mesh_operations krona::warnings)
foreach(_mode IN ITEMS xy xz yz no-bc no-boundary custom)
    foreach(_ranks IN ITEMS 1 2 4 11)
        if(DEFINED MPIEXEC_MAX_NUMPROCS AND _ranks GREATER MPIEXEC_MAX_NUMPROCS)
            continue()
        endif()
        add_test(NAME "krona.mesh.extrusion.${_mode}.${_ranks}"
            COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${_ranks}"
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_mesh_extrusion_probe>
                ${MPIEXEC_POSTFLAGS} "${_mode}")
        set_tests_properties("krona.mesh.extrusion.${_mode}.${_ranks}" PROPERTIES
            PROCESSORS "${_ranks}" TIMEOUT 45)
    endforeach()
endforeach()
set(_reject_ranks 4)
if(DEFINED MPIEXEC_MAX_NUMPROCS AND MPIEXEC_MAX_NUMPROCS LESS _reject_ranks)
    set(_reject_ranks "${MPIEXEC_MAX_NUMPROCS}")
endif()
set(_negative_modes warped degenerate concave overflow collapse collision bad-node bad-csr)
set(_errors "axis-aligned planar" "Degenerate extrusion cell" "Non-convex or folded"
    "global counts" "layer collapses" "cap name collides" "Invalid extrusion node ID"
    "Invalid extrusion CSR")
list(LENGTH _negative_modes _count)
math(EXPR _last "${_count} - 1")
foreach(_index RANGE 0 ${_last})
    list(GET _negative_modes ${_index} _mode)
    list(GET _errors ${_index} _error)
    add_test(NAME "krona.mesh.extrusion.reject.${_mode}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${_reject_ranks}"
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_mesh_extrusion_probe>
            ${MPIEXEC_POSTFLAGS} "${_mode}")
    set_tests_properties("krona.mesh.extrusion.reject.${_mode}" PROPERTIES
        PASS_REGULAR_EXPRESSION "${_error}" PROCESSORS "${_reject_ranks}" TIMEOUT 45)
endforeach()
