# Optional: enable_testing() in the root, then add_subdirectory(tests/cgns).
if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "CGNS MPI tests require MPIEXEC_EXECUTABLE")
endif()
add_executable(krona_cgns_fixture fixture.cpp)
target_compile_features(krona_cgns_fixture PRIVATE cxx_std_20)
target_link_libraries(krona_cgns_fixture PRIVATE krona::cgns)
add_executable(krona_cgns_reader_probe reader_probe.cpp)
target_link_libraries(krona_cgns_reader_probe PRIVATE krona::cgns_io krona::warnings)
add_executable(krona_mesh_core_probe core.cpp)
target_link_libraries(krona_mesh_core_probe PRIVATE krona::mesh_core krona::warnings)
add_test(NAME krona.mesh.core COMMAND krona_mesh_core_probe)

set(_fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
file(MAKE_DIRECTORY "${_fixture_dir}")
set(_positive_modes uniform mixed mixed-i4 cross-mixed many-sections
    2d 2d-mixed 2d-embedded no-bc no-boundary ignore)
set(_negative_modes vertex conflict missing-id bad-node nan malformed-mixed)
set(_sources ${_positive_modes} ${_negative_modes})
list(REMOVE_ITEM _sources ignore)
list(REMOVE_DUPLICATES _sources)
foreach(_mode IN LISTS _sources)
    add_test(NAME "krona.cgns.fixture.${_mode}"
        COMMAND krona_cgns_fixture "${_fixture_dir}/${_mode}.cgns" "${_mode}")
    set_tests_properties("krona.cgns.fixture.${_mode}" PROPERTIES
        FIXTURES_SETUP "cgns.${_mode}" TIMEOUT 30)
endforeach()
foreach(_mode IN LISTS _positive_modes)
    set(_source "${_mode}")
    if(_mode STREQUAL ignore)
        set(_source vertex)
    endif()
    foreach(_ranks IN ITEMS 1 2 4 11)
        if(DEFINED MPIEXEC_MAX_NUMPROCS AND _ranks GREATER MPIEXEC_MAX_NUMPROCS)
            continue()
        endif()
        add_test(NAME "krona.cgns.${_mode}.${_ranks}"
            COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${_ranks}"
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_cgns_reader_probe>
                ${MPIEXEC_POSTFLAGS} "${_fixture_dir}/${_source}.cgns" "${_mode}" 2)
        set_tests_properties("krona.cgns.${_mode}.${_ranks}" PROPERTIES
            FIXTURES_REQUIRED "cgns.${_source}" PROCESSORS "${_ranks}" TIMEOUT 45)
    endforeach()
endforeach()
# Check exact importer diagnostics, not merely any nonzero process exit.
set(_error_patterns "Unsupported BC" "conflicting BC patches"
    "non-boundary or missing element" "invalid node ID"
    "Non-finite coordinate" "MIXED offsets")
list(LENGTH _negative_modes _negative_count)
math(EXPR _last_negative "${_negative_count} - 1")
foreach(_index RANGE 0 ${_last_negative})
    list(GET _negative_modes ${_index} _mode)
    list(GET _error_patterns ${_index} _pattern)
    add_test(NAME "krona.cgns.reject.${_mode}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_cgns_reader_probe>
            ${MPIEXEC_POSTFLAGS} "${_fixture_dir}/${_mode}.cgns" negative 2)
    set_tests_properties("krona.cgns.reject.${_mode}" PROPERTIES
        FIXTURES_REQUIRED "cgns.${_mode}" PASS_REGULAR_EXPRESSION "${_pattern}" TIMEOUT 45)
endforeach()
unset(_fixture_dir)
unset(_positive_modes)
unset(_negative_modes)
unset(_sources)
unset(_error_patterns)
unset(_negative_count)
unset(_last_negative)

# Actual CLI import/extrusion on valid planar MIXED meshes.
if(TARGET krona_mesh_partition AND TARGET krona::mesh_operations)
    add_executable(krona_cgns_extrusion_fixture extrusion_fixture.cpp)
    target_link_libraries(krona_cgns_extrusion_fixture PRIVATE krona::cgns krona::warnings)
    set(_extrusion_dir "${CMAKE_CURRENT_BINARY_DIR}/extrusion-fixtures")
    file(MAKE_DIRECTORY "${_extrusion_dir}")
    foreach(_plane IN ITEMS xy xz yz)
        add_test(NAME "krona.cgns.extrusion.fixture.${_plane}"
            COMMAND krona_cgns_extrusion_fixture "${_extrusion_dir}/${_plane}.cgns" "${_plane}")
        set_tests_properties("krona.cgns.extrusion.fixture.${_plane}" PROPERTIES
            FIXTURES_SETUP "extrusion.${_plane}" TIMEOUT 30)
        foreach(_ranks IN ITEMS 1 4 11)
            if(DEFINED MPIEXEC_MAX_NUMPROCS AND _ranks GREATER MPIEXEC_MAX_NUMPROCS)
                continue()
            endif()
            add_test(NAME "krona.cgns.extrusion.cli.${_plane}.${_ranks}"
                COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${_ranks}"
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_mesh_partition>
                    ${MPIEXEC_POSTFLAGS} "${_extrusion_dir}/${_plane}.cgns"
                    "${_extrusion_dir}/${_plane}.unused.h5" -v)
            set_tests_properties("krona.cgns.extrusion.cli.${_plane}.${_ranks}" PROPERTIES
                FIXTURES_REQUIRED "extrusion.${_plane}" PROCESSORS "${_ranks}" TIMEOUT 45
                PASS_REGULAR_EXPRESSION "imported: cells=2 nodes=10 boundary=9")
        endforeach()
    endforeach()
    add_test(NAME krona.partition.help
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_mesh_partition> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(krona.partition.help PROPERTIES TIMEOUT 45
        PASS_REGULAR_EXPRESSION "Usage:")
endif()
