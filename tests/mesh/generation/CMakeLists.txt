# Generator tests use MPI and CGNS directly; no Kokkos test-support target.
if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Mesh generator tests require an MPI launcher")
endif()

add_executable(krona_test_mesh_generator test_generator.cpp)
target_link_libraries(krona_test_mesh_generator PRIVATE
    krona_test_support_host_only krona::mesh_generation krona::cgns)
target_compile_definitions(krona_test_mesh_generator PRIVATE
    "KRONA_GENERATOR_TEST_DATA=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")

# Resolve write() from this translation unit so 64-byte buffers exercise chunking.
add_executable(krona_test_mesh_generator_chunks test_generator.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/mesh/generation/writer.cpp")
target_link_libraries(krona_test_mesh_generator_chunks PRIVATE
    krona_test_support_host_only krona::mesh_generation krona::cgns)
target_compile_definitions(krona_test_mesh_generator_chunks PRIVATE
    KRONA_MESH_WRITE_BUFFER_BYTES=64
    "KRONA_GENERATOR_TEST_DATA=\"${CMAKE_CURRENT_SOURCE_DIR}/data\"")

function(krona_generator_test target case ranks)
    set(name "mesh.generator.${case}.np${ranks}")
    if(target STREQUAL "krona_test_mesh_generator_chunks")
        string(APPEND name ".chunks")
    endif()
    set(workdir "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    file(MAKE_DIRECTORY "${workdir}")
    add_test(NAME "${name}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
            ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:${target}>" ${MPIEXEC_POSTFLAGS}
            "${case}")
    set_tests_properties("${name}" PROPERTIES
        LABELS "mesh;generator;mpi" PROCESSORS "${ranks}" RUN_SERIAL TRUE TIMEOUT 180
        WORKING_DIRECTORY "${workdir}")
endfunction()

foreach(case IN ITEMS topology grading config write)
    krona_generator_test(krona_test_mesh_generator "${case}" 1)
endforeach()
foreach(case IN ITEMS topology config write)
    krona_generator_test(krona_test_mesh_generator "${case}" 3)
endforeach()
krona_generator_test(krona_test_mesh_generator write_empty 4)
krona_generator_test(krona_test_mesh_generator_chunks write 3)
krona_generator_test(krona_test_mesh_generator_chunks write_empty 4)

set(expected_zero_cells "Cell counts must be positive")
set(expected_bad_vertex "vertex index is out of bounds")
set(expected_unused_vertex "Every macro vertex must be used")
set(expected_duplicate_block "Duplicate blocks")
set(expected_inverted "inverted or degenerate")
set(expected_nonfinite "coordinates must be finite")
set(expected_grading_type "Invalid grading type")
set(expected_collapsed "Grading collapses adjacent nodes")
set(expected_shared_counts "cell count mismatch")
set(expected_shared_grading "incompatible grading")
set(expected_internal_boundary "internal block face")
set(expected_duplicate_boundary "assigned more than once")
set(expected_crossed_boundary "face perimeter")
set(expected_parse_syntax "Error while parsing")
set(expected_parse_unknown "unexpected key")
set(expected_parse_integer "nonnegative integer")
set(expected_parse_scale "finite number") 
set(expected_parse_nonfinite "finite number")
set(expected_parse_array "must be an array")
set(expected_parse_coordinates "three numbers")
foreach(case IN ITEMS zero_cells bad_vertex unused_vertex duplicate_block inverted nonfinite
                      grading_type collapsed shared_counts shared_grading internal_boundary
                      duplicate_boundary crossed_boundary parse_syntax parse_unknown parse_integer
                      parse_scale parse_nonfinite parse_array parse_coordinates)
    set(workdir "${CMAKE_CURRENT_BINARY_DIR}/reject_${case}")
    file(MAKE_DIRECTORY "${workdir}")
    add_test(NAME "mesh.generator.reject_${case}"
        COMMAND "${CMAKE_COMMAND}"
            "-DPROGRAM=$<TARGET_FILE:krona_test_mesh_generator>" "-DCASE=${case}"
            "-DEXPECT=${expected_${case}}" "-DRANKS=2"
            "-DOUTPUT=${workdir}/invalid.toml"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
    set_tests_properties("mesh.generator.reject_${case}" PROPERTIES
        LABELS "mesh;generator;contract" PROCESSORS 2 RUN_SERIAL TRUE TIMEOUT 60
        WORKING_DIRECTORY "${workdir}")
endforeach()
