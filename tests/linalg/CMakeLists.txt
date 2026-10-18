if(NOT KRONA_ENABLE_TRILINOS)
    return()
endif()

add_executable(krona_test_linalg test_linalg.cpp)
target_link_libraries(krona_test_linalg PRIVATE krona_test_support krona::linalg)

if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Linalg tests require an MPI launcher")
endif()

function(krona_add_linalg_test name ranks)
    add_test(NAME "linalg.${name}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
            ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_linalg>" ${MPIEXEC_POSTFLAGS} ${ARGN})
    set_tests_properties("linalg.${name}" PROPERTIES
        LABELS "linalg;mpi"
        PROCESSORS "${ranks}"
        TIMEOUT 180
        RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endfunction()

# A small first case isolates MPI/Kokkos/Tpetra startup from numerical work.
set(cases initialization mapping stopping empty minres_indefinite fields_left fields_right
    mesh_adapter precision_bridge gmres fgmres bicgstab cg minres ifpack amg amg_reuse
    mixed_precision float_solver bsr_mapping bsr_solvers bsr_ifpack bsr_amg bsr_amg_reuse
    bsr_mixed_precision bsr_empty bsr_factory multiple_fields_left multiple_fields_right)
foreach(case IN LISTS cases)
    krona_add_linalg_test("${case}" 1 "${case}")
endforeach()
set_tests_properties(linalg.initialization PROPERTIES TIMEOUT 60)

# No case argument runs all positive cases with one MPI/Kokkos startup.
foreach(ranks IN ITEMS 2 3)
    krona_add_linalg_test("mpi${ranks}" "${ranks}")
endforeach()

set(cases duplicate_column missing_owner pending_read pending_write pending_matrix_read
    bicgstab_left bad_method precision_overflow bsr_zero_block bsr_wrong_map bsr_pending_read
    bsr_wrong_values bsr_precision_overflow pack_mismatch pack_short expanded_gid_overflow bsr_different_blocks
    bsr_overlap bsr_kokkos_amg)
set(expected "duplicate column" "column has no owner" "pending reads or writes"
    "pending reads or writes" "pending reads or writes" "requires right preconditioning"
    "NO_SUCH_METHOD" "nonfinite in the preconditioner precision" "nonpositive component count"
    "vectors must share" "pending reads or writes" "wrong BSR coefficient count"
    "nonfinite in the preconditioner precision" "field pack/DOF size mismatch"
    "field is shorter" "expanded GID overflows" "component count differs between ranks"
    "block SCHWARZ requires zero overlap" "block AMG requires use kokkos refactor=false")
list(LENGTH cases count)
math(EXPR last "${count}-1")
foreach(i RANGE "${last}")
    list(GET cases "${i}" name)
    list(GET expected "${i}" diagnostic)
    if(name STREQUAL "bsr_different_blocks")
        set(ranks 2)
    else()
        set(ranks 1)
    endif()
    add_test(NAME "linalg.reject_${name}"
        COMMAND "${CMAKE_COMMAND}"
            "-DPROGRAM=$<TARGET_FILE:krona_test_linalg>"
            "-DCASE=${name}" "-DEXPECT=${diagnostic}"
            "-DRANKS=${ranks}"
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
            "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
            "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake")
    set_tests_properties("linalg.reject_${name}" PROPERTIES
        LABELS "linalg;mpi;negative"
        PROCESSORS "${ranks}"
        TIMEOUT 60
        RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()