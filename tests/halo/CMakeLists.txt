add_executable(krona_test_halo test_halo.cpp)
target_link_libraries(krona_test_halo PRIVATE krona_test_support krona::halo)

if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Halo tests require an MPI launcher")
endif()

foreach(case IN ITEMS loopback roundtrip multi_peer one_way isolated integer selected)
    set(ranks 2)
    if(case STREQUAL "loopback")
        set(ranks 1)
    elseif(case STREQUAL "multi_peer" OR case STREQUAL "isolated")
        set(ranks 3)
    endif()
    add_test(NAME "halo.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:krona_test_halo>
                ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("halo.${case}" PROPERTIES
        LABELS "halo;mpi"
        PROCESSORS "${ranks}"
        TIMEOUT 120
        RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()