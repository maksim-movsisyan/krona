add_executable(krona_test_math test_math.cpp)
target_link_libraries(krona_test_math PRIVATE krona_test_support krona::math)

foreach(case IN ITEMS vector matrix tensor inverse inverse_failure precision scale kernel)
    add_test(NAME "math.${case}"
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 1
            ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:krona_test_math>" ${MPIEXEC_POSTFLAGS} "${case}")
    set_tests_properties("math.${case}" PROPERTIES
        LABELS "math;unit" TIMEOUT 120 RUN_SERIAL TRUE
        ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endforeach()
 
# The public math headers require only Kokkos, with no KRONA runtime/config/MPI dependency.
add_executable(krona_test_host_math test_host_math.cpp)
target_link_libraries(krona_test_host_math PRIVATE krona::math)
add_test(NAME math.host_without_runtime COMMAND "$<TARGET_FILE:krona_test_host_math>")
set_tests_properties(math.host_without_runtime PROPERTIES LABELS "math;unit" TIMEOUT 30)
