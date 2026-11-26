add_executable(krona_flow_steady ../flow_steady.cpp cli.cpp driver.cpp)
target_link_libraries(krona_flow_steady PRIVATE
    krona::solver_flow krona::mesh_io krona::toml_config krona::warnings)
