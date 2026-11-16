add_executable(krona_euler_steady ../euler_steady.cpp cli.cpp config.cpp driver.cpp)
target_link_libraries(krona_euler_steady PRIVATE
    krona::transport_euler krona::gradient krona::mesh_io krona::halo krona::toml_config krona::warnings)
