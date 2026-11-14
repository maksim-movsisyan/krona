add_executable(krona_scalar_transport ../scalar_transport.cpp cli.cpp driver.cpp)
target_link_libraries(krona_scalar_transport PRIVATE
    krona::transport_scalar krona::mesh_io krona::time_derivative
    krona::toml_config krona::warnings)
