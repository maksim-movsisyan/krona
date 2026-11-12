add_executable(krona_mesh_partitioner ../mesh_partitioner.cpp cli.cpp driver.cpp)
target_link_libraries(krona_mesh_partitioner PRIVATE
    krona::cgns_io krona::mesh_partition krona::mesh_io
    krona::boundary_config_io krona::toml_config krona::warnings) 
