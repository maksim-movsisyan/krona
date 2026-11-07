add_library(krona_mesh_generator_cli STATIC cli.cpp)
target_compile_features(krona_mesh_generator_cli PUBLIC cxx_std_20)
target_link_libraries(krona_mesh_generator_cli PRIVATE krona::warnings)

if(KRONA_BUILD_APPLICATIONS)
    add_executable(krona_mesh_generator ../mesh_generator.cpp driver.cpp)
    target_link_libraries(krona_mesh_generator PRIVATE
        krona_mesh_generator_cli krona::mesh_generation krona::toml_config krona::warnings)
endif()
