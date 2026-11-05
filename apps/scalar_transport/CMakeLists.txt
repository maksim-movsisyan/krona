# Application paths/CLI only; named scalar settings live in krona::transport_scalar_config.
add_library(krona_scalar_transport_app_config STATIC config.cpp cli.cpp)
target_compile_features(krona_scalar_transport_app_config PUBLIC cxx_std_20)
target_link_libraries(krona_scalar_transport_app_config
    PUBLIC krona::transport_scalar_config PRIVATE krona::tomlplusplus krona::warnings)
target_include_directories(krona_scalar_transport_app_config PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if(KRONA_BUILD_APPLICATIONS AND KRONA_ENABLE_TRILINOS)
    add_executable(krona_scalar_transport ../scalar_transport.cpp driver.cpp)
    target_link_libraries(krona_scalar_transport PRIVATE
        krona_scalar_transport_app_config krona::transport_scalar krona::mesh_io krona::warnings)
endif()
