#pragma once

#include <krona/execution/types.hpp>
#include <krona/io/vtu/vtu.hpp>

#include <array>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

namespace krona::apps::flow_steady {

/// @brief Copies owned values into contiguous storage for the VTU writer.
template<class Component>
auto copy_owned(Component component, std::size_t count) {
    const auto owned = Kokkos::subview(component, std::pair{std::size_t{0}, count});
    const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, owned);
    std::vector<typename Component::non_const_value_type> output(count);
    for (std::size_t i = 0; i < count; ++i) output[i] = values(i);
    return output;
}

/// @brief Exports the owned primitive fields and density.
template<class Primitives, class Conserved>
void write_solution(const mesh::host::MeshPart& host, const std::filesystem::path& directory, const char* name,
                    const Primitives& q, const Conserved& conserved, mpi::CommView comm) {
    if (directory.empty()) return;

    const auto count = conserved.size();
    const auto density = copy_owned(conserved.template component<0>(), count);
    const auto pressure = copy_owned(q.template component<0>(), count);
    const auto u = copy_owned(q.template component<1>(), count);
    const auto v = copy_owned(q.template component<2>(), count);
    const auto w = copy_owned(q.template component<3>(), count);
    const auto temperature = copy_owned(q.template component<4>(), count);

    const std::array fields{io::vtk::field("density", std::span{density}),
        io::vtk::field("pressure", std::span{pressure}), io::vtk::field("temperature", std::span{temperature}),
        io::vtk::vector_field("velocity", std::span{u}, std::span{v}, std::span{w})};
    io::vtk::write_vtu(host, directory, name, comm, {{}, fields, {}});
}

} // namespace krona::apps::flow_steady
