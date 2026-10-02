#include "krona/io/cgns/reader.hpp"
#include "krona/mpi/runtime.hpp"
#include "krona/mpi/collectives.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

void verify(bool ok, krona::mpi::CommView comm, const char *message)
{
    if (!ok)
        krona::mpi::fatal(comm.abort_scope(), message);
}
int main(int argc, char **argv)
{
    krona::mpi::Runtime runtime(argc, argv);
    auto owner = krona::mpi::Communicator::duplicate(krona::mpi::CommView::world());
    auto comm = owner.view();
    if (argc < 3)
        return 2;
    const std::string mode = argv[2];
    krona::io::cgns::ReaderOptions options;
    options.batch_elements = argc > 3 ? std::stoull(argv[3]) : 2;
    if (mode == "ignore")
        options.unsupported_boundary = krona::io::cgns::UnsupportedBoundary::ignore;
    auto result = krona::io::cgns::read_parallel(argv[1], comm, options);
    auto &mesh = result.mesh;
    verify(mesh.boundary.size() == static_cast<std::size_t>(
                                       mesh.distribution.boundary_ordinals.local_size(comm.rank())),
           comm, "balanced boundary ownership");
    const bool two = mode.find("2d") != std::string::npos;
    const bool mixed = mode.find("mixed") != std::string::npos;
    const bool no_bc =
        mode.find("no-bc") != std::string::npos || mode == "ignore" || mode == "no-boundary";
    verify(mesh.cell_dimension == (two ? 2 : 3), comm, "cell dimension");
    verify(mesh.distribution.cells.global_size() == 4 && mesh.distribution.nodes.global_size() == 8,
           comm, "global counts");
    verify(mesh.cells.size() ==
               static_cast<std::size_t>(mesh.distribution.cells.local_size(comm.rank())),
           comm, "cell ownership");
    for (std::size_t i = 0; i < mesh.coordinates.size(); ++i)
    {
        auto id = mesh.distribution.nodes.begin(comm.rank()) + static_cast<krona::GlobalIndex>(i);
        verify(mesh.coordinates.x[i] ==
                   static_cast<krona::GeometryReal>(static_cast<double>(id) + 0.125),
               comm, "X by name/type");
        verify(mesh.coordinates.y[i] ==
                   static_cast<krona::GeometryReal>(static_cast<double>(id) * 2 + 0.5),
               comm, "Y by name/type");
        const auto expected =
            two && mode != "2d-embedded" ? 0.0 : static_cast<double>(id) * 3 + 0.25;
        verify(mesh.coordinates.z[i] == static_cast<krona::GeometryReal>(expected), comm,
               "Z/native2D");
    }
    const std::array<krona::mesh::CellType, 4> types{
        krona::mesh::CellType::TET, krona::mesh::CellType::PYRA, krona::mesh::CellType::PRISM,
        krona::mesh::CellType::HEXA};
    for (std::size_t i = 0; i < mesh.cells.size(); ++i)
    {
        auto id = mesh.distribution.cells.begin(comm.rank()) + static_cast<krona::GlobalIndex>(i);
        auto expected =
            mixed ? (two ? (id % 2 ? krona::mesh::CellType::QUAD : krona::mesh::CellType::TRI)
                         : types[static_cast<std::size_t>(id)])
                  : (two ? krona::mesh::CellType::TRI : krona::mesh::CellType::TET);
        verify(mesh.cells.types[i] == expected, comm, "cell type/id ordering");
        auto nodes = mesh.cells.nodes(i);
        verify(nodes.size() == static_cast<std::size_t>(krona::mesh::nodes_per_element(expected)),
               comm, "cell arity");
        for (std::size_t j = 0; j < nodes.size(); ++j)
            verify(nodes[j] == static_cast<krona::GlobalIndex>(j), comm, "cell node conversion");
    }
    verify(mesh.patches.size() == (no_bc ? 0U : 2U), comm, "optional patches");
    if (!no_bc)
        verify(mesh.patches[0].family == "MovingWall" && mesh.patches[0].name == "RangeWall", comm,
               "family/name preservation");
    if (mode == "ignore")
        verify(result.source.ignored_boundary_conditions.size() == 2, comm,
               "explicit ignored BC diagnostics");
    for (std::size_t i = 0; i < mesh.boundary.size(); ++i)
    {
        auto id = mesh.boundary.global_ids[i];
        verify(mesh.boundary.patch_ids[i] ==
                   (no_bc ? krona::mesh::invalid_patch_id : (id < 6 ? 0 : 1)),
               comm, "BC range/list ownership");
        auto nodes = mesh.boundary.elements.nodes(i);
        verify(nodes[0] == 2 && nodes[1] == 1, comm, "boundary orientation preserved");
    }
    auto global_cells = krona::mpi::allreduce<krona::mpi::Reduction::sum>(
        comm, static_cast<std::int64_t>(mesh.cells.size()));
    auto global_nodes = krona::mpi::allreduce<krona::mpi::Reduction::sum>(
        comm, static_cast<std::int64_t>(mesh.coordinates.size()));
    auto global_boundary = krona::mpi::allreduce<krona::mpi::Reduction::sum>(
        comm, static_cast<std::int64_t>(mesh.boundary.size()));
    verify(global_cells == 4 && global_nodes == 8 &&
               global_boundary == (mode == "no-boundary" ? 0 : 4),
           comm, "global conservation");
    // A second import in one process checks close and CGNS global-state reuse.
    auto second = krona::io::cgns::read_parallel(argv[1], comm, options);
    verify(second.mesh.cells.node_ids == mesh.cells.node_ids, comm, "repeat import");
    if (comm.rank() == 0)
        std::printf("PASS %s ranks=%d G=%zu geometry=%zu batch=%zu\n", mode.c_str(), comm.size(),
                    sizeof(krona::GlobalIndex) * 8, sizeof(krona::GeometryReal) * 8,
                    options.batch_elements);
    return 0;
}
