#include <cstddef>
#include <krona/io/solver_mesh/hdf5_writer.hpp>

#include "hdf5_detail.hpp"

#include <numeric>
#include <algorithm>

namespace krona::io::solver_mesh {

/// @brief Writes the complete solver snapshot using one directory exchange and direct SoA I/O.
void export_mesh_hdf5(const mesh::MeshPart& m, const std::filesystem::path& path,
                      mpi::CommView comm, std::span<const int> part2rank)
{
    using namespace detail;
    auto access = file_access(comm);
    mpi::require(!path.empty(), "Solver mesh output path is empty");
    mpi::require(m.rank == comm.rank() && m.nprocs == comm.size() && m.faces_grouped,
            "HDF5 export requires a grouped mesh on the supplied communicator");

    // A partition SDK is not required by the file format: only its rank permutation is used.
    std::vector<int> identity;
    if (part2rank.empty()) {
        identity.resize(static_cast<std::size_t>(comm.size()));
        std::iota(identity.begin(), identity.end(), 0);
        part2rank = identity;
    }
    const auto rank2part = inverse_mapping(part2rank, comm.size());
    const int part = rank2part[static_cast<std::size_t>(comm.rank())];

    // Record all payload lengths in one compact row, including full local CSR offsets.
    std::array<U64, row_size> row{};
    row[cells] = static_cast<U64>(m.n_cells);
    row[cell_indices] = m.cell_nodes.size();
    row[nodes] = static_cast<U64>(m.n_nodes);
    row[faces] = static_cast<U64>(m.n_faces);
    row[face_indices] = m.face_nodes.size();
    row[peers] = m.nb_ranks.size();
    row[send_indices] = m.send_owned_local.size();
    row[recv_indices] = m.recv_ghost_local.size();
    row[patch_indices] = m.patch_faces.size();
    row[cell_offsets] = row[cells] + 1;
    row[face_offsets] = row[faces] + 1;
    row[peer_offsets] = row[peers] + 1;
    row[patch_offsets] = m.patches.size() + 1;
    row[owned] = static_cast<U64>(m.n_own);
    row[owned_nodes] = static_cast<U64>(m.n_nodes_own);
    row[internal_faces] = static_cast<U64>(m.n_internal_faces);
    row[inner_faces] = static_cast<U64>(m.n_inner_faces);
    row[grouped] = 1;
    row[flipped] = m.n_flipped;

    std::vector<U64> rank_rows(static_cast<std::size_t>(comm.size()) * row_size);
    mpi::check(MPI_Allgather(row.data(), static_cast<int>(row_size), MPI_UINT64_T,
                            rank_rows.data(), static_cast<int>(row_size), MPI_UINT64_T,
                            comm.native()), "MPI_Allgather(mesh file directory)");
    std::vector<U64> directory(rank_rows.size());
    for (int p = 0; p < comm.size(); ++p)
        std::copy_n(rank_rows.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(part2rank[static_cast<std::size_t>(p)]) * row_size),
                    row_size, directory.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(p) * row_size));
    const Layout layout{directory, part, comm.size()};
    rank_rows.clear();
    rank_rows.shrink_to_fit();

    // Encode all patch strings without a fixed-length truncation or HDF5 variable-length types.
    std::vector<U64> string_offsets{0};
    std::vector<std::uint8_t> string_bytes;
    for (const auto& patch : m.patches) {
        for (const auto* text : {&patch.name, &patch.family, &patch.source_type}) {
            string_bytes.insert(string_bytes.end(), text->begin(), text->end());
            string_offsets.push_back(string_bytes.size());
        }
    }
    const std::array<U64, header_size> header{
        magic, version, static_cast<U64>(comm.size()), sizeof(LocalIndex) * 8,
        sizeof(GlobalIndex) * 8, sizeof(GeometryReal) * 8, m.patches.size(),
        m.n_cells_g, m.n_nodes_g, m.n_faces_g, m.n_bfaces_g, m.n_labeled_bfaces_g,
        m.n_cut_faces_g, m.n_ghost_copies_g, string_bytes.size(), 64
    };

    Handle file{H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, access.get()), H5Fclose};
    auto properties = collective_transfer();
    for (const char* group : {"meta", "partition", "cells", "nodes", "faces", "halo", "patches"}) {
        Handle created{H5Gcreate2(file.get(), group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose};
    }

    // Shared metadata is written once, while every rank participates in the collective calls.
    write_shared(file.get(), properties.get(), "meta/header", std::span<const U64>{header}, comm.rank());
    write_shared(file.get(), properties.get(), "meta/bbox_lo", std::span<const GeometryReal>{m.bbox_lo}, comm.rank());
    write_shared(file.get(), properties.get(), "meta/bbox_hi", std::span<const GeometryReal>{m.bbox_hi}, comm.rank());
    write_shared(file.get(), properties.get(), "partition/part2rank", part2rank, comm.rank());
    write_shared(file.get(), properties.get(), "partition/directory", std::span<const U64>{directory}, comm.rank());
    write_shared(file.get(), properties.get(), "patches/string_offsets", std::span<const U64>{string_offsets}, comm.rank());
    write_shared(file.get(), properties.get(), "patches/string_bytes", std::span<const std::uint8_t>{string_bytes}, comm.rank());

    // Write mesh arrays directly from their final storage, with no AoS conversion or mesh gathering.
    visit_arrays(m, [&](const char* name, const auto& values, Category category) {
        using T = typename std::remove_cvref_t<decltype(values)>::value_type;
        mpi::require(values.size() == layout.count[category], std::string{"Invalid export array: "} + name);
        const bool offsets = category >= cell_offsets;
        write(file.get(), properties.get(), name, layout.total[category], layout.start[category],
              std::span<const T>{values}, offsets);
    });
    // Report close/flush errors before announcing a successful export.
    file.close();
}

} // namespace krona::io::solver_mesh
