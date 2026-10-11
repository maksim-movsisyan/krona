#include <krona/io/solver_mesh/hdf5_reader.hpp>
#include <krona/mesh/validate.hpp>

#include "hdf5_detail.hpp"

#include <algorithm>

namespace krona::io::solver_mesh {

/// @brief Restores the exact local snapshot from its saved partition row and validates it once.
mesh::MeshPart import_mesh_hdf5(const std::filesystem::path& path, mpi::CommView comm)
{
    using namespace detail;
    auto access = file_access(comm);
    Handle file{H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, access.get()), H5Fclose};
    auto properties = collective_transfer();

    // Check the schema and build widths before allocating or converting mesh payloads.
    mpi::require(H5Lexists(file.get(), "meta/header", H5P_DEFAULT) > 0,
            "Unsupported KRONA mesh format; regenerate the mesh with the current writer");
    std::array<U64, header_size> header{};
    read(file.get(), properties.get(), "meta/header", header.size(), 0, std::span{header});
    mpi::require(header[0] == magic && header[1] == version,
            "Unsupported KRONA mesh format; regenerate the mesh with the current writer");
    mpi::require(header[2] == static_cast<U64>(comm.size()),
            "Saved mesh requires the original MPI size; a different size needs repartitioning");
    mpi::require((header[3] == 32 || header[3] == 64) && header[3] <= sizeof(LocalIndex) * 8 &&
            (header[4] == 32 || header[4] == 64) && header[4] <= sizeof(GlobalIndex) * 8 &&
            (header[5] == 32 || header[5] == 64) && header[5] == sizeof(GeometryReal) * 8 && header[15] == 64,
            "Mesh file needs matching geometry precision and equally wide or wider index types");
    mpi::require(header[6] < static_cast<U64>(std::numeric_limits<mesh::PatchId>::max()),
            "Too many boundary patches in the mesh file");

    std::vector<int> part2rank(static_cast<std::size_t>(comm.size()));
    read(file.get(), properties.get(), "partition/part2rank", part2rank.size(), 0,
         std::span{part2rank});
    const auto rank2part = inverse_mapping(part2rank, comm.size());
    const int part = rank2part[static_cast<std::size_t>(comm.rank())];
    std::vector<U64> directory(static_cast<std::size_t>(comm.size()) * row_size);
    read(file.get(), properties.get(), "partition/directory", directory.size(), 0,
         std::span{directory});
    const Layout layout{directory, part, comm.size()};
    const auto row = std::span<const U64>{directory}.subspan(static_cast<std::size_t>(part) * row_size, row_size);
    mpi::require(row[owned] <= row[cells] && row[owned_nodes] <= row[nodes] &&
            row[internal_faces] <= row[inner_faces] && row[inner_faces] <= row[faces] && row[grouped] == 1 &&
            row[cell_offsets] == row[cells] + 1 && row[face_offsets] == row[faces] + 1 &&
            row[peer_offsets] == row[peers] + 1 && row[patch_offsets] == header[6] + 1 &&
            row[peers] < static_cast<U64>(comm.size()), "Invalid local partition directory");

    mesh::MeshPart m;
    m.rank = comm.rank();
    m.nprocs = comm.size();
    m.n_own = local_count(row[owned]);
    m.n_cells = local_count(row[cells]);
    m.n_nodes_own = local_count(row[owned_nodes]);
    m.n_nodes = local_count(row[nodes]);
    m.n_faces = local_count(row[faces]);
    m.n_internal_faces = local_count(row[internal_faces]);
    m.n_inner_faces = local_count(row[inner_faces]);
    m.faces_grouped = true;
    m.n_flipped = row[flipped];
    m.n_cells_g = header[7]; m.n_nodes_g = header[8]; m.n_faces_g = header[9];
    m.n_bfaces_g = header[10]; m.n_labeled_bfaces_g = header[11];
    m.n_cut_faces_g = header[12]; m.n_ghost_copies_g = header[13];
    read(file.get(), properties.get(), "meta/bbox_lo", 3, 0, std::span{m.bbox_lo});
    read(file.get(), properties.get(), "meta/bbox_hi", 3, 0, std::span{m.bbox_hi});

    // Shared string metadata is small and replicated; mesh payloads remain distributed.
    std::vector<U64> string_offsets;
    std::vector<std::uint8_t> string_bytes;
    mpi::require(header[6] <= (std::numeric_limits<U64>::max() - 1) / 3, "Patch string count overflow");
    resize(string_offsets, header[6] * 3 + 1);
    resize(string_bytes, header[14]);
    read(file.get(), properties.get(), "patches/string_offsets", string_offsets.size(), 0,
         std::span{string_offsets});
    read(file.get(), properties.get(), "patches/string_bytes", string_bytes.size(), 0,
         std::span{string_bytes});
    mpi::require(string_offsets.front() == 0 && string_offsets.back() == string_bytes.size() &&
            std::is_sorted(string_offsets.begin(), string_offsets.end()), "Invalid patch string offsets");
    m.patches.resize(static_cast<std::size_t>(header[6]));
    std::size_t string = 0;
    for (auto& patch : m.patches) {
        for (auto* text : {&patch.name, &patch.family, &patch.source_type}) {
            const auto begin = string_offsets[string], end = string_offsets[string + 1];
            if (end != begin)
                text->assign(reinterpret_cast<const char*>(string_bytes.data() + begin),
                             static_cast<std::size_t>(end - begin));
            ++string;
        }
    }

    // All arrays use their own saved prefix, including halo lists and patch-face lists.
    visit_arrays(m, [&](const char* name, auto& values, Category category) {
        resize(values, layout.count[category]);
        read(file.get(), properties.get(), name, layout.total[category], layout.start[category],
             std::span{values}, category >= cell_offsets);
    });
    file.close();

    // Validate once after loading, before any solver kernel can index the connectivity.
    mesh::validate_meshpart(m);
    return m;
}

} // namespace krona::io::solver_mesh
