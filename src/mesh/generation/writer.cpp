#include <krona/mesh/generation/writer.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <krona/mesh/generation/hex.hpp>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <cgnslib.h>
#include <krona/mpi/error.hpp>
#include <pcgnslib.h>

namespace krona::mesh::generation {

namespace {

/// @brief Converts a CGNS status code into the MPI fatal-error policy.
void check_cgns(int status, const char *msg) {
    if (status != CG_OK) {
        mpi::fatal(std::string{"CGNS Error ["} + msg + "]: " + cg_get_error());
    }
}

/// @brief Splits a global interval into balanced zero-based rank intervals.
std::pair<std::uint64_t, std::uint64_t> decompose_1d(std::uint64_t total_count, int rank,
                                                     int nprocs) noexcept {
    if (total_count == 0)
        return {0, 0};
    const std::uint64_t base = total_count / static_cast<std::uint64_t>(nprocs);
    const std::uint64_t rem = total_count % static_cast<std::uint64_t>(nprocs);

    const std::uint64_t start =
        static_cast<std::uint64_t>(rank) * base + std::min(static_cast<std::uint64_t>(rank), rem);
    const std::uint64_t count = base + (static_cast<std::uint64_t>(rank) < rem ? 1 : 0);
    return {start, start + count};
}

// Total application-side data-buffer budget per rank. Override at compile time
// on every rank, e.g. -DKRONA_MESH_WRITE_BUFFER_BYTES=134217728 for 128 MiB.
#ifndef KRONA_MESH_WRITE_BUFFER_BYTES
#ifdef CFD_MESH_WRITE_BUFFER_BYTES
#define KRONA_MESH_WRITE_BUFFER_BYTES CFD_MESH_WRITE_BUFFER_BYTES
#else
#define KRONA_MESH_WRITE_BUFFER_BYTES (64ULL * 1024ULL * 1024ULL)
#endif
#endif
/// @brief Maximum application-side coordinate or connectivity buffer bytes per rank.
constexpr std::uint64_t kWriteBufferBytes = KRONA_MESH_WRITE_BUFFER_BYTES;
static_assert(kWriteBufferBytes >= 8 * sizeof(cgsize_t) && kWriteBufferBytes >= 3 * sizeof(double),
              "Mesh write buffer is too small");

// Every rank executes the same number of collective calls, including ranks
// with no local data or an exhausted final chunk. A null data pointer denotes
// an empty contribution to CGNS; never skip a collective on just one rank.
/// @brief Issues matching chunk calls, including empty rank contributions.
template <class WriteChunk>
void for_each_chunk(std::uint64_t total, std::uint64_t local_count, int nprocs, std::uint64_t capacity,
                    WriteChunk &&write_chunk) {
    const auto ranks = static_cast<std::uint64_t>(nprocs);
    const std::uint64_t max_local = total / ranks + (total % ranks != 0);
    for (std::uint64_t offset = 0; offset < max_local;) {
        const auto local_offset = std::min(offset, local_count);
        const auto count = std::min(capacity, local_count - local_offset);
        write_chunk(local_offset, count);
        offset += std::min(capacity, max_local - offset);
    }
}

/// @brief Identifies the file, base and unstructured zone used by all writes.
struct CgnsZone {
    int file{0};
    int base{0};
    int zone{0};
};

/// @brief Maps a patch interval to one block face and its subdivision.
struct ResolvedPatchFace {
    std::size_t block_idx{0};
    std::size_t local_face_idx{0};
    std::size_t Nu{0};
    std::size_t Nv{0};
    std::uint64_t face_cell_offset{0};
};

/// @brief Stores a patch's resolved faces and total boundary-quad count.
struct ResolvedPatch {
    std::vector<ResolvedPatchFace> faces;
    std::uint64_t quads{0};
};

/// @brief Resolves patch faces once and checks all CGNS element-count limits.
[[nodiscard]] std::vector<ResolvedPatch> resolve_patches(const GeneratorConfig &config,
                                                         const MacroTopology &topo, std::uint64_t limit) {
    // Shared topology has already validated exterior membership and uniqueness.
    std::map<hex::FaceKey, std::pair<std::size_t, std::size_t>> block_faces;
    for (std::size_t block = 0; block < config.blocks.size(); ++block)
        for (std::size_t face = 0; face < hex::faces.size(); ++face)
            block_faces.emplace(hex::face_key(hex::face_vertices(config.blocks[block].vertices, face)),
                                std::pair{block, face});

    std::vector<ResolvedPatch> result;
    result.reserve(topo.boundary_patches().size());
    auto total_elements = topo.total_cells();
    for (const auto &patch : topo.boundary_patches()) {
        ResolvedPatch resolved;
        resolved.faces.reserve(patch.faces.size());
        for (const auto &macro_face : patch.faces) {
            const auto it = block_faces.find(hex::face_key(macro_face.vertices));
            mpi::require(it != block_faces.end(), "Boundary face does not match any block face");
            const auto [block, face] = it->second;
            const auto &source = config.blocks[block];
            mpi::require(hex::same_perimeter(macro_face.vertices, hex::face_vertices(source.vertices, face)),
                         "Boundary vertices must follow the face perimeter");
            const auto &layout = hex::layouts[face];
            const auto Nu = source.cells[layout.u];
            const auto Nv = source.cells[layout.v];
            const auto quads = mpi::checked_product(Nu, Nv, "Boundary size overflows size_t");
            mpi::require(quads <= limit - total_elements, "CGNS element numbering overflows");
            resolved.faces.push_back({block, face, Nu, Nv, resolved.quads});
            resolved.quads += quads;
            total_elements += quads;
        }
        mpi::require(resolved.quads <= limit / 4, "QUAD_4 connectivity exceeds CGNS index range");
        result.push_back(std::move(resolved));
    }
    return result;
}

/// @brief Writes balanced coordinate intervals within the per-rank buffer budget.
void write_coordinates(CgnsZone zone, const GeneratorConfig &config, const MacroTopology &topo,
                       mpi::CommView comm) {
    const auto total_nodes = topo.total_nodes();
    const auto rank = comm.rank();
    const auto nprocs = comm.size();
    int cx_id = 0, cy_id = 0, cz_id = 0;
    check_cgns(
        cgp_coord_write(zone.file, zone.base, zone.zone, CGNS_ENUMV(RealDouble), "CoordinateX", &cx_id),
        "cgp_coord_write(X)");
    check_cgns(
        cgp_coord_write(zone.file, zone.base, zone.zone, CGNS_ENUMV(RealDouble), "CoordinateY", &cy_id),
        "cgp_coord_write(Y)");
    check_cgns(
        cgp_coord_write(zone.file, zone.base, zone.zone, CGNS_ENUMV(RealDouble), "CoordinateZ", &cz_id),
        "cgp_coord_write(Z)");

    const auto [node_start, node_end] = decompose_1d(total_nodes, rank, nprocs);
    const std::uint64_t local_nodes_cnt = node_end - node_start;
    {
        constexpr auto capacity = kWriteBufferBytes / (3 * sizeof(double));
        const auto allocation = static_cast<std::size_t>(std::min(capacity, local_nodes_cnt));
        std::vector<double> px(allocation), py(allocation), pz(allocation);
        for_each_chunk(
            total_nodes, local_nodes_cnt, nprocs, capacity, [&](std::uint64_t offset, std::uint64_t count) {
                const bool has_data = count != 0;
                if (has_data) {
                    topo.evaluate_node_coords(node_start + offset + 1, static_cast<std::size_t>(count),
                                              config, px.data(), py.data(), pz.data());
                }
                const cgsize_t first = has_data ? static_cast<cgsize_t>(node_start + offset + 1) : 1;
                const cgsize_t last = has_data ? static_cast<cgsize_t>(node_start + offset + count) : 1;
                check_cgns(cgp_coord_write_data(zone.file, zone.base, zone.zone, cx_id, &first, &last,
                                                has_data ? px.data() : nullptr),
                           "cgp_coord_write_data(X)");
                check_cgns(cgp_coord_write_data(zone.file, zone.base, zone.zone, cy_id, &first, &last,
                                                has_data ? py.data() : nullptr),
                           "cgp_coord_write_data(Y)");
                check_cgns(cgp_coord_write_data(zone.file, zone.base, zone.zone, cz_id, &first, &last,
                                                has_data ? pz.data() : nullptr),
                           "cgp_coord_write_data(Z)");
            });
    } // Release all coordinate buffers before allocating connectivity.
}

/// @brief Writes HEXA_8 connectivity in block/i/j/k order using bounded chunks.
void write_volume(CgnsZone zone, const GeneratorConfig &config, const MacroTopology &topo,
                  mpi::CommView comm) {
    const auto total_cells = topo.total_cells();
    const auto rank = comm.rank();
    const auto nprocs = comm.size();
    const cgsize_t hex_start = 1;
    const cgsize_t hex_end = static_cast<cgsize_t>(total_cells);

    int hex_sec_id = 0;
    check_cgns(cgp_section_write(zone.file, zone.base, zone.zone, "Volume_HEXA", CGNS_ENUMV(HEXA_8),
                                 hex_start, hex_end, 0, &hex_sec_id),
               "cgp_section_write(HEXA_8)");

    const auto [cell_start, cell_end] = decompose_1d(total_cells, rank, nprocs);
    const std::uint64_t local_cells_cnt = cell_end - cell_start;
    {
        constexpr auto capacity = kWriteBufferBytes / (8 * sizeof(cgsize_t));
        std::vector<cgsize_t> hex_conn(static_cast<std::size_t>(std::min(capacity, local_cells_cnt)) * 8);
        for_each_chunk(
            total_cells, local_cells_cnt, nprocs, capacity, [&](std::uint64_t offset, std::uint64_t count) {
                // Decode once per chunk, then walk cells in block/i/j/k order.
                std::size_t b = 0, i = 0, j = 0, k = 0;
                if (count != 0)
                    topo.get_cell_block_and_ijk(cell_start + offset, b, i, j, k);
                for (std::uint64_t c = 0; c < count; ++c) {
                    const auto nodes = topo.get_block_cell_nodes(b, i, j, k);
                    for (std::size_t n = 0; n < 8; ++n)
                        hex_conn[c * 8 + n] = static_cast<cgsize_t>(nodes[n]);
                    const auto &dims = config.blocks[b].cells;
                    if (++i == dims[0]) {
                        i = 0;
                        if (++j == dims[1]) {
                            j = 0;
                            if (++k == dims[2]) {
                                k = 0;
                                ++b;
                            }
                        }
                    }
                }
                const bool has_data = count != 0;
                const cgsize_t first = has_data ? static_cast<cgsize_t>(cell_start + offset + 1) : 1;
                const cgsize_t last = has_data ? static_cast<cgsize_t>(cell_start + offset + count) : 1;
                check_cgns(cgp_elements_write_data(zone.file, zone.base, zone.zone, hex_sec_id, first, last,
                                                   has_data ? hex_conn.data() : nullptr),
                           "cgp_elements_write_data(HEXA_8)");
            });
    } // Release volume connectivity before writing boundary patches.
}

/// @brief Writes outward QUAD_4 sections and matching FaceCenter BC ranges.
void write_boundaries(CgnsZone zone, const GeneratorConfig &config, const MacroTopology &topo,
                      const std::vector<ResolvedPatch> &resolved, mpi::CommView comm) {
    const auto &boundaries = topo.boundary_patches();
    auto current_global_eid = static_cast<cgsize_t>(topo.total_cells() + 1);
    const auto rank = comm.rank();
    const auto nprocs = comm.size();
    for (std::size_t p = 0; p < boundaries.size(); ++p) {
        const auto &patch = boundaries[p];
        const auto &resolved_faces = resolved[p].faces;
        const auto patch_total_quads = resolved[p].quads;

        if (patch_total_quads == 0)
            continue;

        const cgsize_t patch_start = current_global_eid;
        const cgsize_t patch_end =
            static_cast<cgsize_t>(static_cast<std::uint64_t>(patch_start) + patch_total_quads - 1);
        current_global_eid = patch_end + 1;

        int bnd_sec_id = 0;
        check_cgns(cgp_section_write(zone.file, zone.base, zone.zone, patch.name.c_str(), CGNS_ENUMV(QUAD_4),
                                     patch_start, patch_end, 0, &bnd_sec_id),
                   "cgp_section_write(QUAD_4)");

        const auto [quad_start, quad_end] = decompose_1d(patch_total_quads, rank, nprocs);
        const std::uint64_t local_quads_cnt = quad_end - quad_start;
        {
            constexpr auto capacity = kWriteBufferBytes / (4 * sizeof(cgsize_t));
            std::vector<cgsize_t> quad_conn(static_cast<std::size_t>(std::min(capacity, local_quads_cnt)) *
                                            4);
            for_each_chunk(
                patch_total_quads, local_quads_cnt, nprocs, capacity,
                [&](std::uint64_t offset, std::uint64_t count) {
                    for (std::uint64_t q = 0; q < count; ++q) {
                        const std::uint64_t glob_q = quad_start + offset + q;
                        auto it = std::upper_bound(resolved_faces.begin(), resolved_faces.end(), glob_q,
                                                   [](std::uint64_t val, const ResolvedPatchFace &f) {
                                                       return val < f.face_cell_offset;
                                                   });
                        if (it != resolved_faces.begin())
                            --it;

                        const auto &rf = *it;
                        const std::uint64_t local_sub_q = glob_q - rf.face_cell_offset;
                        const std::size_t u = static_cast<std::size_t>(local_sub_q % rf.Nu);
                        const std::size_t v = static_cast<std::size_t>(local_sub_q / rf.Nu);

                        const auto &blk = config.blocks[rf.block_idx];
                        const auto index = hex::boundary_cell(blk.cells, rf.local_face_idx, u, v);
                        const auto c_nodes =
                            topo.get_block_cell_nodes(rf.block_idx, index[0], index[1], index[2]);
                        for (std::size_t k = 0; k < 4; ++k) {
                            quad_conn[q * 4 + k] =
                                static_cast<cgsize_t>(c_nodes[hex::outward_faces[rf.local_face_idx][k]]);
                        }
                    }
                    const bool has_data = count != 0;
                    const cgsize_t first =
                        has_data ? static_cast<cgsize_t>(static_cast<std::uint64_t>(patch_start) +
                                                         quad_start + offset)
                                 : patch_start;
                    const cgsize_t last =
                        has_data ? static_cast<cgsize_t>(static_cast<std::uint64_t>(patch_start) +
                                                         quad_start + offset + count - 1)
                                 : patch_start;
                    check_cgns(cgp_elements_write_data(zone.file, zone.base, zone.zone, bnd_sec_id, first,
                                                       last, has_data ? quad_conn.data() : nullptr),
                               "cgp_elements_write_data(QUAD_4)");
                });
        }

        int boco_id = 0;
        const cgsize_t range[2] = {patch_start, patch_end};

        check_cgns(cg_boco_write(zone.file, zone.base, zone.zone, patch.name.c_str(),
                                 CGNS_ENUMV(BCTypeUserDefined), CGNS_ENUMV(PointRange), 2, range, &boco_id),
                   "cg_boco_write");

        check_cgns(
            cg_boco_gridlocation_write(zone.file, zone.base, zone.zone, boco_id, CGNS_ENUMV(FaceCenter)),
            "cg_boco_gridlocation_write");
    }
}

} // namespace

/// @brief Writes the full mesh collectively using bounded host buffers.
void write(const std::string &filepath, const GeneratorConfig &config, const MacroTopology &topo,
           mpi::CommView comm) noexcept {
    mpi::guarded([&] {
        mpi::require(!comm.is_null(), "CGNS writing requires a live communicator");
        mpi::require(!filepath.empty() && filepath.find('\0') == std::string::npos,
                     "CGNS output path must be nonempty and contain no NULs");
        static_assert(std::numeric_limits<cgsize_t>::is_signed && sizeof(cgsize_t) == 8,
                      "Generator requires signed 64-bit CGNS indices");
        const auto limit = static_cast<std::uint64_t>(std::numeric_limits<cgsize_t>::max()) - 1;
        mpi::require(topo.total_nodes() > 0 && topo.total_cells() > 0 && topo.total_nodes() <= limit &&
                         topo.total_cells() <= limit / 8,
                     "CGNS node IDs and HEXA_8 connectivity must fit signed 64-bit indices");
        const auto patches = resolve_patches(config, topo, limit);

        // Create shared metadata before entering the collective data writes.
        check_cgns(cgp_mpi_comm(comm.native()), "cgp_mpi_comm");
        check_cgns(cgp_pio_mode(CGP_COLLECTIVE), "cgp_pio_mode(CGP_COLLECTIVE)");
        CgnsZone zone;
        check_cgns(cgp_open(filepath.c_str(), CG_MODE_WRITE, &zone.file), "cgp_open(WRITE)");
        check_cgns(cg_base_write(zone.file, "Base_3D", 3, 3, &zone.base), "cg_base_write");
        const cgsize_t sizes[3] = {static_cast<cgsize_t>(topo.total_nodes()),
                                   static_cast<cgsize_t>(topo.total_cells()), 0};
        check_cgns(cg_zone_write(zone.file, zone.base, "Zone1", sizes, CGNS_ENUMV(Unstructured), &zone.zone),
                   "cg_zone_write");

        // Stage lifetimes release each buffer before allocating the next one.
        write_coordinates(zone, config, topo, comm);
        write_volume(zone, config, topo, comm);
        write_boundaries(zone, config, topo, patches, comm);
        check_cgns(cgp_close(zone.file), "cgp_close");

        if (comm.rank() == 0) {
            std::fputs("Parallel CGNS mesh successfully written to '", stdout);
            std::fputs(filepath.c_str(), stdout);
            std::fputs("'\n", stdout);
        }
    });
}

} // namespace krona::mesh::generation
