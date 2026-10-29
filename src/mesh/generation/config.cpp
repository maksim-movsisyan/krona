#include "krona/mesh/generation/config.hpp"
#include "krona/mpi/error.hpp"

#include <krona/mpi/collectives.hpp>
#include <krona/mesh/generation/tfi.hpp>
#include <toml++/toml.hpp>

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set> 
#include <span>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace krona::mesh::generation {

namespace {

[[noreturn]] void fail(const std::string& what) {
    mpi::fatal("config: " + what);
}

[[nodiscard]] std::string broadcast_file_content(const std::string& path, mpi::CommView comm) {
    mpi::require(!comm.is_null(), "Config parsing requires a live communicator");
    std::string content;
    if (comm.rank() == 0) {
        std::ifstream file(path, std::ios::binary);
        if (!file) fail("cannot open file '" + path + "' on rank 0");
        std::ostringstream buffer;
        buffer << file.rdbuf();
        if (file.bad() || buffer.bad()) fail("cannot read file '" + path + "'");
        content = buffer.str();
    }

    std::uint64_t size = content.size();
    mpi::broadcast(comm, size);
    mpi::require(size <= std::string{}.max_size(), "Configuration file is too large");
    if (comm.rank() != 0) content.resize(static_cast<std::size_t>(size));
    // MPI 3.1 counts are int; never narrow a file length to int.
    constexpr std::size_t chunk = 1024 * 1024;
    for (std::size_t offset = 0; offset < content.size(); ) {
        const auto count = std::min(chunk, content.size() - offset);
        mpi::broadcast(comm, std::span{content.data() + offset, count});
        offset += count;
    }

    return content;
}

[[nodiscard]] toml::table parse_in_memory_or_die(const std::string& content,
                                                const std::string& source_path) {
    try {
        return toml::parse(content, source_path);
    } catch (const toml::parse_error& err) {
        fail("cannot parse '" + source_path + "': " + std::string(err.description()));
    }
}

void check_allowed_keys(const toml::table& t,
                        const std::initializer_list<const char*> allowed,
                        const std::string& ctx) {
    for (const auto& [k, v] : t) {
        (void)v;
        const std::string_view name = k.str();
        bool known = false;
        for (const char* a : allowed) {
            if (name == std::string_view(a)) {
                known = true;
                break;
            }
        }
        if (!known) {
            fail(ctx + ": unknown key '" + std::string(name) + "'");
        }
    }
}

} // namespace

GeneratorConfig parse_generator_config(const std::string& path, mpi::CommView comm) noexcept {
    return mpi::guarded([&](){
        GeneratorConfig cfg;

        const std::string raw_content = broadcast_file_content(path, comm);
        const toml::table root = parse_in_memory_or_die(raw_content, path);
        
        check_allowed_keys(root, {"mesh", "blocks", "boundaries"}, "'" + path + "'");

        // -------------------------------------------------------------------------
        // 1. [mesh] section (name, scale, vertices)
        // -------------------------------------------------------------------------
        {
            const auto* mesh_node = root["mesh"].as_table();
            if (!mesh_node) {
                fail("missing mandatory [mesh] section in '" + path + "'");
            }
            check_allowed_keys(*mesh_node, {"name", "scale", "vertices"}, "[mesh]");

            if ((*mesh_node)["name"] && !(*mesh_node)["name"].is_string())
                fail("[mesh].name must be a string");
            cfg.name = (*mesh_node)["name"].value_or(std::string("Mesh"));
            if ((*mesh_node)["scale"] && !(*mesh_node)["scale"].is_number())
                fail("[mesh].scale must be numeric");

            if (auto scale = (*mesh_node)["scale"].value<double>()) {
                cfg.scale = *scale;
                if (!std::isfinite(cfg.scale) || cfg.scale <= 0.0) {
                    fail("[mesh].scale must be strictly positive, got " + std::to_string(cfg.scale));
                }
            } else {
                cfg.scale = 1.0;
            }

            const auto* vertices_node = (*mesh_node)["vertices"].as_array();
            if (!vertices_node) {
                fail("missing mandatory 'vertices' array inside [mesh] section");
            }
            if (vertices_node->empty()) {
                fail("[mesh].vertices array must not be empty");
            }

            cfg.vertices.reserve(vertices_node->size());
            for (std::size_t i = 0; i < vertices_node->size(); ++i) {
                const auto* v_arr = (*vertices_node)[i].as_array();
                if (!v_arr || v_arr->size() != 3) {
                    fail("[mesh].vertices [" + std::to_string(i) + "] must be an array of 3 numbers [x, y, z]");
                }

                auto x = (*v_arr)[0].value<double>();
                auto y = (*v_arr)[1].value<double>();
                auto z = (*v_arr)[2].value<double>();

                if (!x || !y || !z) {
                    fail("[mesh].vertices [" + std::to_string(i) + "] coordinates must be numeric");
                }

                cfg.vertices.emplace_back(*x * cfg.scale, *y * cfg.scale, *z * cfg.scale);
            }
        }

        // -------------------------------------------------------------------------
        // 2. [[blocks]] array
        // -------------------------------------------------------------------------
        {
            const auto* blocks_node = root["blocks"].as_array();
            if (!blocks_node || blocks_node->empty()) {
                fail("missing or empty [[blocks]] list in '" + path + "'");
            }

            cfg.blocks.reserve(blocks_node->size());
            for (std::size_t b = 0; b < blocks_node->size(); ++b) {
                const auto* b_tbl = (*blocks_node)[b].as_table();
                if (!b_tbl) {
                    fail("block [" + std::to_string(b) + "] must be a table");
                }
                check_allowed_keys(*b_tbl, {"name", "vertices", "cells", "grading_type", "grading"},
                                "[[blocks]] [" + std::to_string(b) + "]");

                if ((*b_tbl)["name"] && !(*b_tbl)["name"].is_string())
                    fail("block name must be a string");
                for (const auto* key : {"grading_type", "grading"}) {
                    if ((*b_tbl)[key] && !(*b_tbl)[key].is_array())
                        fail(std::string("block '") + key + "' must be an array");
                }
                MacroBlock block;
                block.name = (*b_tbl)["name"].value_or(std::string("block_") + std::to_string(b));

                // Macro-vertices (8 indices conforming to HEXA_8)
                const auto* v_indices = (*b_tbl)["vertices"].as_array();
                if (!v_indices || v_indices->size() != 8) {
                    fail("block '" + block.name + "': 'vertices' must contain exactly 8 indices");
                }
                for (std::size_t i = 0; i < 8; ++i) {
                    auto idx = (*v_indices)[i].value<std::int64_t>();
                    if (!(*v_indices)[i].is_integer() || !idx || *idx < 0 || static_cast<std::uint64_t>(*idx) >= cfg.vertices.size()) {
                        fail("block '" + block.name + "': vertex index [" + std::to_string(i) + "] must be a non-negative integer");
                    }
                    block.vertices[i] = static_cast<std::size_t>(*idx);
                }

                // Cell counts [Nx, Ny, Nz]
                const auto* c_arr = (*b_tbl)["cells"].as_array();
                if (!c_arr || c_arr->size() != 3) {
                    fail("block '" + block.name + "': 'cells' must contain 3 integers [Nx, Ny, Nz]");
                }
                for (std::size_t i = 0; i < 3; ++i) {
                    auto n = (*c_arr)[i].value<std::int64_t>();
                    if (!(*c_arr)[i].is_integer() || !n || *n <= 0 || static_cast<std::uint64_t>(*n) >= std::numeric_limits<std::size_t>::max()) {
                        fail("block '" + block.name + "': cell count along axis " + std::to_string(i) + " must be strictly positive");
                    }
                    block.cells[i] = static_cast<std::size_t>(*n);
                }

                // Grading types (optional, default uniform)
                if (const auto* gt_arr = (*b_tbl)["grading_type"].as_array()) {
                    if (gt_arr->size() != 3) {
                        fail("block '" + block.name + "': 'grading_type' must contain exactly 3 strings");
                    }
                    for (std::size_t i = 0; i < 3; ++i) {
                        auto gt_str = (*gt_arr)[i].value<std::string_view>();
                        if (!gt_str) {
                            fail("block '" + block.name + "': invalid string in 'grading_type' [" + std::to_string(i) + "]");
                        }
                        block.grading_type[i] = parse_grading_type(*gt_str);
                    }
                } else {
                    block.grading_type.fill(GradingType::UNIFORM);
                }

                // Grading strengths (optional, default 1.0)
                if (const auto* g_arr = (*b_tbl)["grading"].as_array()) {
                    if (g_arr->size() != 3) {
                        fail("block '" + block.name + "': 'grading' must contain exactly 3 numbers");
                    }
                    for (std::size_t i = 0; i < 3; ++i) {
                        auto g_val = (*g_arr)[i].value<double>();
                        if (!g_val) {
                            fail("block '" + block.name + "': 'grading' values must be numeric");
                        }
                        block.grading[i] = *g_val;
                    }
                } else {
                    block.grading.fill(1.0);
                }

                cfg.blocks.push_back(std::move(block));
            }
        }

        // -------------------------------------------------------------------------
        // 3. [[boundaries]] array
        // -------------------------------------------------------------------------
        if (root["boundaries"] && !root["boundaries"].is_array())
            fail("'boundaries' must be an array of tables");
        if (const auto* bnd_node = root["boundaries"].as_array()) {
            cfg.boundaries.reserve(bnd_node->size());
            for (std::size_t p = 0; p < bnd_node->size(); ++p) {
                const auto* p_tbl = (*bnd_node)[p].as_table();
                if (!p_tbl) {
                    fail("boundary patch [" + std::to_string(p) + "] must be a table");
                }
                check_allowed_keys(*p_tbl, {"name", "faces"}, "[[boundaries]] [" + std::to_string(p) + "]");

                if ((*p_tbl)["name"] && !(*p_tbl)["name"].is_string())
                    fail("boundary patch name must be a string");
                BoundaryPatch patch;
                patch.name = (*p_tbl)["name"].value_or(std::string("patch_") + std::to_string(p));

                const auto* f_list = (*p_tbl)["faces"].as_array();
                if (!f_list || f_list->empty()) {
                    fail("boundary patch '" + patch.name + "': 'faces' array must not be empty");
                }

                patch.faces.reserve(f_list->size());
                for (std::size_t fi = 0; fi < f_list->size(); ++fi) {
                    const auto* f_arr = (*f_list)[fi].as_array();
                    if (!f_arr || f_arr->size() != 4) {
                        fail("boundary patch '" + patch.name + "' face [" + std::to_string(fi) + "] must specify 4 vertex indices");
                    }

                    MacroFace face;
                    for (std::size_t vi = 0; vi < 4; ++vi) {
                        auto v_idx = (*f_arr)[vi].value<std::int64_t>();
                        if (!(*f_arr)[vi].is_integer() || !v_idx || *v_idx < 0 || static_cast<std::uint64_t>(*v_idx) >= cfg.vertices.size()) {
                            fail("patch '" + patch.name + "' face [" + std::to_string(fi) + "] vertex [" + std::to_string(vi) + "] must be non-negative integer");
                        }
                        face.vertices[vi] = static_cast<std::size_t>(*v_idx);
                    }
                    patch.faces.push_back(face);
                }
                cfg.boundaries.push_back(std::move(patch));
            }
        }

        validate_generator_config(cfg);
        return cfg;
    });
}

void validate_generator_config(const GeneratorConfig& cfg) noexcept {
    mpi::guarded([&](){
        mpi::require(!cfg.vertices.empty() && !cfg.blocks.empty(), "Mesh needs vertices and blocks");
        mpi::require(std::isfinite(cfg.scale) && cfg.scale > 0, "Mesh scale must be finite and positive");
        for (const auto& v : cfg.vertices)
            mpi::require(std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z),
                        "Mesh coordinates must be finite after scaling");
        const std::size_t num_vertices = cfg.vertices.size();
        std::vector<bool> used(num_vertices, false);
        std::set<std::array<std::size_t, 8>> block_vertices;
        for (const auto& block : cfg.blocks) {
            auto unique = block.vertices;
            std::sort(unique.begin(), unique.end());
            mpi::require(std::adjacent_find(unique.begin(), unique.end()) == unique.end(),
                        "A block must reference eight distinct vertices");
            mpi::require(block_vertices.insert(unique).second, "Duplicate blocks reference the same eight vertices");
            std::array<Vec3, 8> corners;
            for (std::size_t i = 0; i < 8; ++i) {
                if (block.vertices[i] >= num_vertices)
                    fail("block '" + block.name + "': vertex index is out of bounds");
                used[block.vertices[i]] = true;
                corners[i] = cfg.vertices[block.vertices[i]];
            }
            // Check the trilinear map at all corners, edge/face centres and the centre.
            // This is a sampling check, not a proof of positivity everywhere.
            for (double u : {0.0, 0.5, 1.0})
                for (double v : {0.0, 0.5, 1.0})
                    for (double w : {0.0, 0.5, 1.0})
                        if (!TFI::positive_jacobian(corners, u, v, w))
                            fail("block '" + block.name + "': inverted or degenerate trilinear map; check HEXA_8 order");
            for (std::size_t d = 0; d < 3; ++d) {
                mpi::require(block.cells[d] > 0 && block.cells[d] < std::numeric_limits<std::size_t>::max(),
                            "Cell counts must be positive and leave room for an endpoint");
                if (!std::isfinite(block.grading[d]) ||
                    (block.grading_type[d] != GradingType::UNIFORM && block.grading[d] <= 0))
                    fail("block '" + block.name + "': grading must be finite and positive for nonuniform distributions");
            }
        }
        mpi::require(std::all_of(used.begin(), used.end(), [](bool v) { return v; }),
                    "Every macro vertex must be used by a block");
        std::set<std::string> names;
        for (const auto& patch : cfg.boundaries) {
            if (patch.name.empty() || patch.name.size() > 32 || patch.name == "." || patch.name == ".." ||
                patch.name.find('/') != std::string::npos || patch.name.find('\0') != std::string::npos ||
                patch.name == "Volume_HEXA" || patch.name == "GridCoordinates" ||
                patch.name == "ZoneBC" || patch.name == "ZoneType" ||
                !names.insert(patch.name).second)
                fail("invalid, duplicate or reserved CGNS boundary name '" + patch.name + "'");
            mpi::require(!patch.faces.empty(), "Boundary patch must contain faces");
            for (const auto& face : patch.faces) {
                auto unique = face.vertices;
                std::sort(unique.begin(), unique.end());
                if (unique.back() >= num_vertices ||
                    std::adjacent_find(unique.begin(), unique.end()) != unique.end())
                    fail("patch '" + patch.name + "': face needs four distinct valid vertex indices");
            }
        }
    });
}

} // namespace krona::mesh::generation
