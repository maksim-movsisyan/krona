#include "common/test_common_host_only.hpp"

#include <krona/mesh/generation/config.hpp>
#include <krona/mesh/generation/grading.hpp>
#include <krona/mesh/generation/hex.hpp>
#include <krona/mesh/generation/topology.hpp>
#include <krona/mesh/generation/writer.hpp>
#include <krona/mpi/collectives.hpp>

#include <cgnslib.h>
#include <pcgnslib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace gen = krona::mesh::generation;
namespace mpi = krona::mpi;
using krona::test::host::Case;
using krona::test::host::check;
using krona::test::host::equal;
using krona::test::host::near;

// Small fixtures make expected node counts and interface mappings explicit.

/// @brief Creates a unit cube with caller-selected subdivisions.
gen::GeneratorConfig cube(std::array<std::size_t, 3> cells = {2, 3, 2}) {
    gen::GeneratorConfig config;
    for (const auto &corner : gen::hex::corners)
        config.vertices.emplace_back(static_cast<double>(corner[0]), static_cast<double>(corner[1]),
                                     static_cast<double>(corner[2]));
    gen::MacroBlock block;
    block.name = "cube";
    block.vertices = {0, 1, 2, 3, 4, 5, 6, 7};
    block.cells = cells;
    config.blocks.push_back(block);
    return config;
}

/// @brief Creates adjacent blocks with swapped, reversed axes and matching grading.
gen::GeneratorConfig rotated_channel() {
    auto config = cube();
    config.vertices.insert(config.vertices.end(), {{2, 0, 0}, {2, 1, 0}, {2, 1, 1}, {2, 0, 1}});
    config.blocks[0].grading_type[1] = gen::GradingType::GEOMETRIC;
    config.blocks[0].grading[1] = 3;
    gen::MacroBlock right;
    right.name = "rotated";
    right.vertices = {2, 9, 10, 6, 1, 8, 11, 5};
    right.cells = {2, 2, 3};
    right.grading_type[2] = gen::GradingType::GEOMETRIC;
    right.grading[2] = 1.0 / 3.0;
    config.blocks.push_back(right);
    return config;
}

/// @brief Checks unique node coverage and agreement of scalar and batched queries.
void check_queries(const gen::GeneratorConfig &config, const gen::MacroTopology &topo) {
    // Every generated node must be referenced, with contiguous one-based IDs.
    std::set<std::uint64_t> used;
    for (std::uint64_t cell = 0; cell < topo.total_cells(); ++cell) {
        const auto nodes = topo.get_cell_nodes(cell);
        std::size_t block = 0, i = 0, j = 0, k = 0;
        topo.get_cell_block_and_ijk(cell, block, i, j, k);
        check(nodes == topo.get_block_cell_nodes(block, i, j, k), "Cell decoding disagrees");
        used.insert(nodes.begin(), nodes.end());
    }
    check(used.size() == topo.total_nodes(), "Some global nodes are missing or duplicated");
    check(*used.begin() == 1 && *used.rbegin() == topo.total_nodes(), "Node IDs are not contiguous");

    // Seven-node batches deliberately cross edge, face and block-interior ranges.
    for (std::uint64_t first = 1; first <= topo.total_nodes();) {
        const auto count =
            static_cast<std::size_t>(std::min<std::uint64_t>(7, topo.total_nodes() - first + 1));
        std::vector<double> x(count), y(count), z(count);
        topo.evaluate_node_coords(first, count, config, x.data(), y.data(), z.data());
        for (std::size_t n = 0; n < count; ++n) {
            const auto point = topo.evaluate_node_coord(first + n, config);
            near(x[n], point.x, 2e-13, "Generator coordinate or distribution mismatch");
            near(y[n], point.y, 2e-13, "Generator coordinate or distribution mismatch");
            near(z[n], point.z, 2e-13, "Generator coordinate or distribution mismatch");
        }
        first += count;
    }
    topo.evaluate_node_coords(0, 0, config, nullptr, nullptr, nullptr);
}

/// @brief Checks collapsed dimensions, rotated interfaces and repeated builds.
void test_topology(mpi::CommView comm) {
    gen::MacroTopology topo;
    // Reuse the same topology with axes containing one cell and with full interiors.
    constexpr std::array<std::array<std::size_t, 3>, 3> cell_counts{{
        {1, 1, 1},
        {1, 3, 2},
        {2, 3, 2},
    }};
    for (const auto &cells : cell_counts) {
        const auto config = cube(cells);
        topo.build(config, comm);
        check(topo.total_nodes() == (cells[0] + 1) * (cells[1] + 1) * (cells[2] + 1),
              "Cube node count is incorrect");
        check_queries(config, topo);
    }

    auto config = rotated_channel();
    config.boundaries = {{"patch_automatic", {{{0, 1, 2, 3}}}}};
    topo.build(config, comm);
    equal(topo.total_nodes(), std::uint64_t{60}, "Shared interface nodes are counted once");
    equal(topo.total_cells(), std::uint64_t{24}, "Both channel blocks contribute their cells");
    check(topo.boundary_patches().size() == 2 && topo.boundary_patches()[1].name == "patch_automatic_1",
          "Automatic patch name collides");
    // Right-block zeta is reversed relative to left-block eta.
    for (std::size_t j = 0; j <= 3; ++j)
        for (std::size_t k = 0; k <= 2; ++k)
            check(topo.get_node_id(0, 2, j, k) == topo.get_node_id(1, 0, k, 3 - j),
                  "Rotated interface has different node IDs");
    check_queries(config, topo);
}

/// @brief Checks spacing formulas, reversal and a large-vector norm.
void test_grading(mpi::CommView) {
    const auto geometric = gen::Grading::compute_distribution(4, gen::GradingType::GEOMETRIC, 8);
    near(geometric[1], 1.0 / 15.0, 2e-13, "Generator coordinate or distribution mismatch");
    near(geometric[2], 3.0 / 15.0, 2e-13, "Generator coordinate or distribution mismatch");
    near(geometric[3], 7.0 / 15.0, 2e-13, "Generator coordinate or distribution mismatch");
    for (const auto type :
         {gen::GradingType::UNIFORM, gen::GradingType::GEOMETRIC, gen::GradingType::TANH_START,
          gen::GradingType::TANH_END, gen::GradingType::TANH_BOTH}) {
        const auto values = gen::Grading::compute_distribution(9, type, 2);
        check(values.front() == 0 && values.back() == 1, "Grading endpoints are not exact");
        check(std::adjacent_find(values.begin(), values.end(), std::greater_equal<double>{}) == values.end(),
              "Grading is not strictly increasing");
    }
    // Start/end clustering must represent the same spacing after axis reversal.
    const auto start = gen::Grading::compute_distribution(9, gen::GradingType::TANH_START, 2);
    const auto end = gen::Grading::compute_distribution(9, gen::GradingType::TANH_END, 2);
    for (std::size_t i = 0; i < start.size(); ++i)
        near(start[i], 1 - end[start.size() - 1 - i], 2e-13, "Generator coordinate or distribution mismatch");
    check(std::isfinite(gen::Vec3{1e300, 1e300, 1e300}.norm()), "Vector norm overflowed unnecessarily");
}

/// @brief Checks documented examples and confirms that only rank zero reads files.
void test_config(mpi::CommView comm) {
    const auto data = std::string{KRONA_GENERATOR_TEST_DATA};
    // An inaccessible path on nonmasters verifies that they never open the input.
    const auto config =
        gen::parse_generator_config(comm.rank() == 0 ? data + "/cube.toml" : "/missing/nonmaster.toml", comm);
    check(config.blocks.size() == 1 && config.boundaries.size() == 6, "Cube decoding changed");
    gen::MacroTopology topo;
    topo.build(config, comm);
    equal(topo.total_cells(), std::uint64_t{480}, "Documented cube cell count");
    equal(topo.total_nodes(), std::uint64_t{693}, "Documented cube node count");
    const auto channel = gen::parse_generator_config(data + "/channel.toml", comm);
    topo.build(channel, comm);
    equal(topo.total_cells(), std::uint64_t{1296}, "Documented channel cell count");
    equal(topo.total_nodes(), std::uint64_t{1729}, "Documented channel node count");
}

/// @brief Reports a failed CGNS test operation through MPI.
void cgns(int status) {
    if (status != CG_OK)
        mpi::fatal(cg_get_error());
}

/// @brief Reads written coordinates, connectivity, patch coverage and BC metadata.
void verify_file(const std::string &path, const gen::GeneratorConfig &config,
                 const gen::MacroTopology &topo) {
    int file = 0;
    cgns(cgp_open(path.c_str(), CG_MODE_READ, &file));
    char name[33]{};
    cgsize_t sizes[3]{};
    cgns(cg_zone_read(file, 1, 1, name, sizes));
    check(static_cast<std::uint64_t>(sizes[0]) == topo.total_nodes() &&
              static_cast<std::uint64_t>(sizes[1]) == topo.total_cells(),
          "CGNS zone sizes changed");
    const auto count = static_cast<std::size_t>(sizes[0]);
    std::vector<double> x(count), y(count), z(count);
    const cgsize_t first = 1, last = sizes[0];
    cgns(cg_coord_read(file, 1, 1, "CoordinateX", CGNS_ENUMV(RealDouble), &first, &last, x.data()));
    cgns(cg_coord_read(file, 1, 1, "CoordinateY", CGNS_ENUMV(RealDouble), &first, &last, y.data()));
    cgns(cg_coord_read(file, 1, 1, "CoordinateZ", CGNS_ENUMV(RealDouble), &first, &last, z.data()));
    for (std::size_t i = 0; i < count; ++i) {
        const auto point = topo.evaluate_node_coord(i + 1, config);
        near(x[i], point.x, 2e-13, "Generator coordinate or distribution mismatch");
        near(y[i], point.y, 2e-13, "Generator coordinate or distribution mismatch");
        near(z[i], point.z, 2e-13, "Generator coordinate or distribution mismatch");
    }
    const gen::Vec3 centre{
        (*std::min_element(x.begin(), x.end()) + *std::max_element(x.begin(), x.end())) / 2,
        (*std::min_element(y.begin(), y.end()) + *std::max_element(y.begin(), y.end())) / 2,
        (*std::min_element(z.begin(), z.end()) + *std::max_element(z.begin(), z.end())) / 2};
    const auto point = [&](cgsize_t id) {
        check(id > 0 && id <= sizes[0], "CGNS connectivity references a missing node");
        const auto i = static_cast<std::size_t>(id - 1);
        return gen::Vec3{x[i], y[i], z[i]};
    };

    // Cell-face incidence distinguishes exterior quads from internal interfaces.
    using Quad = std::array<cgsize_t, 4>;
    std::map<Quad, unsigned> face_counts;
    int sections = 0;
    cgns(cg_nsections(file, 1, 1, &sections));
    check(sections == static_cast<int>(topo.boundary_patches().size() + 1), "CGNS section count changed");
    std::set<Quad> written;
    std::vector<std::array<cgsize_t, 2>> patch_ranges;
    for (int section = 1; section <= sections; ++section) {
        CGNS_ENUMT(ElementType_t) type;
        cgsize_t begin = 0, end = 0;
        int boundary = 0, parents = 0;
        cgns(cg_section_read(file, 1, 1, section, name, &type, &begin, &end, &boundary, &parents));
        const std::size_t width = section == 1 ? 8 : 4;
        std::vector<cgsize_t> nodes(static_cast<std::size_t>(end - begin + 1) * width);
        cgns(cg_elements_read(file, 1, 1, section, nodes.data(), nullptr));
        if (section == 1) {
            check(type == CGNS_ENUMV(HEXA_8) && begin == 1 && end == sizes[1], "Invalid volume section");
            for (std::uint64_t cell = 0; cell < topo.total_cells(); ++cell) {
                const auto expected = topo.get_cell_nodes(cell);
                for (std::size_t v = 0; v < 8; ++v)
                    check(static_cast<std::uint64_t>(nodes[cell * 8 + v]) == expected[v],
                          "HEXA_8 ordering changed");
                for (const auto &face : gen::hex::outward_faces) {
                    Quad key{};
                    for (std::size_t v = 0; v < 4; ++v)
                        key[v] = nodes[cell * 8 + face[v]];
                    std::sort(key.begin(), key.end());
                    ++face_counts[key];
                }
            }
        } else {
            check(type == CGNS_ENUMV(QUAD_4), "Invalid boundary section type");
            check(name == topo.boundary_patches()[static_cast<std::size_t>(section - 2)].name,
                  "Patch section names changed");
            patch_ranges.push_back({begin, end});
            for (std::size_t q = 0; q < nodes.size(); q += 4) {
                Quad key{nodes[q], nodes[q + 1], nodes[q + 2], nodes[q + 3]};
                const auto a = point(key[0]), b = point(key[1]), c = point(key[2]), d = point(key[3]);
                // Fixtures fill a rectangular domain, so its centre lies behind every outer face.
                const auto normal = (b - a).cross(d - a);
                check(normal.dot((a + b + c + d) / 4 - centre) > 0, "A boundary quad points inward");
                std::sort(key.begin(), key.end());
                check(face_counts[key] == 1, "A boundary quad is absent or internal");
                check(written.insert(key).second, "A boundary quad was emitted twice");
            }
        }
    }
    const auto exterior = std::count_if(face_counts.begin(), face_counts.end(),
                                        [](const auto &face) { return face.second == 1; });
    check(written.size() == static_cast<std::size_t>(exterior), "Some exterior faces were not written");
    // Each patch BC must reference exactly the element range of its QUAD_4 section.
    int bocos = 0;
    cgns(cg_nbocos(file, 1, 1, &bocos));
    check(bocos == static_cast<int>(patch_ranges.size()), "BC count differs from patch count");
    for (int bc = 1; bc <= bocos; ++bc) {
        CGNS_ENUMT(BCType_t) type;
        CGNS_ENUMT(PointSetType_t) point_type;
        CGNS_ENUMT(DataType_t) normal_type;
        CGNS_ENUMT(GridLocation_t) location;
        cgsize_t points = 0, normal_size = 0;
        int normal_index[3]{}, datasets = 0;
        cgns(cg_boco_info(file, 1, 1, bc, name, &type, &point_type, &points, normal_index, &normal_size,
                          &normal_type, &datasets));
        cgns(cg_boco_gridlocation_read(file, 1, 1, bc, &location));
        std::array<cgsize_t, 2> range{};
        check(points == 2 && point_type == CGNS_ENUMV(PointRange), "BC is not a point range");
        cgns(cg_boco_read(file, 1, 1, bc, range.data(), nullptr));
        check(location == CGNS_ENUMV(FaceCenter) && type == CGNS_ENUMV(BCTypeUserDefined),
              "BC metadata changed");
        check(range == patch_ranges[static_cast<std::size_t>(bc - 1)], "BC range differs from its section");
        check(name == topo.boundary_patches()[static_cast<std::size_t>(bc - 1)].name, "BC name changed");
    }
    cgns(cgp_close(file));
}

/// @brief Writes and reads a fixture collectively, then removes its output.
void write_and_verify(gen::GeneratorConfig config, mpi::CommView comm) {
    const std::string path = "generator-test.cgns";
    // Reverse input winding; output must still point outward.
    config.boundaries = {{"inlet", {{{0, 3, 7, 4}}}}};
    gen::MacroTopology topo;
    topo.build(config, comm);
    gen::write(path, config, topo, comm);
    mpi::barrier(comm);
    verify_file(path, config, topo);
    mpi::barrier(comm);
    if (comm.rank() == 0)
        std::remove(path.c_str());
}

/// @brief Checks a rotated multiblock CGNS round trip, also with tiny write buffers.
void test_write(mpi::CommView comm) {
    write_and_verify(rotated_channel(), comm);
}

/// @brief Writes one volume cell and one-quad patches across several ranks.
void test_write_empty(mpi::CommView comm) {
    // Four ranks leave some volume and inlet intervals empty; all must participate.
    write_and_verify(cube({1, 1, 1}), comm);
}

/// @brief Builds a minimal valid TOML document for parser rejection cases.
std::string input_text() {
    return "[mesh]\nscale = 1.0\nvertices = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],"
           "[0,0,1],[1,0,1],[1,1,1],[0,1,1]]\n"
           "[[blocks]]\nvertices = [0,1,2,3,4,5,6,7]\ncells = [1,1,1]\n";
}

/// @brief Replaces one known TOML fragment in a test input.
void replace(std::string &text, std::string_view from, std::string_view to) {
    const auto at = text.find(from);
    check(at != std::string::npos, "Missing test input fragment");
    text.replace(at, from.size(), to);
}

// Each rejection case changes one valid input and must stop in production validation.

/// @brief Requires topology construction to reject the supplied configuration.
void expect_rejected(const gen::GeneratorConfig &config, mpi::CommView comm) {
    gen::MacroTopology topo;
    topo.build(config, comm);
    check(false, "Invalid generator input was unexpectedly accepted");
}

/// @brief Writes malformed TOML on rank zero and requires collective parsing to fail.
void expect_parse_rejected(std::string_view text, mpi::CommView comm) {
    const std::string path = "invalid.toml";
    if (comm.rank() == 0) {
        std::ofstream output(path);
        output << text;
        check(static_cast<bool>(output), "Cannot write rejection input");
    }
    mpi::barrier(comm);
    (void)gen::parse_generator_config(path, comm);
    check(false, "Invalid TOML was unexpectedly accepted");
}

/// @brief Rejects a block with no cells along one axis.
void reject_zero_cells(mpi::CommView comm) {
    auto config = cube();
    config.blocks[0].cells[0] = 0;
    expect_rejected(config, comm);
}

/// @brief Rejects a block vertex ID outside the shared vertex list.
void reject_bad_vertex(mpi::CommView comm) {
    auto config = cube();
    config.blocks[0].vertices[0] = 8;
    expect_rejected(config, comm);
}

/// @brief Rejects macro vertices that do not belong to any block.
void reject_unused_vertex(mpi::CommView comm) {
    auto config = cube();
    config.vertices.emplace_back(3, 3, 3);
    expect_rejected(config, comm);
}

/// @brief Rejects two blocks formed from the same eight macro vertices.
void reject_duplicate_block(mpi::CommView comm) {
    auto config = cube();
    config.blocks.push_back(config.blocks.front());
    expect_rejected(config, comm);
}

/// @brief Rejects invalid HEXA_8 winding through the sampled Jacobian check.
void reject_inverted(mpi::CommView comm) {
    auto config = cube();
    std::swap(config.blocks[0].vertices[0], config.blocks[0].vertices[1]);
    expect_rejected(config, comm);
}

/// @brief Rejects nonfinite coordinates before topology construction.
void reject_nonfinite(mpi::CommView comm) {
    auto config = cube();
    config.vertices[0].x = std::numeric_limits<double>::infinity();
    expect_rejected(config, comm);
}

/// @brief Rejects invalid grading enum values supplied by C++ callers.
void reject_grading_type(mpi::CommView comm) {
    auto config = cube();
    config.blocks[0].grading_type[0] = static_cast<gen::GradingType>(255);
    expect_rejected(config, comm);
}

/// @brief Rejects clustering that rounds adjacent nodes to the same position.
void reject_collapsed(mpi::CommView comm) {
    auto config = cube();
    config.blocks[0].grading_type[1] = gen::GradingType::TANH_BOTH;
    config.blocks[0].grading[1] = 1000;
    expect_rejected(config, comm);
}

/// @brief Rejects mismatched counts on a shared rotated interface.
void reject_shared_counts(mpi::CommView comm) {
    auto config = rotated_channel();
    ++config.blocks[1].cells[2];
    expect_rejected(config, comm);
}

/// @brief Rejects spacing that disagrees after reversing the shared axis.
void reject_shared_grading(mpi::CommView comm) {
    auto config = rotated_channel();
    // This direction needs reciprocal grading; uniform spacing does not match.
    config.blocks[1].grading[2] = 1;
    expect_rejected(config, comm);
}

/// @brief Rejects an internal block interface assigned to a boundary patch.
void reject_internal_boundary(mpi::CommView comm) {
    auto config = rotated_channel();
    config.boundaries = {{"bad", {{{1, 2, 6, 5}}}}};
    expect_rejected(config, comm);
}

/// @brief Rejects repeated assignment of the same exterior macro face.
void reject_duplicate_boundary(mpi::CommView comm) {
    auto config = cube();
    config.boundaries = {{"bad", {{{0, 1, 2, 3}}, {{0, 1, 2, 3}}}}};
    expect_rejected(config, comm);
}

/// @brief Rejects a face whose vertices cross instead of following its perimeter.
void reject_crossed_boundary(mpi::CommView comm) {
    auto config = cube();
    config.boundaries = {{"bad", {{{0, 2, 3, 1}}}}};
    expect_rejected(config, comm);
}

/// @brief Rejects malformed TOML syntax.
void reject_parse_syntax(mpi::CommView comm) {
    std::string text = "[mesh";
    expect_parse_rejected(text, comm);
}

/// @brief Rejects misspelled top-level configuration keys.
void reject_parse_unknown(mpi::CommView comm) {
    auto text = "unknown = 1\n" + input_text();
    expect_parse_rejected(text, comm);
}

/// @brief Rejects floating-point cell counts even when their values are integral.
void reject_parse_integer(mpi::CommView comm) {
    auto text = input_text();
    replace(text, "cells = [1,1,1]", "cells = [1.0,1,1]");
    expect_parse_rejected(text, comm);
}

/// @brief Rejects an unconvertible scale instead of silently using 1.0.
void reject_parse_scale(mpi::CommView comm) {
    auto text = input_text();
    replace(text, "scale = 1.0", "scale = 9223372036854775807");
    expect_parse_rejected(text, comm);
}

/// @brief Rejects NaN scale values.
void reject_parse_nonfinite(mpi::CommView comm) {
    auto text = input_text();
    replace(text, "scale = 1.0", "scale = nan");
    expect_parse_rejected(text, comm);
}

/// @brief Rejects a scalar where an optional grading array is expected.
void reject_parse_array(mpi::CommView comm) {
    auto text = input_text() + "grading = 1\n";
    expect_parse_rejected(text, comm);
}

/// @brief Rejects incomplete three-dimensional vertex coordinates.
void reject_parse_coordinates(mpi::CommView comm) {
    auto text = input_text();
    replace(text, "[0,0,0]", "[0,0]");
    expect_parse_rejected(text, comm);
}

} // namespace

/// @brief Registers generator cases with the common host-only MPI runner.
int main(int argc, char **argv) {
    constexpr std::array cases{
        Case{"topology", test_topology},
        Case{"grading", test_grading},
        Case{"config", test_config},
        Case{"write", test_write},
        Case{"write_empty", test_write_empty},
        Case{"zero_cells", reject_zero_cells},
        Case{"bad_vertex", reject_bad_vertex},
        Case{"unused_vertex", reject_unused_vertex},
        Case{"duplicate_block", reject_duplicate_block},
        Case{"inverted", reject_inverted},
        Case{"nonfinite", reject_nonfinite},
        Case{"grading_type", reject_grading_type},
        Case{"collapsed", reject_collapsed},
        Case{"shared_counts", reject_shared_counts},
        Case{"shared_grading", reject_shared_grading},
        Case{"internal_boundary", reject_internal_boundary},
        Case{"duplicate_boundary", reject_duplicate_boundary},
        Case{"crossed_boundary", reject_crossed_boundary},
        Case{"parse_syntax", reject_parse_syntax},
        Case{"parse_unknown", reject_parse_unknown},
        Case{"parse_integer", reject_parse_integer},
        Case{"parse_scale", reject_parse_scale},
        Case{"parse_nonfinite", reject_parse_nonfinite},
        Case{"parse_array", reject_parse_array},
        Case{"parse_coordinates", reject_parse_coordinates},
    };
    return krona::test::host::run(argc, argv, "mesh.generator", cases);
}
