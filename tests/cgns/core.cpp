#include "krona/distribution/block.hpp"
#include "krona/mesh/element.hpp"
#include <cstdint>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cmath>
void expect(bool ok)
{
    if (!ok)
        std::abort();
}
void check_reference_faces();
int main()
{
    using Block = krona::distribution::BlockDistribution<std::int64_t>;
    for (std::int64_t n = 0; n < 80; ++n)
        for (int p = 1; p < 90; ++p)
        {
            Block block(n, p);
            std::int64_t previous = 0;
            for (int r = 0; r < p; ++r)
            {
                expect(block.begin(r) == previous);
                auto count = block.local_size(r);
                expect(count == n / p + (r < n % p ? 1 : 0));
                for (auto id = block.begin(r); id < block.end(r); ++id)
                    expect(block.owner(id) == r);
                previous = block.end(r);
            }
            expect(previous == n && block.owner(-1) == -1 && block.owner(n) == -1);
        }
    auto max = std::numeric_limits<std::int64_t>::max();
    for (int p : {1, 2, 3, 2147483647})
    {
        Block block(max, p);
        expect(block.end(p - 1) == max);
        expect(block.owner(max - 1) == p - 1);
        expect(block.owner(0) == 0);
    }
    bool caught = false;
    try
    {
        [[maybe_unused]] krona::distribution::BlockDistribution<std::int8_t> invalid(10, 128);
    }
    catch (...)
    {
        caught = true;
    }
    expect(caught);
    std::array<krona::GlobalIndex, 3> nodes{3, 1, 2};
    auto key = krona::mesh::make_face_key(nodes);
    expect(key.nodes[0] == 1 && key.nodes[1] == 2 && key.nodes[2] == 3 && key.nodes[3] == -1);
    check_reference_faces();
    std::puts("PASS core distributions: zero, empty ranks, narrow indices, INT64_MAX; face keys");
}

using V = std::array<double, 3>;
V sub(V a, V b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
V cross(V a, V b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(V a, V b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
void check_reference_faces()
{
    const std::array<std::array<V, 8>, 4> cells{
        {{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
         {{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}}},
         {{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}}},
         {{{0, 0, 0},
           {1, 0, 0},
           {1, 1, 0},
           {0, 1, 0},
           {0, 0, 1},
           {1, 0, 1},
           {1, 1, 1},
           {0, 1, 1}}}}};
    for (std::size_t type = 0; type < 4; ++type)
    {
        V center{};
        for (int i = 0; i < krona::mesh::nodes_per_type[type]; ++i)
            for (int a = 0; a < 3; ++a)
                center[a] += cells[type][i][a] / krona::mesh::nodes_per_type[type];
        for (int face = 0; face < krona::mesh::faces_per_type[type]; ++face)
        {
            auto f = krona::mesh::face_table[type][face];
            auto a = cells[type][f[0]], b = cells[type][f[1]], c = cells[type][f[2]];
            auto normal = cross(sub(b, a), sub(c, a));
            if (dot(normal, sub(a, center)) <= 0)
                std::abort();
        }
    }
    std::puts("PASS outward reference facet orientation for all four 3D cell types");
}
