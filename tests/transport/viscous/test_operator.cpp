#include "fixture.hpp"
#include "common/line_mesh.hpp"

#include <krona/boundary/physical/manager.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/fields/pack.hpp>
#include <krona/mesh/device/aux_storage.hpp>
#include <krona/mesh/device/mesh_storage.hpp>
#include <krona/mpi/collectives.hpp>
#include <krona/transport/viscous/operator.hpp>

namespace viscous_test {
namespace {

using Space = Exec::memory_space;
using IndexView = Kokkos::View<std::size_t*, Space>;
using Storage = krona::StorageReal;

struct Entries {
    using memory_space = Space;
    IndexView diagonal, owner_entries, neighbour_entries;
};

/// @brief Binds rows to known block indices without requiring a Trilinos installation.
Entries entry_map(const Exec& exec, const krona::mesh::host::MeshPart& mesh) {
    const auto n = static_cast<std::size_t>(mesh.n_own);
    const auto f = static_cast<std::size_t>(mesh.n_inner_faces);
    Entries map{IndexView("diagonal", n), IndexView("owner entries", f),
                IndexView("neighbour entries", static_cast<std::size_t>(mesh.n_internal_faces))};
    const auto fill = [&](IndexView view, std::size_t first) {
        Kokkos::parallel_for(Kokkos::RangePolicy<Exec>(exec, 0, view.extent(0)),
            KOKKOS_LAMBDA(std::size_t i) { view(i) = first + i; });
    };
    fill(map.diagonal, 0);
    fill(map.owner_entries, n);
    fill(map.neighbour_entries, n + f);
    return map;
}

void check_rows(const Exec& exec, krona::mpi::CommView comm, int start, int owned) {
    using namespace krona;
    const auto host = test::line_mesh(comm, start, owned, 2);
    const mesh::device::MeshStorage storage(host, {}, exec);
    const auto mesh = storage.view();
    const auto incidence = test::line_incidence(host);
    const mesh::device::AuxConnectivityStorage connectivity(incidence, exec);
    const vt::ViscousPlan<Exec, R> plan(host, mesh, connectivity.view(), comm, exec);
    const vt::ViscousOperator viscous(plan, properties, gas);
    const boundary::BoundaryLayout<Exec, R> layout(host, mesh, exec);
    using Model = ph::Compressible<decltype(gas)>;
    ph::Manager<Model, Exec, R> boundary(layout, Model{gas, {}});
    boundary.set(0, ph::IsothermalWall<R>{300, {0, 0, 0}});
    boundary.set(1, ph::AdiabaticWall<R>{{0, 2, 0}});
    boundary.set(2, ph::Symmetry{});
    boundary.finalize();

    const auto n = static_cast<std::size_t>(host.n_cells);
    fields::FieldStorage<Storage, 5, Exec> q("q", n, fields::FieldLocation::Cell, exec);
    fields::FieldStorage<Storage, 15, Exec> grad("grad", n, fields::FieldLocation::Cell, exec);
    fields::FieldStorage<A, 5, Exec> r("r", n, fields::FieldLocation::Cell, exec);
    const auto raw = q.view().kokkos_view();
    auto q_host = Kokkos::create_mirror(raw);
    for (std::size_t i = 0; i < n; ++i) {
        q_host(i, 0) = 300;
        q_host(i, 1) = 0;
        q_host(i, 2) = static_cast<Storage>(host.cell_gid[i] * 2);
        q_host(i, 3) = 0;
        q_host(i, 4) = static_cast<Storage>(300 + host.cell_gid[i] * 2);
    }
    Kokkos::deep_copy(exec, raw, q_host);
    grad.fill_async(exec, Storage{0});
    r.fill_async(exec, A{0});

    const auto fields_in = fields::pack(q.view());
    const auto gradients = fields::pack(grad.view());
    const auto output = fields::pack(r.view());
    const auto entries = entry_map(exec, host);
    using Blocks = Kokkos::View<A*[5][5], Space>;
    const Blocks blocks("blocks", entries.diagonal.extent(0) + entries.owner_entries.extent(0) +
        entries.neighbour_entries.extent(0));
    Kokkos::deep_copy(exec, blocks, A{0});
    const transport::BlockAssemblyView target{blocks, entries};

    // A shear jump and temperature jump give equal and opposite intercell contributions.
    vt::AssemblyStatus status(exec);
    status.begin_async();
    viscous.add_inner_residual_async(fields_in, gradients, output, status);
    viscous.add_inner_jacobian_async(fields_in, gradients, target, status);
    status.check();
    const auto rank_residual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, r.view().kokkos_view());
    const auto block = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, blocks);
    const auto owned_count = static_cast<std::size_t>(owned);
    for (std::size_t i = 0; i < owned_count; ++i) {
        const bool first = host.cell_gid[i] == 0;
        test::equal(rank_residual(i, 0), A{0}, "zero mass diffusion");
        test::near(static_cast<double>(rank_residual(i, 2)), first ? 4. : -4., tolerance(), "oriented shear residual");
        test::near(static_cast<double>(rank_residual(i, 4)), first ? 10. : -10., tolerance(), "oriented energy residual");
        test::check(block(i, 2, 2) > A{0}, "positive owner shear slope");
    }
    test::check(block(owned_count, 2, 2) < A{0}, "negative neighbour shear slope");

    // Add wall contributions onto the existing target, then check the same row increments.
    status.begin_async();
    viscous.add_boundary_residual_async(fields_in, gradients, output, boundary, status);
    viscous.add_boundary_jacobian_async(fields_in, gradients, target, boundary, status);
    status.check();
    const auto with_walls = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, r.view().kokkos_view());
    const auto with_wall_blocks = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, blocks);
    for (std::size_t i = 0; i < owned_count; ++i) {
        test::near(static_cast<double>(with_walls(i, 4)), host.cell_gid[i] == 0 ? 10. : -10.,
                   tolerance(), "walls at matching velocity and temperature add no energy flux");
        test::check(with_wall_blocks(i, 2, 2) > A{2}, "wall adds to the existing momentum diagonal");
    }
}

} // namespace

void rows(const Exec& exec) {
    check_rows(exec, krona::mpi::CommView::world(), 0, 2);
}

void distributed(const Exec& exec) {
    const auto comm = krona::mpi::CommView::world();
    test::equal(comm.size(), 2, "two ranks for the shared face");
    check_rows(exec, comm, comm.rank(), 1);
}

} // namespace viscous_test
