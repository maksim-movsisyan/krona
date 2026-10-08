#include "common/test_support.hpp"

#include <krona/fields/fields_manager.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace {

using namespace krona::test;
using krona::mesh::MeshPart;
using namespace krona::mesh::compute;

int count(int sender, int receiver, std::string_view mode)
{
    if (sender == receiver) return 0;
    if (mode == "isolated" && (sender == 2 || receiver == 2)) return 0;
    if (mode == "one_way") return sender == 0 ? 3 : 0;
    return (sender + 2 * receiver) % 3 + 1;
}

struct Fixture {
    MeshPart mesh;
    std::vector<int> donor, donor_cell;
};

/// @brief Directed peer counts differ; ghost destinations reverse the receive-map order.
Fixture make_fixture(int rank, int ranks, std::string_view mode)
{
    Fixture f;
    f.mesh.rank = rank;
    f.mesh.nprocs = ranks;
    f.mesh.n_own = mode == "isolated" && rank == 2 ? 0 : 3;
    constexpr std::array<int, 3> order{2, 0, 1};
    for (int peer = 0; peer < ranks; ++peer) {
        const int send = count(rank, peer, mode), recv = count(peer, rank, mode);
        if (send == 0 && recv == 0) continue;
        f.mesh.nb_ranks.push_back(peer);
        for (int i = 0; i < send; ++i) f.mesh.send_owned_local.push_back(order[static_cast<std::size_t>(i)]);
        f.mesh.send_offsets.push_back(f.mesh.send_owned_local.size());
        for (int i = 0; i < recv; ++i) {
            f.donor.push_back(peer);
            f.donor_cell.push_back(order[static_cast<std::size_t>(i)]);
        }
        f.mesh.recv_offsets.push_back(f.donor.size());
    }
    const auto ghosts = f.donor.size();
    f.mesh.n_cells = f.mesh.n_own + static_cast<Index>(ghosts);
    for (std::size_t i = 0; i < ghosts; ++i)
        f.mesh.recv_ghost_local.push_back(f.mesh.n_own + static_cast<Index>(ghosts - 1 - i));
    return f;
}

template<class T>
void run_exchange(const Exec& exec, std::string_view mode)
{
    int rank = 0, ranks = 0;
    krona::mpi::check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    krona::mpi::check(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    equal(ranks, mode == "multi_peer" || mode == "isolated" ? 3 : 2, "Expected MPI rank count");
    const auto fixture = make_fixture(rank, ranks, mode);
    MeshStorage<> mesh(fixture.mesh, {.data = MeshData::Halo}, exec);
    const auto n = static_cast<std::size_t>(fixture.mesh.n_cells);
    const auto owned = static_cast<std::size_t>(fixture.mesh.n_own);
    krona::fields::FieldsManager<> fields({owned, n, 0, 0}, exec);
    using Location = krona::fields::FieldLocation;
    const auto scalar = fields.template add_field<T>("scalar", Location::Cell);
    const auto vector = fields.template add_field<T, 3>("vector", Location::Cell);
    const auto tensor = fields.template add_field<T, 9, Kokkos::LayoutRight>("tensor", Location::Cell);

    // Two independent groups are active together; finish order differs from start order.
    krona::halo::Exchange<T> primary(mesh.view(), mesh.halo_plan(), MPI_COMM_WORLD, 910, exec);
    primary.add_field(scalar);
    primary.add_field(vector);
    primary.prepare();
    using Memory = typename Exec::memory_space;
    // CPU direct mode is safe; GPU tests keep the portable host-staging contract.
    constexpr auto secondary_transfer = std::is_same_v<Memory, Kokkos::HostSpace>
        ? krona::halo::Transfer::DeviceDirect : krona::halo::Transfer::HostStaged;
    krona::halo::Exchange<T> secondary(mesh.view(), mesh.halo_plan(), MPI_COMM_WORLD, 911, exec,
                                     secondary_transfer);
    for (int c = 0; c < 9; ++c) secondary.add_component(tensor.component(c));
    secondary.prepare();
    equal(primary.component_count(), std::size_t{4}, "Primary group has four scalar planes");
    equal(secondary.component_count(), std::size_t{9}, "Secondary group has nine scalar planes");

    Kokkos::View<int*[2], Memory> expected("test/expected_donors", n);
    auto expected_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, expected);
    for (std::size_t i = 0; i < owned; ++i) {
        expected_host(i, 0) = rank;
        expected_host(i, 1) = static_cast<int>(i);
    }
    for (std::size_t i = 0; i < fixture.donor.size(); ++i) {
        const auto ghost = static_cast<std::size_t>(fixture.mesh.recv_ghost_local[i]);
        expected_host(ghost, 0) = fixture.donor[i];
        expected_host(ghost, 1) = fixture.donor_cell[i];
    }
    Kokkos::deep_copy(exec, expected, expected_host);
    Kokkos::View<int*, Memory> errors("test/errors", 1);
    auto errors_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, errors);

    for (int iteration = 0; iteration < 4; ++iteration) {
        Kokkos::parallel_for("test/fill_owned", Policy(exec, 0, static_cast<Index>(n)),
            KOKKOS_LAMBDA(Index index) {
                const auto i = static_cast<std::size_t>(index);
                const int base = 10000 * iteration + 100 * rank + 10 * static_cast<int>(i);
                scalar(i) = i < owned ? static_cast<T>(base) : T{-777};
                for (int c = 0; c < 3; ++c)
                    vector(i, c) = i < owned ? static_cast<T>(base + c + 1) : T{-777};
                for (int c = 0; c < 9; ++c)
                    tensor(i, c) = i < owned ? static_cast<T>(base + c + 4) : T{-777};
            });
        // Iteration 1 deliberately reuses buffers while the previous unpack state is pending.
        primary.start();
        secondary.start();
        secondary.finish_async();
        primary.finish_async();
        Kokkos::deep_copy(exec, errors, 0);
        Kokkos::parallel_for("test/consume_ghosts", Policy(exec, 0, static_cast<Index>(n)),
            KOKKOS_LAMBDA(Index index) {
                const auto i = static_cast<std::size_t>(index);
                const int base = 10000 * iteration + 100 * expected(i, 0) + 10 * expected(i, 1);
                int failures = scalar(i) != static_cast<T>(base) ? 1 : 0;
                for (int c = 0; c < 3; ++c)
                    failures += vector(i, c) != static_cast<T>(base + c + 1) ? 1 : 0;
                for (int c = 0; c < 9; ++c)
                    failures += tensor(i, c) != static_cast<T>(base + c + 4) ? 1 : 0;
                if (failures != 0) Kokkos::atomic_add(&errors(0), failures);
            });
        Kokkos::deep_copy(exec, errors_host, errors);
        exec.fence("KRONA halo test: consumer result ready");
        equal(errors_host(0), 0, "Ghost consumers see the correct remote components; owned values survive");
        if (iteration != 0) { primary.wait(); secondary.wait(); }
    }

    // A started exchange must be drained safely by its destructor.
    {
        krona::halo::Exchange<T> temporary(mesh.view(), mesh.halo_plan(), MPI_COMM_WORLD, 912, exec);
        temporary.add_field(scalar);
        temporary.prepare();
        temporary.start();
    }
    primary.exchange();
    check(!primary.pending(), "Blocking exchange leaves no pending communication work");
}

void roundtrip(const Exec& exec) { run_exchange<Real>(exec, "roundtrip"); }
void multi_peer(const Exec& exec) { run_exchange<Real>(exec, "multi_peer"); }
void one_way(const Exec& exec) { run_exchange<Real>(exec, "one_way"); }
void isolated(const Exec& exec) { run_exchange<Real>(exec, "isolated"); }
void integer(const Exec& exec) { run_exchange<std::int64_t>(exec, "roundtrip"); }

/// @brief A synthetic self-peer exercises packing/MPI/staging/scatter on one available GPU.
/// Production MeshStorage forbids self-peers; this tests the communication primitive directly.
void loopback(const Exec& exec)
{
    int ranks = 0;
    krona::mpi::check(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    equal(ranks, 1, "Loopback test uses one MPI process");
    struct TestComm {
        MPI_Comm value = MPI_COMM_NULL;
        TestComm() { krona::mpi::check(MPI_Comm_split(MPI_COMM_WORLD, 0, 0, &value), "MPI_Comm_split"); }
        ~TestComm() { krona::mpi::check(MPI_Comm_free(&value), "MPI_Comm_free"); }
    } test_comm;
    using Memory = typename Exec::memory_space;
    Kokkos::View<Index*, Memory> send("test/send_map", 3), recv("test/recv_map", 3);
    auto send_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, send);
    auto recv_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, recv);
    send_host(0) = 2; send_host(1) = 0; send_host(2) = 1;
    recv_host(0) = 5; recv_host(1) = 3; recv_host(2) = 4;
    Kokkos::deep_copy(exec, send, send_host);
    Kokkos::deep_copy(exec, recv, recv_host);
    MeshView<Memory> mesh;
    mesh.sizes.n_own = 3;
    mesh.sizes.n_cells = 6;
    mesh.active_mask = MeshData::Halo;
    mesh.halo.send_owned_local = send;
    mesh.halo.recv_ghost_local = recv;
    const std::array peers{0};
    const std::array<std::size_t, 2> offsets{0, 3};
    const HaloPlanView plan{peers, offsets, offsets};
    krona::fields::FieldsManager<> fields({3, 6, 0, 0}, exec);
    const auto field = fields.add_field<Real, 3>("loopback", krona::fields::FieldLocation::Cell);
    krona::halo::Exchange<> halo(mesh, plan, test_comm.value, 913, exec);
    halo.add_field(field);
    halo.prepare();
    Kokkos::parallel_for("test/fill_loopback", Policy(exec, 0, 6),
        KOKKOS_LAMBDA(Index i) {
            for (int c = 0; c < 3; ++c)
                field(static_cast<std::size_t>(i), c) = i < 3 ? static_cast<Real>(100 * c + i) : Real{-777};
        });
    halo.start();
    halo.finish_async();
    auto host = fields.copy_to_host_async<Real, 3>("loopback");
    halo.wait();
    for (std::size_t i = 0; i < 6; ++i)
        for (int c = 0; c < 3; ++c)
            equal(host(i, c), static_cast<Real>(100 * c + static_cast<int>(i % 3)),
                  "Loopback scatters each component to the correct ghost");
}

} // namespace

int main(int argc, char** argv)
{
    constexpr std::array cases{
        Case{"loopback", loopback}, Case{"roundtrip", roundtrip}, Case{"multi_peer", multi_peer}, Case{"one_way", one_way},
        Case{"isolated", isolated}, Case{"integer", integer}};
    return krona::test::run(argc, argv, "halo", cases);
}
