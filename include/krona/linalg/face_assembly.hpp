#pragma once

#include <krona/execution/utils.hpp>
#include <krona/linalg/mesh_adapter.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace krona::linalg {

/// @brief Mesh-face to native graph-entry positions, independent of coefficient type and block size.
/// owner_entries covers inner faces; neighbour_entries covers owned/owned faces; diagonal covers owned cells.
/// For CSR an entry indexes a scalar; for BSR it indexes a dense block, not its flat coefficients.
template<class Space>
struct FaceAssemblyMap {
    using memory_space = Space;
    using view_type = Kokkos::View<const std::size_t*, Space>;
    view_type owner_entries, neighbour_entries, diagonal;
};

/// @brief Shared cell-neighbour graph and face-to-entry map for scalar CSR and block BSR matrices.
/// Construct once per indexed mesh topology; copies share all graph/map allocations.
/// Construction is collective/blocking. The mesh is not retained; the borrowed communicator must outlive users.
/// Preserve local cell/face numbering and halo topology; rebuild after connectivity or numbering changes.
template<class Exec = execution::ExecutionSpace>
class FaceAssemblyPlan {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using pattern_type = CsrPattern<Exec>;
    using entry_map_type = FaceAssemblyMap<memory_space>;

    FaceAssemblyPlan(const mesh::host::MeshPart& mesh, mpi::CommView comm, const Exec& exec = {})
        : prepared_(prepare(mesh, comm, exec)), comm_(comm) {}

    [[nodiscard]] const pattern_type& pattern() const noexcept { return prepared_.pattern; }
    [[nodiscard]] entry_map_type entry_map() const noexcept { return prepared_.entries; }
    [[nodiscard]] mpi::CommView communicator() const noexcept { return comm_; }

private:
    struct Prepared {
        pattern_type pattern;
        entry_map_type entries;
    };
    Prepared prepared_;
    mpi::CommView comm_;

    static Prepared prepare(const mesh::host::MeshPart& mesh, mpi::CommView comm, const Exec& exec)
    {
        return mpi::guarded([&] {
            execution::require_kokkos();
            mpi::require(mesh.faces_grouped && mesh.n_internal_faces >= 0 &&
                mesh.n_internal_faces <= mesh.n_inner_faces && mesh.n_inner_faces <= mesh.n_faces,
                "Linalg: face assembly requires grouped face intervals");
            const auto numbering = build_cell_numbering(mesh, comm);
            const auto input = make_face_csr(mesh, numbering);
            for (std::size_t f = 0; f < mesh.face_neigh.size(); ++f) {
                const auto neighbour = mesh.face_neigh[f];
                const bool valid = f < static_cast<std::size_t>(mesh.n_internal_faces) ?
                    (neighbour >= 0 && neighbour < mesh.n_own) :
                    (f < static_cast<std::size_t>(mesh.n_inner_faces) ? neighbour >= mesh.n_own : neighbour < 0);
                mpi::require(valid, "Linalg: face endpoints disagree with grouped face intervals");
            }
            Distribution<Exec> distribution(numbering.owned_gids(), comm);
            // Native-entry assembly needs no additional original-input permutation.
            pattern_type pattern(distribution, input, exec, false);
            const auto native = pattern.native()->getLocalGraphHost();
            const auto find_entry = [&](std::size_t row, GlobalOrdinal gid) {
                const auto column = pattern.native()->getColMap()->getLocalElement(gid);
                mpi::require(column != Teuchos::OrdinalTraits<LocalOrdinal>::invalid(),
                             "Linalg: face assembly column is absent from the graph");
                auto begin = native.row_map(row), end = native.row_map(row+1);
                const auto limit = end;
                while (begin < end) {
                    const auto middle = begin+(end-begin)/2;
                    if (native.entries(middle) < column) begin = middle+1;
                    else end = middle;
                }
                mpi::require(begin < limit && native.entries(begin) == column,
                             "Linalg: missing face assembly graph entry");
                return static_cast<std::size_t>(begin);
            };
            std::vector<std::size_t> diagonal(static_cast<std::size_t>(mesh.n_own)),
                owner(static_cast<std::size_t>(mesh.n_inner_faces)),
                neighbour(static_cast<std::size_t>(mesh.n_internal_faces));
            for (std::size_t i = 0; i < diagonal.size(); ++i)
                diagonal[i] = find_entry(i, numbering.cell_to_global[i]);
            for (std::size_t f = 0; f < owner.size(); ++f) {
                const std::size_t i = static_cast<std::size_t>(mesh.face_owner[f]);
                const std::size_t j = static_cast<std::size_t>(mesh.face_neigh[f]);
                owner[f] = find_entry(i, numbering.cell_to_global[j]);
                if (f < neighbour.size()) neighbour[f] = find_entry(j, numbering.cell_to_global[i]);
            }
            using Buffer = Kokkos::View<std::size_t*, memory_space>;
            Buffer d(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "linalg/face_diagonal"), diagonal.size());
            Buffer o(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "linalg/face_owner_entries"), owner.size());
            Buffer n(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "linalg/face_neighbour_entries"), neighbour.size());
            const auto upload = [&](auto target, const auto& source) {
                using Host = Kokkos::View<const std::size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
                Kokkos::deep_copy(exec, target, Host(source.data(), source.size()));
            };
            execution::complete(exec, [&] { upload(d, diagonal); upload(o, owner); upload(n, neighbour); });
            return Prepared{std::move(pattern), {o, n, d}};
        });
    }
};

} // namespace krona::linalg