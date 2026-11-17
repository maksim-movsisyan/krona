#pragma once

#include <krona/execution/utils.hpp>
#include <krona/linalg/mesh_adapter.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace krona::linalg {

/// @brief Precomputed slots for adding face contributions to a finalized sparse matrix.
/// @tparam Space Kokkos memory space holding the lookup arrays.
/// @details Each slot indexes one scalar in CSR or one whole dense block in BSR.
template<class Space>
struct FaceAssemblyMap {
    using memory_space = Space;
    using view_type = Kokkos::View<const std::size_t*, Space>;
    // owner_entries[f] gives A(owner, neighbour) for internal and MPI interface faces.
    // neighbour_entries[f] gives A(neighbour, owner) for internal faces with two owned cells.
    // diagonal[i] gives A(i, i) for owned cell i.
    view_type owner_entries, neighbour_entries, diagonal;
};

/// @brief Builds the sparse cell-neighbour graph and direct lookup arrays used by assembly kernels.
/// @tparam Exec Kokkos execution space used to build the pattern and upload the lookup arrays.
/// @details Construction is collective and waits for the upload. Copies share all prepared allocations.
/// The plan retains the prepared data and borrows the communicator. Rebuild after changes to
/// connectivity, the halo plan, or local cell and face indices.
template<class Exec = execution::ExecutionSpace>
class FaceAssemblyPlan {
public:
    using execution_space = Exec;
    using memory_space = typename Exec::memory_space;
    using pattern_type = CsrPattern<Exec>;
    using entry_map_type = FaceAssemblyMap<memory_space>;

    /// @brief Prepares a plan for a host partition with grouped internal, interface, and boundary faces.
    /// @param mesh Source partition. The plan does not retain it after construction.
    /// @param comm Communicator matching the mesh partition. Keep it alive while the plan is in use.
    /// @param exec Execution instance used to upload the lookup arrays.
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

    static Prepared prepare(const mesh::host::MeshPart& mesh, mpi::CommView comm, const Exec& exec) {
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
            // Resolve the final native entry once on the host so assembly kernels can use direct indexing.
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
            // An interface face contributes to the owned owner's row on this rank.
            // The second row is local only for faces whose two cells are owned here.
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
            // The upload borrows these host arrays until it finishes.
            execution::complete(exec, [&] {
                upload(d, diagonal);
                upload(o, owner);
                upload(n, neighbour);
            });
            return Prepared{std::move(pattern), {o, n, d}};
        });
    }
};

} // namespace krona::linalg
