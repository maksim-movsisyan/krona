#pragma once

#include <krona/mpi/communicator.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/datatype.hpp>
#include <krona/mpi/error.hpp>

#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <source_location>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace krona::mpi {

// Blocking operations on host memory and live intracommunicators.
// All ranks must match call order, types, counts, roots and reduction operations.
// Buffers and metadata must not overlap except in explicit in-place operations.
// MPI 3.1 counts and displacements are int, measured in elements, not bytes.
// Assertions are local debug checks; no wrapper adds collective status checks.

/// @brief Supported predefined MPI reduction operations.
enum class Reduction {
    sum, product, min, max,
    logical_and, logical_or, logical_xor,
    bit_and, bit_or, bit_xor
};

namespace detail {

/// @brief Rejects an empty communicator without querying MPI state.
inline void collective_comm(CommView comm, std::source_location where) noexcept
{
    // The caller guarantees that the borrowed handle remains live.
    if (comm.is_null()) {
        fatal("Collective requires a live communicator", where);
    }
}

/// @brief Checks the communicator and asserts the root locally in debug builds.
inline void collective_root(
    CommView comm, [[maybe_unused]] int root, std::source_location where) noexcept
{
    collective_comm(comm, where);
    // Root agreement across ranks is the caller's responsibility.
    assert(root >= 0 && root < comm.size());
}

/// @brief Converts an element count to int without silent truncation.
[[nodiscard]] inline int collective_count(
    std::size_t count, std::source_location where) noexcept
{
    // MPI 3.1 cannot represent larger per-call counts.
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        fatal("Collective element count exceeds MPI 3.1 INT_MAX", where);
    }
    return static_cast<int>(count);
}

/// @brief Computes an expected buffer size for debug assertions without overflow.
[[nodiscard]] inline std::size_t collective_product(
    CommView comm, std::size_t count, std::source_location where) noexcept
{
    // collective_comm() has already ensured a nonempty communicator.
    const auto peers = static_cast<std::size_t>(comm.size());
    if (count > std::numeric_limits<std::size_t>::max() / peers) {
        fatal("Collective buffer size overflows size_t", where);
    }
    return count * peers;
}

/// @brief Tests buffer overlap for local debug assertions.
template<class S, std::size_t NS, class R, std::size_t NR>
[[nodiscard]] bool buffers_overlap(std::span<S, NS> send, std::span<R, NR> recv) noexcept
{
    // Empty buffers do not occupy memory.
    if (send.empty() || recv.empty()) return false;
    const std::less<const void*> before;
    return before(send.data(), recv.data() + recv.size()) &&
           before(recv.data(), send.data() + send.size());
}

/// @brief Validates a variable-block layout locally when assertions are enabled.
inline void collective_layout(
    [[maybe_unused]] CommView comm,
    [[maybe_unused]] std::span<const int> counts,
    [[maybe_unused]] std::span<const int> displacements,
    [[maybe_unused]] std::size_t capacity) noexcept
{
    // No peer scan or assertions remain when NDEBUG is defined.
#ifndef NDEBUG
    const auto peers = static_cast<std::size_t>(comm.size());
    assert(counts.size() == peers && displacements.size() == peers);
    for (std::size_t i = 0; i < peers; ++i) {
        assert(counts[i] >= 0 && displacements[i] >= 0);
        if (counts[i] == 0) continue;
        const auto offset = static_cast<std::size_t>(displacements[i]);
        const auto count = static_cast<std::size_t>(counts[i]);
        assert(offset <= capacity && count <= capacity - offset);
    }
#endif
}

template<class T> inline constexpr bool reduction_integer =
    std::is_integral_v<T> && !std::same_as<T, bool> &&
    !std::same_as<T, char> && !std::same_as<T, wchar_t>;

template<class T> inline constexpr bool reduction_complex = false;
template<class T> inline constexpr bool reduction_complex<std::complex<T>> = true;

// These checks run at compile time and add no runtime work.
template<Reduction Op, class T>
inline constexpr bool reduction_supported = [] {
    using U = std::remove_cvref_t<T>;
    if constexpr (Op == Reduction::sum || Op == Reduction::product) {
        return reduction_integer<U> || std::is_floating_point_v<U> || reduction_complex<U>;
    } else if constexpr (Op == Reduction::min || Op == Reduction::max) {
        return reduction_integer<U> || std::is_floating_point_v<U>;
    } else if constexpr (Op == Reduction::logical_and || Op == Reduction::logical_or ||
                         Op == Reduction::logical_xor) {
        return reduction_integer<U> || std::same_as<U, bool>;
    } else if constexpr (Op == Reduction::bit_and || Op == Reduction::bit_or ||
                         Op == Reduction::bit_xor) {
        return reduction_integer<U> || std::same_as<U, std::byte>;
    } else {
        return false;
    }
}();

/// @brief Maps a compile-time reduction choice to its MPI operation.
template<Reduction Op>
[[nodiscard]] MPI_Op reduction_operation() noexcept
{
    // Only the selected operation is compiled into the wrapper.
    if constexpr (Op == Reduction::sum) return MPI_SUM;
    else if constexpr (Op == Reduction::product) return MPI_PROD;
    else if constexpr (Op == Reduction::min) return MPI_MIN;
    else if constexpr (Op == Reduction::max) return MPI_MAX;
    else if constexpr (Op == Reduction::logical_and) return MPI_LAND;
    else if constexpr (Op == Reduction::logical_or) return MPI_LOR;
    else if constexpr (Op == Reduction::logical_xor) return MPI_LXOR;
    else if constexpr (Op == Reduction::bit_and) return MPI_BAND;
    else if constexpr (Op == Reduction::bit_or) return MPI_BOR;
    else return MPI_BXOR;
}

/// @brief Performs one variable-block exchange for a supplied MPI datatype.
inline void alltoallv_buffers(
    CommView comm, const void* send, std::size_t send_size,
    std::span<const int> send_counts, std::span<const int> send_displacements,
    void* recv, std::size_t recv_size,
    std::span<const int> recv_counts, std::span<const int> recv_displacements,
    MPI_Datatype type, std::source_location where) noexcept
{
    // Layout validation is debug-only; no status collective is inserted.
    collective_layout(comm, send_counts, send_displacements, send_size);
    collective_layout(comm, recv_counts, recv_displacements, recv_size);
    check(MPI_Alltoallv(send, send_counts.data(), send_displacements.data(), type,
                       recv, recv_counts.data(), recv_displacements.data(), type,
                       comm.native()), "MPI_Alltoallv", where);
}

} // namespace detail

template<Reduction Op, class T>
concept MpiReduction = MpiDatatype<T> && detail::reduction_supported<Op, T>;

/// @brief Synchronizes all participants with one MPI_Barrier.
inline void barrier(
    CommView comm, std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    // No extra synchronization is added around the requested barrier.
    check(MPI_Barrier(comm.native()), "MPI_Barrier", where);
}

/// @brief Broadcasts a buffer; all ranks provide matching counts, types and root.
template<MpiDatatype T, std::size_t N> requires (!std::is_const_v<T>)
void broadcast(
    CommView comm, std::span<T, N> buffer, int root = 0,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_root(comm, root, where);
    const int count = detail::collective_count(buffer.size(), where);
    // Zero-count buffers may be empty.
    check(MPI_Bcast(buffer.data(), count, datatype<T>(), root, comm.native()),
          "MPI_Bcast", where);
}

/// @brief Broadcasts one scalar from the selected root.
template<MpiDatatype T> requires (!std::is_const_v<T>)
void broadcast(
    CommView comm, T& value, int root = 0,
    std::source_location where = std::source_location::current()) noexcept
{
    // Reuse the buffer wrapper without allocating memory.
    broadcast(comm, std::span<T>{&value, 1}, root, where);
}

/// @brief Reduces equal-sized separate buffers into a result on every rank.
template<Reduction Op, MpiDatatype T, std::size_t NS, std::size_t NR>
    requires MpiReduction<Op, T>
void allreduce(
    CommView comm, std::span<T, NS> send, std::span<std::remove_const_t<T>, NR> recv,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int count = detail::collective_count(send.size(), where);
    // Aliasing requires the explicit in-place wrapper.
    assert(recv.size() == send.size());
    assert(!detail::buffers_overlap(send, recv));
    check(MPI_Allreduce(send.data(), recv.data(), count, datatype<T>(),
                        detail::reduction_operation<Op>(), comm.native()),
          "MPI_Allreduce", where);
}

/// @brief Reduces one scalar and returns the result on every rank.
template<Reduction Op, MpiDatatype T> requires MpiReduction<Op, T>
[[nodiscard]] T allreduce(
    CommView comm, T value,
    std::source_location where = std::source_location::current()) noexcept
{
    // Stack storage keeps the scalar path allocation-free.
    T result{};
    allreduce<Op>(comm, std::span<const T>{&value, 1}, std::span<T>{&result, 1}, where);
    return result;
}

/// @brief Replaces a buffer with its element-wise reduction on every rank.
template<Reduction Op, MpiDatatype T, std::size_t N>
    requires MpiReduction<Op, T> && (!std::is_const_v<T>)
void allreduce_in_place(
    CommView comm, std::span<T, N> buffer,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int count = detail::collective_count(buffer.size(), where);
    // MPI_IN_PLACE explicitly permits the shared input/output buffer.
    check(MPI_Allreduce(MPI_IN_PLACE, buffer.data(), count, datatype<T>(),
                        detail::reduction_operation<Op>(), comm.native()),
          "MPI_Allreduce(in place)", where);
}

/// @brief Reduces into root's receive buffer; other ranks may pass an empty span.
template<Reduction Op, MpiDatatype T, std::size_t NS, std::size_t NR>
    requires MpiReduction<Op, T>
void reduce(
    CommView comm, std::span<T, NS> send, std::span<std::remove_const_t<T>, NR> recv,
    int root = 0, std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_root(comm, root, where);
    const int count = detail::collective_count(send.size(), where);
    // The receive buffer is significant only on root.
    if (comm.rank() == root) {
        assert(recv.size() == send.size());
        assert(!detail::buffers_overlap(send, recv));
    }
    check(MPI_Reduce(send.data(), recv.data(), count, datatype<T>(),
                     detail::reduction_operation<Op>(), root, comm.native()),
          "MPI_Reduce", where);
}

/// @brief Returns the reduced scalar on root and nullopt elsewhere.
template<Reduction Op, MpiDatatype T> requires MpiReduction<Op, T>
[[nodiscard]] std::optional<T> reduce(
    CommView comm, T value, int root = 0,
    std::source_location where = std::source_location::current()) noexcept
{
    // All participants still call the same buffer reduction.
    T result{};
    reduce<Op>(comm, std::span<const T>{&value, 1}, std::span<T>{&result, 1}, root, where);
    if (comm.rank() == root) return result;
    return std::nullopt;
}

/// @brief Gathers equal-sized blocks in rank order on every participant.
template<MpiDatatype T, std::size_t NS, std::size_t NR>
void allgather(
    CommView comm, std::span<T, NS> send, std::span<std::remove_const_t<T>, NR> recv,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int count = detail::collective_count(send.size(), where);
    // Every rank receives one block from every participant.
    assert(recv.size() == detail::collective_product(comm, send.size(), where));
    assert(!detail::buffers_overlap(send, recv));
    check(MPI_Allgather(send.data(), count, datatype<T>(), recv.data(), count,
                        datatype<T>(), comm.native()), "MPI_Allgather", where);
}

/// @brief Gathers variable-sized blocks into disjoint receive regions.
template<MpiDatatype T, std::size_t NS, std::size_t NR>
void allgatherv(
    CommView comm, std::span<T, NS> send, std::span<std::remove_const_t<T>, NR> recv,
    std::span<const int> recv_counts, std::span<const int> recv_displacements,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int count = detail::collective_count(send.size(), where);
    // Pairwise type/count agreement remains the caller's responsibility.
    detail::collective_layout(comm, recv_counts, recv_displacements, recv.size());
    assert(send.size() == static_cast<std::size_t>(recv_counts[static_cast<std::size_t>(comm.rank())]));
    assert(!detail::buffers_overlap(send, recv));
    check(MPI_Allgatherv(send.data(), count, datatype<T>(), recv.data(),
                         recv_counts.data(), recv_displacements.data(), datatype<T>(),
                         comm.native()), "MPI_Allgatherv", where);
}

/// @brief Exchanges equal-sized blocks; count is the number of elements per peer.
template<MpiDatatype T, std::size_t NS, std::size_t NR>
void alltoall(
    CommView comm, std::span<T, NS> send, std::span<std::remove_const_t<T>, NR> recv,
    std::size_t count = 1, std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int mpi_count = detail::collective_count(count, where);
    // Input and output contain one equally sized block per peer.
    assert(send.size() == detail::collective_product(comm, count, where));
    assert(recv.size() == send.size());
    assert(!detail::buffers_overlap(send, recv));
    check(MPI_Alltoall(send.data(), mpi_count, datatype<T>(), recv.data(),
                       mpi_count, datatype<T>(), comm.native()), "MPI_Alltoall", where);
}

/// @brief Exchanges equal-sized blocks in place on every rank.
template<MpiDatatype T, std::size_t N> requires (!std::is_const_v<T>)
void alltoall_in_place(
    CommView comm, std::span<T, N> buffer, std::size_t count = 1,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int mpi_count = detail::collective_count(count, where);
    assert(buffer.size() == detail::collective_product(comm, count, where));
    // Send arguments are ignored when MPI_IN_PLACE is used.
    check(MPI_Alltoall(MPI_IN_PLACE, 0, datatype<T>(), buffer.data(), mpi_count,
                       datatype<T>(), comm.native()), "MPI_Alltoall(in place)", where);
}

/// @brief Exchanges variable blocks with matching pairwise counts and disjoint outputs.
template<MpiDatatype T, std::size_t NS, std::size_t NR>
void alltoallv(
    CommView comm, std::span<T, NS> send,
    std::span<const int> send_counts, std::span<const int> send_displacements,
    std::span<std::remove_const_t<T>, NR> recv,
    std::span<const int> recv_counts, std::span<const int> recv_displacements,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    // This wrapper issues exactly one data collective.
    assert(!detail::buffers_overlap(send, recv));
    detail::alltoallv_buffers(comm, send.data(), send.size(), send_counts, send_displacements,
                             recv.data(), recv.size(), recv_counts, recv_displacements,
                             datatype<T>(), where);
}

/// @brief Exchanges variable blocks in place; counts must be pairwise symmetric.
template<MpiDatatype T, std::size_t N> requires (!std::is_const_v<T>)
void alltoallv_in_place(
    CommView comm, std::span<T, N> buffer,
    std::span<const int> counts, std::span<const int> displacements,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    detail::collective_layout(comm, counts, displacements, buffer.size());
    // Receive layout describes both directions of the in-place exchange.
    check(MPI_Alltoallv(MPI_IN_PLACE, nullptr, nullptr, datatype<T>(), buffer.data(),
                        counts.data(), displacements.data(), datatype<T>(), comm.native()),
          "MPI_Alltoallv(in place)", where);
}

// Raw records require identical ABI/representation and no process-local pointers.
template<class T>
concept RawRecord = std::is_class_v<T> && std::is_trivially_copyable_v<T> &&
    std::is_standard_layout_v<T> && !std::is_const_v<T> && !std::is_volatile_v<T>;

/// @brief Owns a reusable byte datatype for a raw record; creation and freeing are local.
/// The datatype must be destroyed before MPI_Finalize.
template<RawRecord T>
class RecordDatatype final {
public:
    /// @brief Creates a committed datatype covering exactly sizeof(T) bytes.
    explicit RecordDatatype(
        std::source_location where = std::source_location::current()) noexcept
    {
        // Datatypes are local MPI resources and need no communicator.
        if (!is_active()) fatal("Record datatype requires active MPI", where);
        static_assert(sizeof(T) <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        check(MPI_Type_contiguous(static_cast<int>(sizeof(T)), MPI_BYTE, &type_),
              "MPI_Type_contiguous(record)", where);
        check(MPI_Type_commit(&type_), "MPI_Type_commit(record)", where);
    }

    /// @brief Frees the owned datatype locally.
    ~RecordDatatype() noexcept
    {
        // Empty and moved-from objects do not enter MPI.
        close();
    }

    /// @brief Datatype ownership cannot be copied.
    RecordDatatype(const RecordDatatype&) = delete;

    /// @brief Datatype ownership cannot be copy-assigned.
    RecordDatatype& operator=(const RecordDatatype&) = delete;

    /// @brief Transfers ownership locally and leaves the source empty.
    RecordDatatype(RecordDatatype&& other) noexcept
        : type_(std::exchange(other.type_, MPI_DATATYPE_NULL))
    {
        // The committed MPI datatype itself remains unchanged.
    }

    /// @brief Datatype ownership cannot be move-assigned.
    RecordDatatype& operator=(RecordDatatype&&) = delete;

    /// @brief Borrows the committed datatype handle.
    [[nodiscard]] MPI_Datatype native(
        std::source_location where = std::source_location::current()) const & noexcept
    {
        // Reject use of an empty or moved-from owner.
        if (type_ == MPI_DATATYPE_NULL) fatal("Record datatype is empty", where);
        return type_;
    }

    /// @brief Prevents borrowing from a temporary owner.
    MPI_Datatype native(std::source_location = std::source_location::current()) const && = delete;

    /// @brief Frees the datatype locally; an empty owner is a no-op.
    void close(std::source_location where = std::source_location::current()) noexcept
    {
        if (type_ == MPI_DATATYPE_NULL) return;
        // Do not free a datatype after MPI has shut down.
        if (!is_active()) fatal("Record datatype outlived MPI", where);
        check(MPI_Type_free(&type_), "MPI_Type_free(record)", where);
    }

private:
    MPI_Datatype type_ = MPI_DATATYPE_NULL;
};

/// @brief Exchanges raw records with a previously committed record datatype.
template<class T, std::size_t NS, std::size_t NR>
    requires RawRecord<std::remove_const_t<T>>
void alltoallv(
    CommView comm, std::span<T, NS> send,
    std::span<const int> send_counts, std::span<const int> send_displacements,
    std::span<std::remove_const_t<T>, NR> recv,
    std::span<const int> recv_counts, std::span<const int> recv_displacements,
    const RecordDatatype<std::remove_const_t<T>>& record_type,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    // Counts and displacements are measured in records, not bytes.
    assert(!detail::buffers_overlap(send, recv));
    detail::alltoallv_buffers(comm, send.data(), send.size(), send_counts, send_displacements,
                             recv.data(), recv.size(), recv_counts, recv_displacements,
                             record_type.native(where), where);
}

/// @brief Reuses metadata capacity for packed exchanges; use one workspace per active call.
/// Returned views expire when the corresponding storage changes or is destroyed.
struct AlltoallvWorkspace final {
    std::vector<int> send_counts;
    std::vector<int> recv_counts;
    std::vector<int> send_displacements;
    std::vector<int> recv_displacements;

    /// @brief Borrows the receive counts from the last packed exchange.
    [[nodiscard]] std::span<const int> received_counts() const & noexcept { return recv_counts; }

    /// @brief Borrows the receive displacements from the last packed exchange.
    [[nodiscard]] std::span<const int> received_displacements() const & noexcept { return recv_displacements; }

    /// @brief Prevents borrowing metadata from a temporary workspace.
    std::span<const int> received_counts() const && = delete;

    /// @brief Prevents borrowing metadata from a temporary workspace.
    std::span<const int> received_displacements() const && = delete;
};

namespace detail {

/// @brief Builds packed offsets and returns the total number of elements.
inline std::size_t packed_offsets(
    std::span<const int> counts, std::span<int> displacements,
    std::source_location where) noexcept
{
    assert(displacements.size() == counts.size());
    std::size_t total = 0;
    // Offsets are computed once as part of building the actual exchange layout.
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] < 0) fatal("Packed count is negative", where);
        const auto count = static_cast<std::size_t>(counts[i]);
        if (count == 0) {
            displacements[i] = 0;
            continue;
        }
        if (total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            fatal("Packed displacement exceeds MPI 3.1 INT_MAX", where);
        }
        if (count > std::numeric_limits<std::size_t>::max() - total) {
            fatal("Packed buffer size overflows size_t", where);
        }
        displacements[i] = static_cast<int>(total);
        total += count;
    }
    return total;
}

/// @brief Exchanges counts, allocates receive storage and transfers packed data.
template<class T, class Allocator, std::size_t N>
void packed_exchange(
    CommView comm, std::span<const int> counts, std::span<T, N> send,
    std::vector<std::remove_const_t<T>, Allocator>& recv, AlltoallvWorkspace& workspace,
    MPI_Datatype type, std::source_location where) noexcept
{
    collective_comm(comm, where);
    const auto peers = static_cast<std::size_t>(comm.size());
    assert(counts.size() == peers);
    assert(!buffers_overlap(send, std::span{recv}));
    assert(!buffers_overlap(send, std::span{workspace.send_counts}));
    assert(!buffers_overlap(send, std::span{workspace.recv_counts}));
    assert(!buffers_overlap(send, std::span{workspace.send_displacements}));
    assert(!buffers_overlap(send, std::span{workspace.recv_displacements}));

    try {
        // Snapshot counts before changing metadata or receive storage.
        workspace.send_counts.assign(counts.begin(), counts.end());
        workspace.recv_counts.resize(peers);
        workspace.send_displacements.resize(peers);
        workspace.recv_displacements.resize(peers);
        [[maybe_unused]] const auto send_total =
            packed_offsets(workspace.send_counts, workspace.send_displacements, where);
        assert(send_total == send.size());

        // This count exchange is needed to determine receive sizes.
        check(MPI_Alltoall(workspace.send_counts.data(), 1, MPI_INT,
                           workspace.recv_counts.data(), 1, MPI_INT, comm.native()),
              "MPI_Alltoall(packed counts)", where);

        const auto recv_total =
            packed_offsets(workspace.recv_counts, workspace.recv_displacements, where);
        recv.resize(recv_total);

        // Exactly one data exchange follows the count exchange.
        check(MPI_Alltoallv(send.data(), workspace.send_counts.data(),
                            workspace.send_displacements.data(), type, recv.data(),
                            workspace.recv_counts.data(), workspace.recv_displacements.data(),
                            type, comm.native()), "MPI_Alltoallv(packed data)", where);
    } catch (const std::exception& error) {
        // Abort WORLD immediately so peers cannot remain blocked after allocation failure.
        fatal(error.what(), where);
    } catch (...) {
        fatal("Unknown exception during packed exchange preparation", where);
    }
}

} // namespace detail

/// @brief Exchanges packed blocks, resizes recv and retains receive metadata.
/// Performs one counts Alltoall and one data Alltoallv; no status collectives.
template<MpiDatatype T, class Allocator, std::size_t N>
    requires (!std::same_as<std::remove_const_t<T>, bool>)
void alltoallv_packed(
    CommView comm, std::span<const int> counts, std::span<T, N> send,
    std::vector<std::remove_const_t<T>, Allocator>& recv, AlltoallvWorkspace& workspace,
    std::source_location where = std::source_location::current()) noexcept
{
    // Reuse workspace capacity across repeated exchanges.
    detail::packed_exchange(comm, counts, send, recv, workspace, datatype<T>(), where);
}

/// @brief Exchanges packed raw records using reusable datatype and metadata storage.
template<class T, class Allocator, std::size_t N>
    requires RawRecord<std::remove_const_t<T>>
void alltoallv_packed(
    CommView comm, std::span<const int> counts, std::span<T, N> send,
    std::vector<std::remove_const_t<T>, Allocator>& recv, AlltoallvWorkspace& workspace,
    const RecordDatatype<std::remove_const_t<T>>& record_type,
    std::source_location where = std::source_location::current()) noexcept
{
    // Record counts use the datatype extent of sizeof(T).
    detail::packed_exchange(comm, counts, send, recv, workspace, record_type.native(where), where);
}

} // namespace krona::mpi