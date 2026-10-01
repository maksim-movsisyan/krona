#pragma once

#include <krona/mpi/collective_status.hpp>
#include <krona/mpi/datatype.hpp>
#include <krona/mpi/error.hpp>

#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <source_location>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace krona::mpi {

// Blocking intracommunicator operations on host memory, while MPI is active.
// All ranks must match call order, types, roots, counts and reduction operations.
// Local contract violations abort; no hidden agreements in buffer-level wrappers.
// MPI 3.1 counts/displacements are int, measured in elements, never bytes.

enum class Reduction {
    sum, product, min, max,
    logical_and, logical_or, logical_xor,
    bit_and, bit_or, bit_xor
};

namespace detail {

inline void collective_comm(CommView comm, std::source_location where) noexcept
{
    if (comm.is_null() || comm.is_intercommunicator()) {
        fatal(comm.abort_scope(), "Collective requires a live intracommunicator", where);
    }
}

inline void collective_root(CommView comm, int root, std::source_location where) noexcept
{
    collective_comm(comm, where);
    if (root < 0 || root >= comm.size()) {
        fatal(comm.abort_scope(), "Collective root is outside the communicator", where);
    }
}

[[nodiscard]] inline int collective_count(
    CommView comm, std::size_t count, std::source_location where) noexcept
{
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        fatal(comm.abort_scope(), "Collective element count exceeds MPI 3.1 INT_MAX", where);
    }
    return static_cast<int>(count);
}

inline void collective_size(
    CommView comm, std::size_t actual, std::size_t expected,
    std::source_location where) noexcept
{
    if (actual != expected) {
        fatal(comm.abort_scope(), "Collective buffer size does not match its layout", where);
    }
}

[[nodiscard]] inline std::size_t collective_product(
    CommView comm, std::size_t count, std::source_location where) noexcept
{
    const auto peers = static_cast<std::size_t>(comm.size());
    if (count > std::numeric_limits<std::size_t>::max() / peers) {
        fatal(comm.abort_scope(), "Collective buffer size overflows size_t", where);
    }
    return count * peers;
}

template<class S, std::size_t NS, class R, std::size_t NR>
[[nodiscard]] bool buffers_overlap(std::span<S, NS> send, std::span<R, NR> recv) noexcept
{
    if (send.empty() || recv.empty()) return false;
    const std::less<const void*> before;
    return before(send.data(), recv.data() + recv.size()) &&
           before(recv.data(), send.data() + send.size());
}

template<class S, std::size_t NS, class R, std::size_t NR>
void separate_buffers(
    CommView comm, std::span<S, NS> send, std::span<R, NR> recv,
    std::source_location where) noexcept
{
    if (buffers_overlap(send, recv)) {
        fatal(comm.abort_scope(), "Collective buffers overlap; use an explicit in-place operation", where);
    }
}

inline std::byte ignored_send_buffer{};
inline std::byte ignored_recv_buffer{};

template<class T, std::size_t N>
[[nodiscard]] const void* send_data(std::span<T, N> buffer) noexcept
{
    return buffer.empty() ? &ignored_send_buffer : static_cast<const void*>(buffer.data());
}

template<class T, std::size_t N>
[[nodiscard]] void* recv_data(std::span<T, N> buffer) noexcept
{
    return buffer.empty() ? &ignored_recv_buffer : static_cast<void*>(buffer.data());
}

inline void collective_layout(
    CommView comm, std::span<const int> counts, std::span<const int> displacements,
    std::size_t capacity, std::source_location where) noexcept
{
    const auto peers = static_cast<std::size_t>(comm.size());
    collective_size(comm, counts.size(), peers, where);
    collective_size(comm, displacements.size(), peers, where);
    for (std::size_t i = 0; i < peers; ++i) {
        if (counts[i] < 0 || displacements[i] < 0) {
            fatal(comm.abort_scope(), "Collective counts/displacements must be nonnegative", where);
        }
        if (counts[i] == 0) continue;
        const auto offset = static_cast<std::size_t>(displacements[i]);
        const auto count = static_cast<std::size_t>(counts[i]);
        if (offset > capacity || count > capacity - offset) {
            fatal(comm.abort_scope(), "Collective block extends beyond the buffer", where);
        }
    }
}

template<class T> inline constexpr bool reduction_integer =
    std::is_integral_v<T> && !std::same_as<T, bool> &&
    !std::same_as<T, char> && !std::same_as<T, wchar_t>;

template<class T> inline constexpr bool reduction_complex = false;
template<class T> inline constexpr bool reduction_complex<std::complex<T>> = true;

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

template<Reduction Op>
[[nodiscard]] MPI_Op reduction_operation() noexcept
{
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

inline void alltoallv_buffers(
    CommView comm, const void* send, std::size_t send_size,
    std::span<const int> send_counts, std::span<const int> send_displacements,
    void* recv, std::size_t recv_size,
    std::span<const int> recv_counts, std::span<const int> recv_displacements,
    MPI_Datatype type, std::source_location where) noexcept
{
    collective_layout(comm, send_counts, send_displacements, send_size, where);
    collective_layout(comm, recv_counts, recv_displacements, recv_size, where);
    check(MPI_Alltoallv(send, send_counts.data(), send_displacements.data(), type,
                        recv, recv_counts.data(), recv_displacements.data(), type, comm.native()),
          comm.abort_scope(), "MPI_Alltoallv", where);
}

template<class T, std::size_t N>
void separate_metadata(
    CommView comm, std::span<T, N> recv,
    std::span<const int> counts, std::span<const int> displacements,
    std::source_location where) noexcept
{
    if (buffers_overlap(counts, recv) || buffers_overlap(displacements, recv)) {
        fatal(comm.abort_scope(), "Collective receive buffer overlaps counts/displacements", where);
    }
}

struct AlltoallvAccess;

} // namespace detail

template<Reduction Op, class T>
concept MpiReduction = MpiDatatype<T> && detail::reduction_supported<Op, T>;

// Synchronize participants; not needed before or after other collectives.
inline void barrier(
    CommView comm, std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    check(MPI_Barrier(comm.native()), comm.abort_scope(), "MPI_Barrier", where);
}

// Every rank provides the same element count, type and root.
template<MpiDatatype T, std::size_t N>
    requires (!std::is_const_v<T>)
void broadcast(
    CommView comm, std::span<T, N> buffer, int root = 0,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_root(comm, root, where);
    const int count = detail::collective_count(comm, buffer.size(), where);
    check(MPI_Bcast(detail::recv_data(buffer), count, datatype<T>(), root, comm.native()),
          comm.abort_scope(), "MPI_Bcast", where);
}

// Broadcast one scalar.
template<MpiDatatype T>
    requires (!std::is_const_v<T>)
void broadcast(
    CommView comm, T& value, int root = 0,
    std::source_location where = std::source_location::current()) noexcept
{
    broadcast(comm, std::span<T>{&value, 1}, root, where);
}

// Element-wise reduction into a separate, equally sized buffer.
template<Reduction Op, MpiDatatype T, std::size_t NS, std::size_t NR>
    requires MpiReduction<Op, T>
void allreduce(
    CommView comm, std::span<T, NS> send, std::span<std::remove_const_t<T>, NR> recv,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int count = detail::collective_count(comm, send.size(), where);
    detail::collective_size(comm, recv.size(), send.size(), where);
    detail::separate_buffers(comm, send, recv, where);
    check(MPI_Allreduce(detail::send_data(send), detail::recv_data(recv), count,
                         datatype<T>(), detail::reduction_operation<Op>(), comm.native()),
          comm.abort_scope(), "MPI_Allreduce", where);
}

// Reduce one scalar and return the result on every rank.
template<Reduction Op, MpiDatatype T>
    requires MpiReduction<Op, T>
[[nodiscard]] T allreduce(
    CommView comm, T value,
    std::source_location where = std::source_location::current()) noexcept
{
    T result{};
    allreduce<Op>(comm, std::span<const T>{&value, 1}, std::span<T>{&result, 1}, where);
    return result;
}

// Replace local values with the element-wise reduction.
template<Reduction Op, MpiDatatype T, std::size_t N>
    requires MpiReduction<Op, T> && (!std::is_const_v<T>)
void allreduce_in_place(
    CommView comm, std::span<T, N> buffer,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int count = detail::collective_count(comm, buffer.size(), where);
    check(MPI_Allreduce(MPI_IN_PLACE, detail::recv_data(buffer), count,
                         datatype<T>(), detail::reduction_operation<Op>(), comm.native()),
          comm.abort_scope(), "MPI_Allreduce(in place)", where);
}

// Only root needs a receive buffer; other ranks may pass an empty span.
template<Reduction Op, MpiDatatype T, std::size_t NS, std::size_t NR>
    requires MpiReduction<Op, T>
void reduce(
    CommView comm, std::span<T, NS> send, std::span<std::remove_const_t<T>, NR> recv,
    int root = 0, std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_root(comm, root, where);
    const int count = detail::collective_count(comm, send.size(), where);
    if (comm.rank() == root) {
        detail::collective_size(comm, recv.size(), send.size(), where);
        detail::separate_buffers(comm, send, recv, where);
    }
    check(MPI_Reduce(detail::send_data(send), detail::recv_data(recv), count,
                      datatype<T>(), detail::reduction_operation<Op>(), root, comm.native()),
          comm.abort_scope(), "MPI_Reduce", where);
}

// Return a reduced scalar on root; nullopt elsewhere.
template<Reduction Op, MpiDatatype T>
    requires MpiReduction<Op, T>
[[nodiscard]] std::optional<T> reduce(
    CommView comm, T value, int root = 0,
    std::source_location where = std::source_location::current()) noexcept
{
    T result{};
    reduce<Op>(comm, std::span<const T>{&value, 1}, std::span<T>{&result, 1}, root, where);
    if (comm.rank() == root) return result;
    return std::nullopt;
}

// Gather equal-sized local blocks into rank order on every participant.
template<MpiDatatype T, std::size_t NS, std::size_t NR>
void allgather(
    CommView comm, std::span<T, NS> send, std::span<std::remove_const_t<T>, NR> recv,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int count = detail::collective_count(comm, send.size(), where);
    detail::collective_size(comm, recv.size(), detail::collective_product(comm, send.size(), where), where);
    detail::separate_buffers(comm, send, recv, where);
    check(MPI_Allgather(detail::send_data(send), count, datatype<T>(),
                         detail::recv_data(recv), count, datatype<T>(), comm.native()),
          comm.abort_scope(), "MPI_Allgather", where);
}

// Gather variable-sized blocks; receive regions must not overlap.
template<MpiDatatype T, std::size_t NS, std::size_t NR>
void allgatherv(
    CommView comm, std::span<T, NS> send, std::span<std::remove_const_t<T>, NR> recv,
    std::span<const int> recv_counts, std::span<const int> recv_displacements,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int count = detail::collective_count(comm, send.size(), where);
    detail::collective_layout(comm, recv_counts, recv_displacements, recv.size(), where);
    detail::collective_size(comm, send.size(),
                            static_cast<std::size_t>(recv_counts[static_cast<std::size_t>(comm.rank())]), where);
    detail::separate_buffers(comm, send, recv, where);
    detail::separate_metadata(comm, recv, recv_counts, recv_displacements, where);
    check(MPI_Allgatherv(detail::send_data(send), count, datatype<T>(),
                          detail::recv_data(recv), recv_counts.data(), recv_displacements.data(),
                          datatype<T>(), comm.native()),
          comm.abort_scope(), "MPI_Allgatherv", where);
}

// Exchange equal-sized blocks; count is the number of elements per peer.
template<MpiDatatype T, std::size_t NS, std::size_t NR>
void alltoall(
    CommView comm, std::span<T, NS> send, std::span<std::remove_const_t<T>, NR> recv,
    std::size_t count = 1, std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int mpi_count = detail::collective_count(comm, count, where);
    const auto total = detail::collective_product(comm, count, where);
    detail::collective_size(comm, send.size(), total, where);
    detail::collective_size(comm, recv.size(), total, where);
    detail::separate_buffers(comm, send, recv, where);
    check(MPI_Alltoall(detail::send_data(send), mpi_count, datatype<T>(),
                        detail::recv_data(recv), mpi_count, datatype<T>(), comm.native()),
          comm.abort_scope(), "MPI_Alltoall", where);
}

// Exchange equal-sized blocks using the same buffer on every rank.
template<MpiDatatype T, std::size_t N>
    requires (!std::is_const_v<T>)
void alltoall_in_place(
    CommView comm, std::span<T, N> buffer, std::size_t count = 1,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    const int mpi_count = detail::collective_count(comm, count, where);
    detail::collective_size(comm, buffer.size(), detail::collective_product(comm, count, where), where);
    check(MPI_Alltoall(MPI_IN_PLACE, 0, datatype<T>(), detail::recv_data(buffer),
                        mpi_count, datatype<T>(), comm.native()),
          comm.abort_scope(), "MPI_Alltoall(in place)", where);
}

// Exchange variable blocks; counts must match pairwise and receive regions be disjoint.
template<MpiDatatype T, std::size_t NS, std::size_t NR>
void alltoallv(
    CommView comm, std::span<T, NS> send,
    std::span<const int> send_counts, std::span<const int> send_displacements,
    std::span<std::remove_const_t<T>, NR> recv,
    std::span<const int> recv_counts, std::span<const int> recv_displacements,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    detail::separate_buffers(comm, send, recv, where);
    detail::separate_metadata(comm, recv, send_counts, send_displacements, where);
    detail::separate_metadata(comm, recv, recv_counts, recv_displacements, where);
    detail::alltoallv_buffers(comm, detail::send_data(send), send.size(), send_counts, send_displacements,
                              detail::recv_data(recv), recv.size(), recv_counts, recv_displacements,
                              datatype<T>(), where);
}

// In-place exchange requires pairwise symmetric counts and disjoint regions.
template<MpiDatatype T, std::size_t N>
    requires (!std::is_const_v<T>)
void alltoallv_in_place(
    CommView comm, std::span<T, N> buffer,
    std::span<const int> counts, std::span<const int> displacements,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::collective_comm(comm, where);
    detail::collective_layout(comm, counts, displacements, buffer.size(), where);
    detail::separate_metadata(comm, buffer, counts, displacements, where);
    check(MPI_Alltoallv(MPI_IN_PLACE, nullptr, nullptr, datatype<T>(), detail::recv_data(buffer),
                         counts.data(), displacements.data(), datatype<T>(), comm.native()),
          comm.abort_scope(), "MPI_Alltoallv(in place)", where);
}

// Raw records require identical ABI/representation and no process-local pointers.
template<class T>
concept RawRecord = std::is_class_v<T> && std::is_trivially_copyable_v<T> &&
    std::is_standard_layout_v<T> && !std::is_const_v<T> && !std::is_volatile_v<T>;

// Local reusable byte datatype; destroy before MPI finalization and abort-scope release.
template<RawRecord T>
class RecordDatatype final {
public:
    explicit RecordDatatype(
        CommView comm, std::source_location where = std::source_location::current()) noexcept
        : abort_scope_(comm.abort_scope()), origin_(where)
    {
        detail::collective_comm(comm, where);
        int initialized = 0;
        int finalized = 0;
        check(MPI_Initialized(&initialized), MPI_COMM_NULL, "MPI_Initialized", where);
        check(MPI_Finalized(&finalized), MPI_COMM_NULL, "MPI_Finalized", where);
        if (initialized == 0 || finalized != 0) {
            fatal(MPI_COMM_NULL, "Record datatype requires active MPI", where);
        }
        static_assert(sizeof(T) <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        check(MPI_Type_contiguous(static_cast<int>(sizeof(T)), MPI_BYTE, &type_),
              abort_scope_, "MPI_Type_contiguous(record)", where);
        check(MPI_Type_commit(&type_), abort_scope_, "MPI_Type_commit(record)", where);
    }

    ~RecordDatatype() noexcept { close(origin_); }
    RecordDatatype(const RecordDatatype&) = delete;
    RecordDatatype& operator=(const RecordDatatype&) = delete;
    RecordDatatype(RecordDatatype&& other) noexcept
        : type_(std::exchange(other.type_, MPI_DATATYPE_NULL)),
          abort_scope_(other.abort_scope_), origin_(other.origin_) {}
    RecordDatatype& operator=(RecordDatatype&&) = delete;

    [[nodiscard]] MPI_Datatype native(
        std::source_location where = std::source_location::current()) const & noexcept
    {
        if (type_ == MPI_DATATYPE_NULL) fatal(abort_scope_, "Record datatype is empty", where);
        return type_;
    }
    MPI_Datatype native(std::source_location = std::source_location::current()) const && = delete;

    // Free locally; an empty or moved-from owner is a no-op.
    void close(std::source_location where = std::source_location::current()) noexcept
    {
        if (type_ == MPI_DATATYPE_NULL) return;
        int finalized = 0;
        check(MPI_Finalized(&finalized), MPI_COMM_NULL, "MPI_Finalized", where);
        if (finalized != 0) fatal(MPI_COMM_NULL, "Record datatype outlived MPI", where);
        check(MPI_Type_free(&type_), abort_scope_, "MPI_Type_free(record)", where);
    }

private:
    MPI_Datatype type_ = MPI_DATATYPE_NULL;
    MPI_Comm abort_scope_ = MPI_COMM_NULL;
    std::source_location origin_{};
};

// Explicit raw-record exchange with a previously committed datatype.
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
    detail::separate_buffers(comm, send, recv, where);
    detail::separate_metadata(comm, recv, send_counts, send_displacements, where);
    detail::separate_metadata(comm, recv, recv_counts, recv_displacements, where);
    detail::alltoallv_buffers(comm, detail::send_data(send), send.size(), send_counts, send_displacements,
                              detail::recv_data(recv), recv.size(), recv_counts, recv_displacements,
                              record_type.native(where), where);
}

// Reuse metadata capacity; successful calls replace the previous layout.
// One workspace per active call; returned views expire at the next call or move.
class AlltoallvWorkspace final {
public:
    [[nodiscard]] std::span<const int> received_counts() const & noexcept { return recv_counts_; }
    [[nodiscard]] std::span<const int> received_displacements() const & noexcept { return recv_displacements_; }
    std::span<const int> received_counts() const && = delete;
    std::span<const int> received_displacements() const && = delete;

private:
    friend struct detail::AlltoallvAccess;
    std::vector<int> send_counts_;
    std::vector<int> recv_counts_;
    std::vector<int> send_displacements_;
    std::vector<int> recv_displacements_;
};

namespace detail {

// Zero-count blocks use displacement zero; only occupied offsets must fit int.
[[nodiscard]] inline LocalStatus packed_offsets(
    std::span<const int> counts, std::span<int> displacements,
    std::size_t& total, std::source_location where) noexcept
{
    total = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] < 0) return LocalStatus::failure("Packed count is negative", where);
        const auto count = static_cast<std::size_t>(counts[i]);
        if (count == 0) {
            displacements[i] = 0;
            continue;
        }
        if (total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return LocalStatus::failure("Packed displacement exceeds MPI 3.1 INT_MAX", where);
        }
        if (count > std::numeric_limits<std::size_t>::max() - total) {
            return LocalStatus::failure("Packed buffer size overflows size_t", where);
        }
        displacements[i] = static_cast<int>(total);
        total += count;
    }
    return {};
}

struct AlltoallvAccess {
    template<class T, class Allocator, std::size_t N>
    static void exchange(
        CommView comm, std::span<const int> counts, std::span<T, N> send,
        std::vector<std::remove_const_t<T>, Allocator>& recv, AlltoallvWorkspace& workspace,
        MPI_Datatype type, std::source_location where) noexcept
    {
        collective_comm(comm, where);
        const auto peers = static_cast<std::size_t>(comm.size());
        const auto prepared = LocalStatus::capture([&]() -> LocalStatus {
            if (counts.size() != peers) return LocalStatus::failure("Packed counts size must equal communicator size", where);
            if (buffers_overlap(send, std::span{recv})) {
                return LocalStatus::failure("Packed send/receive buffers must not overlap", where);
            }
            if (buffers_overlap(send, std::span{workspace.recv_counts_}) ||
                buffers_overlap(send, std::span{workspace.recv_displacements_})) {
                return LocalStatus::failure("Packed send buffer must not alias workspace metadata", where);
            }
            // Snapshot counts before any receive buffer or receive-metadata resize.
            workspace.send_counts_.assign(counts.begin(), counts.end());
            workspace.recv_counts_.resize(peers);
            workspace.send_displacements_.resize(peers);
            workspace.recv_displacements_.resize(peers);
            std::size_t total = 0;
            const auto layout = packed_offsets(workspace.send_counts_, workspace.send_displacements_, total, where);
            if (!layout.ok()) return layout;
            if (total != send.size()) return LocalStatus::failure("Packed send buffer size must equal the sum of counts", where);
            return {};
        }, where);
        require_collective_success(comm, prepared, "Alltoallv send preparation", where);

        check(MPI_Alltoall(workspace.send_counts_.data(), 1, datatype<int>(),
                            workspace.recv_counts_.data(), 1, datatype<int>(), comm.native()),
              comm.abort_scope(), "MPI_Alltoall(packed counts)", where);

        const auto allocated = LocalStatus::capture([&]() -> LocalStatus {
            std::size_t total = 0;
            const auto layout = packed_offsets(workspace.recv_counts_, workspace.recv_displacements_, total, where);
            if (!layout.ok()) return layout;
            recv.resize(total);
            return {};
        }, where);
        require_collective_success(comm, allocated, "Alltoallv receive preparation", where);

        check(MPI_Alltoallv(send_data(send), workspace.send_counts_.data(), workspace.send_displacements_.data(), type,
                             recv_data(std::span{recv}), workspace.recv_counts_.data(),
                             workspace.recv_displacements_.data(), type, comm.native()),
              comm.abort_scope(), "MPI_Alltoallv(packed data)", where);
    }
};

} // namespace detail

// Exchange counts, resize recv, then transfer packed peer blocks in rank order.
// Success: two status Allreduces, one counts Alltoall and one data Alltoallv.
template<MpiDatatype T, class Allocator, std::size_t N>
    requires (!std::same_as<std::remove_const_t<T>, bool>)
void alltoallv_packed(
    CommView comm, std::span<const int> counts, std::span<T, N> send,
    std::vector<std::remove_const_t<T>, Allocator>& recv, AlltoallvWorkspace& workspace,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::AlltoallvAccess::exchange(comm, counts, send, recv, workspace, datatype<T>(), where);
}

// Raw-record packed exchange; reuse record_type and workspace between calls.
template<class T, class Allocator, std::size_t N>
    requires RawRecord<std::remove_const_t<T>>
void alltoallv_packed(
    CommView comm, std::span<const int> counts, std::span<T, N> send,
    std::vector<std::remove_const_t<T>, Allocator>& recv, AlltoallvWorkspace& workspace,
    const RecordDatatype<std::remove_const_t<T>>& record_type,
    std::source_location where = std::source_location::current()) noexcept
{
    detail::AlltoallvAccess::exchange(comm, counts, send, recv, workspace, record_type.native(where), where);
}

} // namespace krona::mpi
