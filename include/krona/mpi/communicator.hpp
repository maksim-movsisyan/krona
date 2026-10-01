#pragma once

#include <mpi.h>

#include <source_location>

namespace krona::mpi {

// Non-owning view; both handles must remain live while the view is used.
// Null views have rank=-1 and both sizes=0. Copies make no MPI calls.
class CommView final {
public:
    CommView() noexcept = default;

    // Cache metadata without changing the borrowed communicator's handler.
    CommView(
        MPI_Comm comm,
        MPI_Comm abort_scope,
        std::source_location where = std::source_location::current()) noexcept;

    [[nodiscard]] static CommView world(
        std::source_location where = std::source_location::current()) noexcept;
    [[nodiscard]] static CommView self(
        std::source_location where = std::source_location::current()) noexcept;

    [[nodiscard]] bool is_null() const noexcept { return comm_ == MPI_COMM_NULL; }
    [[nodiscard]] int rank() const noexcept { return rank_; }
    [[nodiscard]] int size() const noexcept { return size_; }
    [[nodiscard]] bool is_intercommunicator() const noexcept { return inter_; }
    // Zero for intracommunicators; size() always describes the local group.
    [[nodiscard]] int remote_size() const noexcept { return remote_size_; }
    [[nodiscard]] MPI_Comm native() const noexcept { return comm_; }
    [[nodiscard]] MPI_Comm abort_scope() const noexcept { return abort_scope_; }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    MPI_Comm abort_scope_ = MPI_COMM_NULL;
    int rank_ = -1;
    int size_ = 0;
    int remote_size_ = 0;
    bool inter_ = false;
};

// Unique ownership with MPI_ERRORS_RETURN. MPI calls obey the thread level.
// Collective creation and freeing must match; abort_scope must outlive ownership.
// Creation errors use the parent's existing MPI error handler.
class Communicator final {
public:
    Communicator() noexcept = default;

    // Free collectively on normal destruction; abort during unwinding.
    ~Communicator() noexcept;

    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;

    // Transfer ownership locally; existing views still refer to the same handle.
    Communicator(Communicator&& other) noexcept;

    // Free the destination collectively, then transfer ownership locally.
    Communicator& operator=(Communicator&& other) noexcept;

    // Collective over both groups for an intercommunicator.
    [[nodiscard]] static Communicator duplicate(
        CommView parent,
        std::source_location where = std::source_location::current()) noexcept;

    // Collective; MPI_UNDEFINED excludes a rank and returns an empty owner.
    [[nodiscard]] static Communicator split(
        CommView parent,
        int color,
        int key = 0,
        std::source_location where = std::source_location::current()) noexcept;

    // Take a unique raw handle locally; null its source. Reject WORLD / SELF.
    [[nodiscard]] static Communicator adopt(
        MPI_Comm& comm,
        MPI_Comm abort_scope,
        std::source_location where = std::source_location::current()) noexcept;

    [[nodiscard]] bool is_null() const noexcept { return view_.is_null(); }
    [[nodiscard]] int rank() const noexcept { return view_.rank(); }
    [[nodiscard]] int size() const noexcept { return view_.size(); }
    [[nodiscard]] bool is_intercommunicator() const noexcept
    {
        return view_.is_intercommunicator();
    }
    [[nodiscard]] int remote_size() const noexcept { return view_.remote_size(); }

    [[nodiscard]] CommView view() const & noexcept { return view_; }
    CommView view() const && = delete;
    [[nodiscard]] MPI_Comm native() const & noexcept { return view_.native(); }
    MPI_Comm native() const && = delete;

    // Free collectively and invalidate all borrowed views; empty is a no-op.
    void close(
        std::source_location where = std::source_location::current()) noexcept;

    // Transfer the raw handle locally; the caller becomes responsible for freeing.
    [[nodiscard]] MPI_Comm release() noexcept;

    // Swap ownership locally, with no MPI calls.
    void swap(Communicator& other) noexcept;

private:
    Communicator(MPI_Comm comm, MPI_Comm abort_scope,
                 std::source_location where) noexcept;

    CommView view_;
    std::source_location origin_{};
};

inline void swap(Communicator& first, Communicator& second) noexcept {
    first.swap(second);
}

} // namespace krona::mpi
