#pragma once

#include <mpi.h>

#include <source_location>

namespace krona::mpi {

/// @brief Borrows an intracommunicator and caches its rank and size.
/// The handle must stay live; an empty view has rank=-1 and size=0.
class CommView final
{
public:
    /// @brief Creates an empty view without calling MPI.
    CommView() noexcept = default;

    /// @brief Borrows a handle without changing its error handler.
    explicit CommView(
        MPI_Comm comm, std::source_location where = std::source_location::current()) noexcept;

    /// @brief Returns a borrowed view of MPI_COMM_WORLD.
    [[nodiscard]] static CommView world(std::source_location where = std::source_location::current()) noexcept;

    /// @brief Returns a borrowed view of MPI_COMM_SELF.
    [[nodiscard]] static CommView self(std::source_location where = std::source_location::current()) noexcept;

    /// @brief Returns whether the view is empty; no MPI call is made.
    [[nodiscard]] bool is_null() const noexcept { return comm_ == MPI_COMM_NULL; }

        /// @brief Returns the cached rank; no MPI call is made.
    [[nodiscard]] int rank() const noexcept { return rank_; }

    /// @brief Returns the cached size; no MPI call is made.
    [[nodiscard]] int size() const noexcept { return size_; }

    /// @brief Returns the borrowed MPI handle.
    [[nodiscard]] MPI_Comm native() const noexcept { return comm_; }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = -1;
    int size_ = 0;
};

/// @brief Owns an intracommunicator configured with MPI_ERRORS_RETURN.
/// Creation and freeing are collective; destroy owners before MPI_Finalize.
class Comm final
{
public:
    /// @brief Creates an empty owner without calling MPI.
    Comm() noexcept = default;

    /// @brief Frees the communicator; aborts if destruction occurs during unwinding.
    ~Comm() noexcept;

    /// @brief Communicator ownership cannot be copied.
    Comm(const Comm&) = delete;
    Comm& operator=(const Comm&) = delete;

    /// @brief Transfers ownership locally and leaves the source empty.
    Comm(Comm&& other) noexcept;
    Comm& operator=(Comm&& other) noexcept;

    /// @brief Duplicates a live parent collectively; creation uses its error handler.
    [[nodiscard]] static Comm duplicate(
        CommView parent, std::source_location where = std::source_location::current()) noexcept;

    /// @brief Splits a live parent collectively; MPI_UNDEFINED returns an empty owner.
    [[nodiscard]] static Comm split(
        CommView parent,
        int color,
        int key = 0,
        std::source_location where = std::source_location::current()) noexcept;

    /// @brief Takes a uniquely owned raw handle and clears its source.
    /// Predefined communicators cannot be adopted; null produces an empty owner.
    [[nodiscard]] static Comm adopt(
        MPI_Comm& comm, std::source_location where = std::source_location::current()) noexcept;

    /// @brief Returns whether the owner is empty; no MPI call is made.
    [[nodiscard]] bool is_null() const noexcept { return view_.is_null(); }

    /// @brief Returns the cached rank; no MPI call is made.
    [[nodiscard]] int rank() const noexcept { return view_.rank(); }

    /// @brief Returns the cached size; no MPI call is made.
    [[nodiscard]] int size() const noexcept { return view_.size(); }

    /// @brief Borrows the communicator; the view must not outlive its handle.
    [[nodiscard]] CommView view() const & noexcept { return view_; }

    /// @brief Prevents borrowing from a temporary owner.
    CommView view() const && = delete;

    /// @brief Returns the borrowed handle; callers must not free it.
    [[nodiscard]] MPI_Comm native() const & noexcept { return view_.native(); }

    /// @brief Prevents extracting a borrowed handle from a temporary owner.
    MPI_Comm native() const && = delete;

    /// @brief Frees the communicator collectively and invalidates borrowed views.
    /// An empty owner is a no-op.
    void close(std::source_location where = std::source_location::current()) noexcept;

    /// @brief Transfers the raw handle to the caller and leaves the owner empty.
    [[nodiscard]] MPI_Comm release() noexcept;

    /// @brief Swaps ownership locally without calling MPI.
    void swap(Comm& other) noexcept;

private:
    /// @brief Initializes cached metadata and sets MPI_ERRORS_RETURN.
    Comm(MPI_Comm comm, std::source_location where) noexcept;

    CommView view_;
};

/// @brief Swaps owners locally without calling MPI.
inline void swap(Comm& first, Comm& second) noexcept
{
    // Delegate to the member swap for ADL-based generic code.
    first.swap(second);
}

} // namespace krona::mpi