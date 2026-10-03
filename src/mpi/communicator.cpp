#include <krona/mpi/communicator.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <exception>
#include <utility>

namespace krona::mpi {

/// @brief Borrows a handle and queries its rank and size once.
CommView::CommView(MPI_Comm comm, std::source_location where) noexcept : comm_(comm)
{
    // Empty views are valid even outside MPI's active lifetime.
    if (is_null()) {
        return;
    }
    if (!is_active()) {
        fatal("Creating a communicator view requires active MPI", where);
    }

    // Cache metadata without modifying the borrowed error handler.
    check(MPI_Comm_rank(comm_, &rank_), "MPI_Comm_rank", where);
    check(MPI_Comm_size(comm_, &size_), "MPI_Comm_size", where);
}

/// @brief Creates a borrowed view of the world communicator.
CommView CommView::world(std::source_location where) noexcept
{
    // WORLD remains owned by MPI.
    return CommView{MPI_COMM_WORLD, where};
}

/// @brief Creates a borrowed view of the single-process communicator.
CommView CommView::self(std::source_location where) noexcept
{
    // SELF remains owned by MPI.
    return CommView{MPI_COMM_SELF, where};
}

/// @brief Initializes an owned handle and its error handler.
Comm::Comm(MPI_Comm comm, std::source_location where) noexcept : view_(comm, where)
{
    // Only owned communicators have their error handler changed.
    if (!is_null()) {
        check(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN),
              "MPI_Comm_set_errhandler", where);
    }
}

/// @brief Releases ownership through the normal collective close path.
Comm::~Comm() noexcept
{
    // Empty and moved-from objects do not call MPI.
    close();
}

/// @brief Transfers ownership without creating or freeing a communicator.
Comm::Comm(Comm&& other) noexcept : view_(std::exchange(other.view_, CommView{}))
{
    // Existing views still refer to the same live MPI handle.
}

/// @brief Closes the destination and transfers ownership from the source.
Comm& Comm::operator=(Comm&& other) noexcept
{
    if (this != &other) {
        // Closing a nonempty destination is collective.
        close();
        view_ = std::exchange(other.view_, CommView{});
    }
    return *this;
}

/// @brief Creates an owned duplicate of a live parent communicator.
Comm Comm::duplicate(CommView parent, std::source_location where) noexcept
{
    // Check the lifetime boundary before entering the collective operation.
    if (parent.is_null() || !is_active()) {
        fatal("Duplicating a communicator requires a live parent and active MPI", where);
    }

    MPI_Comm comm = MPI_COMM_NULL;
    check(MPI_Comm_dup(parent.native(), &comm), "MPI_Comm_dup", where);
    return Comm{comm, where};
}

/// @brief Creates a communicator for ranks with the same split color.
Comm Comm::split(
    CommView parent,
    int color,
    int key,
    std::source_location where) noexcept
{
    // Every parent rank participates, including excluded ranks.
    if (parent.is_null() || !is_active()) {
        fatal("Splitting a communicator requires a live parent and active MPI", where);
    }
    if (color < 0 && color != MPI_UNDEFINED) {
        fatal("Split color must be nonnegative or MPI_UNDEFINED", where);
    }

    MPI_Comm comm = MPI_COMM_NULL;
    check(MPI_Comm_split(parent.native(), color, key, &comm), "MPI_Comm_split", where);
    return Comm{comm, where};
}

/// @brief Takes responsibility for freeing a uniquely owned raw communicator.
Comm Comm::adopt(MPI_Comm& comm, std::source_location where) noexcept
{
    // Predefined handles are borrowed and must never be freed by an owner.
    if (comm == MPI_COMM_WORLD || comm == MPI_COMM_SELF) {
        fatal("Predefined communicators cannot be adopted", where);
    }

    Comm owner{comm, where};
    // Remove the raw source handle to make the ownership transfer explicit.
    comm = MPI_COMM_NULL;
    return owner;
}


/// @brief Frees a live owned communicator and clears cached metadata.
void Comm::close(std::source_location where) noexcept
{
    if (is_null()) {
        return;
    }

    // Never call MPI_Comm_free after MPI_Finalize or during exception unwinding.
    if (!is_active()) {
        fatal("An owned communicator must be closed before MPI_Finalize", where);
    }
    if (std::uncaught_exceptions() != 0) {
        fatal("Cannot free an owned communicator during exception unwinding", where);
    }

    MPI_Comm comm = view_.native();
    check(MPI_Comm_free(&comm), "MPI_Comm_free", where);
    view_ = CommView{};
}

/// @brief Releases the raw handle without freeing it.
MPI_Comm Comm::release() noexcept
{
    // The caller now owns the handle; existing views remain borrowed from it.
    const MPI_Comm comm = view_.native();
    view_ = CommView{};
    return comm;
}

/// @brief Exchanges owners without any MPI communication.
void Comm::swap(Comm& other) noexcept
{
    // Metadata and the handle move together.
    std::swap(view_, other.view_);
}

} // namespace krona::mpi