#include <krona/mpi/communicator.hpp>
#include <krona/mpi/error.hpp>

#include <exception>
#include <utility>

namespace krona::mpi {
namespace {

void require_active(MPI_Comm abort_scope, std::source_location where) noexcept {
    int initialized = 0;
    int finalized = 0;
    check(MPI_Initialized(&initialized), MPI_COMM_NULL, "MPI_Initialized", where);
    check(MPI_Finalized(&finalized), MPI_COMM_NULL, "MPI_Finalized", where);
    if (initialized == 0 || finalized != 0) {
        fatal(MPI_COMM_NULL, "Communicator operation requires active MPI", where);
    }
    if (abort_scope == MPI_COMM_NULL) {
        fatal(MPI_COMM_NULL, "A live abort scope is required for a communicator", where);
    }
}

void require_parent(CommView parent, std::source_location where) noexcept {
    if (parent.is_null()) {
        fatal(parent.abort_scope(), "A null communicator cannot be used as a parent", where);
    }
    require_active(parent.abort_scope(), where);
}

} // namespace

CommView::CommView(
    MPI_Comm comm,
    MPI_Comm abort_scope,
    std::source_location where) noexcept
    : comm_(comm), abort_scope_(abort_scope) {
    if (is_null()) {
        return;
    }
    require_active(abort_scope_, where);
    check(MPI_Comm_rank(comm_, &rank_), abort_scope_, "MPI_Comm_rank", where);
    check(MPI_Comm_size(comm_, &size_), abort_scope_, "MPI_Comm_size", where);

    int inter = 0;
    check(MPI_Comm_test_inter(comm_, &inter), abort_scope_, "MPI_Comm_test_inter", where);
    inter_ = inter != 0;
    if (inter_) {
        check(MPI_Comm_remote_size(comm_, &remote_size_), abort_scope_,
              "MPI_Comm_remote_size", where);
    }
}

CommView CommView::world(std::source_location where) noexcept {
    return CommView{MPI_COMM_WORLD, MPI_COMM_WORLD, where};
}

CommView CommView::self(std::source_location where) noexcept {
    return CommView{MPI_COMM_SELF, MPI_COMM_WORLD, where};
}

Communicator::Communicator(
    MPI_Comm comm,
    MPI_Comm abort_scope,
    std::source_location where) noexcept
    : origin_(where) {
    if (comm != MPI_COMM_NULL) {
        check(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN), abort_scope,
              "MPI_Comm_set_errhandler", where);
    }
    view_ = CommView{comm, abort_scope, where};
}

Communicator::~Communicator() noexcept {
    close(origin_);
}

Communicator::Communicator(Communicator&& other) noexcept
    : view_(std::exchange(other.view_, CommView{})), origin_(other.origin_)
{}

Communicator& Communicator::operator=(Communicator&& other) noexcept {
    if (this != &other) {
        close(origin_);
        view_ = std::exchange(other.view_, CommView{});
        origin_ = other.origin_;
    }
    return *this;
}

Communicator Communicator::duplicate(CommView parent, std::source_location where) noexcept {
    require_parent(parent, where);
    MPI_Comm comm = MPI_COMM_NULL;
    check(MPI_Comm_dup(parent.native(), &comm), parent.abort_scope(),
          "MPI_Comm_dup", where);
    return Communicator{comm, parent.abort_scope(), where};
}

Communicator Communicator::split(
    CommView parent,
    int color,
    int key,
    std::source_location where) noexcept {
    require_parent(parent, where);
    if (color < 0 && color != MPI_UNDEFINED) {
        fatal(parent.abort_scope(), "Split color must be nonnegative or MPI_UNDEFINED", where);
    }
    MPI_Comm comm = MPI_COMM_NULL;
    check(MPI_Comm_split(parent.native(), color, key, &comm), parent.abort_scope(),
          "MPI_Comm_split", where);
    return Communicator{comm, parent.abort_scope(), where};
}

Communicator Communicator::adopt(
    MPI_Comm& comm,
    MPI_Comm abort_scope,
    std::source_location where) noexcept {
    if (comm != MPI_COMM_NULL) {
        require_active(abort_scope, where);
        if (comm == MPI_COMM_WORLD || comm == MPI_COMM_SELF) {
            fatal(abort_scope, "Predefined communicators cannot be adopted", where);
        }
        if (comm == abort_scope) {
            fatal(abort_scope, "An owned communicator cannot be its own abort scope", where);
        }
    }
    Communicator owner{comm, abort_scope, where};
    comm = MPI_COMM_NULL;
    return owner;
}

void Communicator::close(std::source_location where) noexcept {
    if (is_null()) {
        return;
    }
    const MPI_Comm abort_scope = view_.abort_scope();
    require_active(abort_scope, where);
    if (std::uncaught_exceptions() != 0) {
        fatal(abort_scope, "Exception unwinding with an owned communicator", where);
    }

    MPI_Comm comm = view_.native();
    check(MPI_Comm_free(&comm), abort_scope, "MPI_Comm_free", where);
    view_ = CommView{};
}

MPI_Comm Communicator::release() noexcept {
    const MPI_Comm comm = view_.native();
    view_ = CommView{};
    return comm;
}

void Communicator::swap(Communicator& other) noexcept {
    std::swap(view_, other.view_);
    std::swap(origin_, other.origin_);
}

} // namespace krona::mpi
