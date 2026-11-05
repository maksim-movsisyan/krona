#pragma once
#include <krona/mpi/communicator.hpp>
#include <krona/mpi/error.hpp>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

namespace krona::apps::scalar_transport {
inline void broadcast_text(std::string& text, mpi::CommView comm)
{
    mpi::require(comm.rank() != 0 || text.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "Scalar transport: text exceeds MPI count range");
    int count = comm.rank() == 0 ? static_cast<int>(text.size()) : 0;
    mpi::check(MPI_Bcast(&count, 1, MPI_INT, 0, comm.native()), "MPI_Bcast(text length)");
    mpi::require(count >= 0, "Scalar transport: invalid text length");
    if (comm.rank() != 0) text.resize(static_cast<std::size_t>(count));
    mpi::check(MPI_Bcast(text.data(), count, MPI_CHAR, 0, comm.native()), "MPI_Bcast(text)");
}
inline std::string read_text(const std::filesystem::path& path, mpi::CommView comm)
{
    return mpi::guarded([&] {
        std::string text;
        if (comm.rank() == 0) {
            std::ifstream in(path, std::ios::binary);
            mpi::require(in.is_open(), "Scalar transport: cannot open " + path.string());
            text.assign(std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{});
            mpi::require(!in.bad(), "Scalar transport: cannot read " + path.string());
        }
        broadcast_text(text, comm);
        return text;
    });
}
} // namespace krona::apps::scalar_transport
