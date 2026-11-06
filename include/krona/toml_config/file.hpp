#pragma once

#include <krona/mpi/collectives.hpp>
#include <krona/mpi/communicator.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>

namespace krona::toml_config {
namespace detail {

inline void file_root(mpi::CommView comm, int master)
{
    if (comm.is_null()) mpi::fatal("Configuration file: communicator is empty");
    if (master < 0 || master >= comm.size())
        mpi::fatal("Configuration file: master rank is outside the communicator");
}

} // namespace detail

/// @brief Reads raw file text on master; other ranks return an empty string.
/// Local and blocking; file is inspected only on master.
[[nodiscard]] inline std::string read_master(const std::filesystem::path& file, mpi::CommView comm, int master = 0) noexcept
{
    return mpi::guarded([&] {
        detail::file_root(comm, master);
        std::string text;
        if (comm.rank() != master) return text;

        std::ifstream input(file, std::ios::binary);
        if (!input) mpi::fatal("Configuration file: cannot open '" + file.string() + "'");

        // Sequential reads avoid requiring a seekable input.
        char buffer[8192];
        while (input.read(buffer, sizeof(buffer)) || input.gcount() != 0)
            text.append(buffer, static_cast<std::size_t>(input.gcount()));
        if (input.bad() || !input.eof())
            mpi::fatal("Configuration file: cannot read '" + file.string() + "'");
        return text;
    });
}

/// @brief Broadcasts master's text, replacing the string on every other rank.
/// Blocking collective: all ranks supply the same communicator and master.
inline void broadcast(std::string& text, mpi::CommView comm, int master = 0) noexcept
{
    mpi::guarded([&] {
        detail::file_root(comm, master);
        std::uint64_t size = 0;
        if (comm.rank() == master) {
            if (text.size() > std::numeric_limits<std::uint64_t>::max())
                mpi::fatal("Configuration file: text size exceeds uint64_t");
            size = static_cast<std::uint64_t>(text.size());
        }

        mpi::broadcast(comm, size, master);
        if (size > text.max_size()) mpi::fatal("Configuration file: received text is too large");
        if (comm.rank() != master) text.resize(static_cast<std::size_t>(size));

        // Each MPI 3.1 call accepts at most INT_MAX characters.
        constexpr auto max_count = static_cast<std::size_t>(std::numeric_limits<int>::max());
        std::size_t offset = 0;
        while (offset < text.size()) {
            const auto count = std::min(text.size() - offset, max_count);
            mpi::broadcast(comm, std::span<char>{text.data() + offset, count}, master);
            offset += count;
        }
    });
}

/// @brief Reads on master and returns identical text on all ranks.
/// Blocking collective; the file path is used only on master.
[[nodiscard]] inline std::string read_broadcast(const std::filesystem::path& file, mpi::CommView comm, int master = 0) noexcept
{
    auto text = read_master(file, comm, master);
    broadcast(text, comm, master);
    return text;
}

} // namespace krona::toml_config
