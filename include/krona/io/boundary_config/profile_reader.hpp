#pragma once

#include <krona/boundary/tabulated_profile.hpp>
#include <krona/mpi/communicator.hpp>

#include <filesystem>

namespace krona::io::boundary_config {

/// @brief Reads a numeric CSV on rank 0 and broadcasts the profile to every rank in comm.
/// The header contains value and one to three distinct axes from x, y, z in any order.
/// Rows can be unordered and must cover every grid point exactly once. Each axis needs at least
/// two coordinates. Fields are unquoted and blank lines are skipped. Invalid input aborts the job.
[[nodiscard]] boundary::profile::TabulatedProfile read_tabulated_profile_csv(
    const std::filesystem::path& path, mpi::CommView comm);

} // namespace krona::io::boundary_config
