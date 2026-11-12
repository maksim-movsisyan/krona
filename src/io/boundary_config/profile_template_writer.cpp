#include <krona/io/boundary_config/profile_template_writer.hpp>

#include <krona/mpi/collectives.hpp>
#include <krona/mpi/error.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace krona::io::boundary_config {
namespace {

struct FaceRow {
    std::size_t patch;
    GlobalIndex owner_cell_gid;
    std::uint8_t local_facet;
    std::array<GeometryReal, 3> centre;
};

void write_templates(const mesh::host::MeshPart& mesh, const std::filesystem::path& directory,
                     std::vector<FaceRow>& rows) {
    mpi::require(!directory.empty(), "Patch profile template directory is empty");
    std::set<std::string_view> names;
    for (const auto& patch : mesh.patches) {
        const auto& name = patch.name;
        mpi::require(!name.empty() && name != "." && name != ".." &&
                     name.find_first_of("/\\") == std::string::npos && name.find('\0') == std::string::npos &&
                     std::filesystem::path{name + ".csv"}.filename() == name + ".csv",
                     "Patch name cannot be used as a CSV filename");
        mpi::require(names.insert(name).second, "Duplicate patch name for CSV export");
    }

    // Keep each patch contiguous and order its rows independently of the MPI partition.
    std::ranges::sort(rows, {}, [](const FaceRow& row) {
        return std::tie(row.patch, row.owner_cell_gid, row.local_facet);
    });
    std::filesystem::create_directories(directory);
    auto row = rows.cbegin();
    for (std::size_t patch = 0; patch < mesh.patches.size(); ++patch) {
        const auto path = directory / (mesh.patches[patch].name + ".csv");
        std::ofstream out{path, std::ios::out | std::ios::trunc};
        if (!out) mpi::fatal("Cannot open patch profile template: " + path.string());
        out.imbue(std::locale::classic());
        out << std::setprecision(std::numeric_limits<GeometryReal>::max_digits10);
        out << "owner_cell_gid,local_facet,x,y,z,value\n";
        for (; row != rows.cend() && row->patch == patch; ++row) {
            const auto& [x, y, z] = row->centre;
            out << row->owner_cell_gid << ',' << static_cast<unsigned>(row->local_facet) << ','
                << x << ',' << y << ',' << z << ",\n";
        }
        out.close();
        if (!out) mpi::fatal("Cannot write patch profile template: " + path.string());
    }
}

} // namespace

void generate_patch_profile_templates(const mesh::host::MeshPart& mesh,
                                      const std::filesystem::path& directory, mpi::CommView comm) {
    mpi::guarded([&] {
        mpi::require(!comm.is_null() && mesh.rank == comm.rank() && mesh.nprocs == comm.size(),
                     "Patch profile export requires the mesh communicator");
        const auto faces = mesh.face_owner.size();
        mpi::require(mesh.face_centroid_x.size() == faces && mesh.face_centroid_y.size() == faces &&
                     mesh.face_centroid_z.size() == faces, "Patch profile export requires face geometry");
        mpi::require(std::in_range<int>(mesh.patch_faces.size()), "Too many local faces for MPI_Gatherv");

        std::vector<FaceRow> local;
        local.reserve(mesh.patch_faces.size());
        for (std::size_t patch = 0; patch < mesh.patches.size(); ++patch) {
            const std::size_t end = mesh.patch_face_offsets[patch + 1];
            for (std::size_t i = mesh.patch_face_offsets[patch]; i < end; ++i) {
                const std::size_t face = static_cast<std::size_t>(mesh.patch_faces[i]);
                const std::size_t owner = static_cast<std::size_t>(mesh.face_owner[face]);
                local.push_back({patch, mesh.cell_gid[owner], mesh.face_local_index[face],
                                 {mesh.face_centroid_x[face], mesh.face_centroid_y[face],
                                  mesh.face_centroid_z[face]}});
            }
        }

        const bool root = comm.rank() == 0;
        const int local_count = static_cast<int>(local.size());
        std::vector<int> counts(root ? static_cast<std::size_t>(comm.size()) : 0);
        mpi::check(MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm.native()),
                   "MPI_Gather(patch profile counts)");

        // Only the writer rank allocates the global payload and gather displacements.
        std::vector<int> displacements(counts.size());
        int total = 0;
        for (std::size_t rank = 0; rank < counts.size(); ++rank) {
            mpi::require(counts[rank] <= std::numeric_limits<int>::max() - total,
                         "Too many global faces for MPI_Gatherv");
            displacements[rank] = total;
            total += counts[rank];
        }
        std::vector<FaceRow> rows(static_cast<std::size_t>(total));
        const mpi::RecordDatatype<FaceRow> record;
        const auto type = record.native();
        mpi::check(MPI_Gatherv(local.data(), local_count, type, rows.data(), counts.data(),
                   displacements.data(), type, 0, comm.native()),
                   "MPI_Gatherv(patch profile rows)");
        if (root) write_templates(mesh, directory, rows);
        mpi::barrier(comm);
    });
}

} // namespace krona::io::boundary_config
