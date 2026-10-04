#pragma once

#include <krona/mesh/localmesh.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mpi/communicator.hpp>

#include <hdf5.h>

#include <array>
#include <exception>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

#ifndef H5_HAVE_PARALLEL
#error "KRONA solver mesh I/O requires a parallel HDF5 build."
#endif

namespace krona::io::solver_mesh::detail {

using U64 = std::uint64_t;
inline constexpr U64 magic = 0x4b524f4e414d4553ULL;
inline constexpr U64 version = 2;
inline constexpr std::size_t header_size = 16;

enum Category : std::size_t {
    cells, cell_indices, nodes, faces, face_indices, peers, send_indices,
    recv_indices, patch_indices, cell_offsets, face_offsets, peer_offsets,
    patch_offsets, category_count
};
inline constexpr std::size_t row_size = category_count + 6;
enum LocalField : std::size_t {
    owned = category_count, owned_nodes, internal_faces, inner_faces, grouped, flipped
};

/// @brief Aborts the MPI job when a file or format invariant fails.
inline void require(bool condition, std::string_view message)
{
    // Failure is terminal: no extra status collective is needed.
    if (!condition) mpi::fatal(message);
}

/// @brief Checks an HDF5 operation and retains its native error stack on failure.
inline void check(herr_t status, std::string_view operation)
{
    // HDF5 prints its detailed error stack; add the KRONA operation name.
    require(status >= 0, operation);
}

/// @brief Owns one HDF5 handle and closes it with its matching H5*close function.
class Handle {
public:
    /// @brief Takes ownership of a newly created HDF5 handle.
    Handle(hid_t value, herr_t (*close_function)(hid_t)) : id_(value), close_(close_function)
    {
        // Negative IDs indicate creation or opening failure.
        require(id_ >= 0, "Cannot create or open an HDF5 object");
    }
    /// @brief Closes normally or aborts before collective cleanup during exception unwinding.
    ~Handle()
    {
        // A lone rank must not enter H5Fclose while its peers continue dataset operations.
        if (id_ >= 0 && std::uncaught_exceptions() != 0)
            mpi::fatal("Exception during collective HDF5 I/O");
        close();
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    /// @brief Transfers ownership without duplicating the HDF5 handle.
    Handle(Handle&& other) noexcept : id_(std::exchange(other.id_, -1)), close_(other.close_) {}
    /// @brief Returns the borrowed HDF5 ID.
    [[nodiscard]] hid_t get() const noexcept { return id_; }
    /// @brief Closes once and reports flush/close failures before MPI finalization.
    void close()
    {
        if (id_ >= 0) {
            // Invalidate ownership before checking the close result.
            check(close_(std::exchange(id_, -1)), "HDF5 close failed");
        }
    }
private:
    hid_t id_;
    herr_t (*close_)(hid_t);
};

/// @brief Selects the native memory datatype without changing an array's representation.
template<class T> hid_t memory_type()
{
    // CellType has an explicitly fixed uint8_t underlying representation.
    if constexpr (std::is_enum_v<T>) return memory_type<std::underlying_type_t<T>>();
    else if constexpr (std::is_same_v<T, float>) return H5T_NATIVE_FLOAT;
    else if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
    else if constexpr (std::is_signed_v<T>) {
        if constexpr (sizeof(T) == 4) return H5T_NATIVE_INT32;
        else if constexpr (sizeof(T) == 8) return H5T_NATIVE_INT64;
        else if constexpr (sizeof(T) == 2) return H5T_NATIVE_INT16;
        else return H5T_NATIVE_INT8;
    } else {
        if constexpr (sizeof(T) == 4) return H5T_NATIVE_UINT32;
        else if constexpr (sizeof(T) == 8) return H5T_NATIVE_UINT64;
        else if constexpr (sizeof(T) == 2) return H5T_NATIVE_UINT16;
        else return H5T_NATIVE_UINT8;
    }
}

/// @brief Selects a fixed-width little-endian datatype for the portable file schema.
template<class T> hid_t file_type()
{
    if constexpr (std::is_enum_v<T>) return file_type<std::underlying_type_t<T>>();
    else if constexpr (std::is_same_v<T, float>) return H5T_IEEE_F32LE;
    else if constexpr (std::is_same_v<T, double>) return H5T_IEEE_F64LE;
    else if constexpr (std::is_signed_v<T>) {
        if constexpr (sizeof(T) == 4) return H5T_STD_I32LE;
        else if constexpr (sizeof(T) == 8) return H5T_STD_I64LE;
        else if constexpr (sizeof(T) == 2) return H5T_STD_I16LE;
        else return H5T_STD_I8LE;
    } else {
        if constexpr (sizeof(T) == 4) return H5T_STD_U32LE;
        else if constexpr (sizeof(T) == 8) return H5T_STD_U64LE;
        else if constexpr (sizeof(T) == 2) return H5T_STD_U16LE;
        else return H5T_STD_U8LE;
    }
}

/// @brief Opens a parallel file access property list on the caller's communicator.
inline Handle file_access(mpi::CommView comm)
{
    require(mpi::is_active() && !comm.is_null(), "HDF5 I/O requires an active communicator");
    mpi::require_main_thread();
    Handle access{H5Pcreate(H5P_FILE_ACCESS), H5Pclose};
    check(H5Pset_fapl_mpio(access.get(), comm.native(), MPI_INFO_NULL), "H5Pset_fapl_mpio");
    // Every participant follows the same metadata operation sequence.
    check(H5Pset_all_coll_metadata_ops(access.get(), 1), "H5Pset_all_coll_metadata_ops");
    check(H5Pset_coll_metadata_write(access.get(), 1), "H5Pset_coll_metadata_write");
    return access;
}

/// @brief Creates one transfer property list reused by all collective dataset operations.
inline Handle collective_transfer()
{
    Handle transfer{H5Pcreate(H5P_DATASET_XFER), H5Pclose};
    // Empty ranks still call H5Dread/H5Dwrite with a none selection.
    check(H5Pset_dxpl_mpio(transfer.get(), H5FD_MPIO_COLLECTIVE), "H5Pset_dxpl_mpio");
    return transfer;
}

/// @brief Checks the replicated partition permutation and builds its inverse.
inline std::vector<int> inverse_mapping(std::span<const int> part2rank, int size)
{
    require(part2rank.size() == static_cast<std::size_t>(size), "Invalid partition mapping size");
    std::vector<int> rank2part(static_cast<std::size_t>(size), -1);
    for (int part = 0; part < size; ++part) {
        const int rank = part2rank[static_cast<std::size_t>(part)];
        require(rank >= 0 && rank < size && rank2part[static_cast<std::size_t>(rank)] == -1,
                "Partition mapping must be a permutation of MPI ranks");
        rank2part[static_cast<std::size_t>(rank)] = part;
    }
    return rank2part;
}

/// @brief Builds per-category disk ranges from the saved partition directory.
struct Layout {
    std::array<U64, category_count> start{}, count{}, total{};
    /// @brief Accumulates partition-ordered prefixes and selects one partition's counts.
    Layout(std::span<const U64> directory, int part, int size)
    {
        require(directory.size() == static_cast<std::size_t>(size) * row_size,
                "Invalid partition directory size");
        for (int p = 0; p < size; ++p) {
            for (std::size_t c = 0; c < category_count; ++c) {
                const U64 n = directory[static_cast<std::size_t>(p) * row_size + c];
                if (p == part) { start[c] = total[c]; count[c] = n; }
                require(n <= std::numeric_limits<U64>::max() - total[c], "File layout overflow");
                total[c] += n;
            }
        }
    }
};

/// @brief Visits all persistent SoA arrays in a fixed collective operation order.
template<class Mesh, class Visitor> void visit_arrays(Mesh& m, Visitor&& visit)
{
    // The category controls a dataset's partition prefix, not its element datatype.
    visit("cells/type", m.cell_type, cells);
    visit("cells/gid", m.cell_gid, cells);
    visit("cells/donor", m.cell_donor, cells);
    visit("cells/offsets", m.cell_nodes_offsets, cell_offsets);
    visit("cells/nodes", m.cell_nodes, cell_indices);
    visit("cells/centroid_x", m.cell_centroid_x, cells);
    visit("cells/centroid_y", m.cell_centroid_y, cells);
    visit("cells/centroid_z", m.cell_centroid_z, cells);
    visit("cells/volume", m.cell_volume, cells);
    visit("nodes/gid", m.node_gid, nodes);
    visit("nodes/x", m.node_x, nodes);
    visit("nodes/y", m.node_y, nodes);
    visit("nodes/z", m.node_z, nodes);
    visit("faces/owner", m.face_owner, faces);
    visit("faces/neighbour", m.face_neigh, faces);
    visit("faces/type", m.face_type, faces);
    visit("faces/patch", m.face_patch, faces);
    visit("faces/local_index", m.face_local_index, faces);
    visit("faces/offsets", m.face_nodes_offsets, face_offsets);
    visit("faces/nodes", m.face_nodes, face_indices);
    visit("faces/centroid_x", m.face_centroid_x, faces);
    visit("faces/centroid_y", m.face_centroid_y, faces);
    visit("faces/centroid_z", m.face_centroid_z, faces);
    visit("faces/normal_x", m.face_normal_x, faces);
    visit("faces/normal_y", m.face_normal_y, faces);
    visit("faces/normal_z", m.face_normal_z, faces);
    visit("faces/area", m.face_area, faces);
    visit("halo/ranks", m.nb_ranks, peers);
    visit("halo/send_offsets", m.send_offsets, peer_offsets);
    visit("halo/recv_offsets", m.recv_offsets, peer_offsets);
    visit("halo/send_owned", m.send_owned_local, send_indices);
    visit("halo/recv_ghost", m.recv_ghost_local, recv_indices);
    visit("patches/offsets", m.patch_face_offsets, patch_offsets);
    visit("patches/faces", m.patch_faces, patch_indices);
}

/// @brief Transfers one one-dimensional slice, including zero-length selections.
template<class T> void transfer(hid_t dataset, hid_t properties, U64 start, U64 count,
                                T* data, bool writing)
{
    Handle disk{H5Dget_space(dataset), H5Sclose};
    const hsize_t extent = count == 0 ? 1 : count;
    Handle memory{H5Screate_simple(1, &extent, nullptr), H5Sclose};
    T dummy{};
    if (count == 0) {
        // Avoid invalid zero-length hyperslabs while preserving collective participation.
        check(H5Sselect_none(disk.get()), "H5Sselect_none(file)");
        check(H5Sselect_none(memory.get()), "H5Sselect_none(memory)");
        data = &dummy;
    } else {
        const hsize_t first = start, length = count;
        check(H5Sselect_hyperslab(disk.get(), H5S_SELECT_SET, &first, nullptr, &length, nullptr),
              "H5Sselect_hyperslab");
    }
    const auto type = memory_type<std::remove_const_t<T>>();
    check(writing ? H5Dwrite(dataset, type, memory.get(), disk.get(), properties, data)
                  : H5Dread(dataset, type, memory.get(), disk.get(), properties, data),
          writing ? "H5Dwrite" : "H5Dread");
}

/// @brief Creates an uncompressed portable dataset and writes this rank's non-overlapping slice.
template<class T> void write(hid_t file, hid_t properties, const char* name,
                            U64 total, U64 start, std::span<const T> values,
                            bool offsets = false)
{
    const hsize_t extent = total;
    Handle space{H5Screate_simple(1, &extent, nullptr), H5Sclose};
    // CSR offsets are UInt64 on disk even when the host uses a smaller size_t.
    Handle dataset{H5Dcreate2(file, name, offsets ? H5T_STD_U64LE : file_type<T>(), space.get(),
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose};
    // A globally empty dataset has no allocated address in older parallel HDF5 versions.
    // All ranks know the same total, so they can skip its payload call together.
    if (total != 0)
        transfer(dataset.get(), properties, start, values.size(), const_cast<T*>(values.data()), true);
}

/// @brief Reads a dataset only after checking its extent and primitive datatype.
template<class T, std::size_t Extent> void read(hid_t file, hid_t properties, const char* name,
                           U64 total, U64 start, std::span<T, Extent> values, bool offsets = false)
{
    Handle dataset{H5Dopen2(file, name, H5P_DEFAULT), H5Dclose};
    Handle space{H5Dget_space(dataset.get()), H5Sclose};
    hsize_t extent = 0;
    require(H5Sget_simple_extent_ndims(space.get()) == 1, "Invalid HDF5 dataset rank");
    check(H5Sget_simple_extent_dims(space.get(), &extent, nullptr), "H5Sget_simple_extent_dims");
    require(extent == total && start <= total && values.size() <= total - start,
            std::string{"Invalid dataset extent: "} + name);
    Handle type{H5Dget_type(dataset.get()), H5Tclose};
    const auto target = offsets ? H5T_STD_U64LE : file_type<T>();
    require(H5Tget_class(type.get()) == H5Tget_class(target), "Invalid dataset datatype class");
    if (H5Tget_class(target) == H5T_INTEGER)
        require(H5Tget_sign(type.get()) == H5Tget_sign(target), "Invalid dataset integer signedness");
    require(H5Tget_size(type.get()) <= (offsets ? 8 : sizeof(T)), "Unsafe dataset type narrowing");
    // HDF5 performs permitted endian conversion and datatype widening directly into the output.
    if (total != 0)
        transfer(dataset.get(), properties, start, values.size(), values.data(), false);
}

/// @brief Writes replicated metadata once; every other rank contributes an empty selection.
template<class T> void write_shared(hid_t file, hid_t properties, const char* name,
                                   std::span<const T> values, int rank)
{
    // Dataset creation remains collective, without overlapping payload writes.
    write(file, properties, name, values.size(), 0, rank == 0 ? values : values.first(0));
}

/// @brief Resizes an output array after checking its allocation limit.
template<class T> void resize(std::vector<T>& values, U64 count)
{
    require(count <= values.max_size(), "HDF5 array exceeds the host allocation range");
    // Storage is final MeshPart storage; no payload staging array is allocated.
    values.resize(static_cast<std::size_t>(count));
}

/// @brief Converts a file count to a representable local mesh count.
inline LocalIndex local_count(U64 value)
{
    require(value <= static_cast<U64>(std::numeric_limits<LocalIndex>::max()),
            "HDF5 mesh count exceeds LocalIndex");
    return static_cast<LocalIndex>(value);
}

} // namespace krona::io::solver_mesh::detail
