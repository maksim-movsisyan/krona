#include <krona/io/vtu/vtu.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mpi/communicator.hpp>
#include <krona/mpi/collectives.hpp>

#include <algorithm>
#include <bit>
#include <fstream>
#include <limits>
#include <locale>
#include <string>
#include <vector>

namespace krona::io::vtk {
namespace {

inline constexpr std::array<const char*, 10> type_names{
    "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64", "Float32", "Float64"
};
inline constexpr std::array<std::size_t, 10> type_sizes{1, 1, 2, 2, 4, 4, 8, 8, 4, 8};
inline constexpr const char* byte_order = std::endian::native == std::endian::little
                                         ? "LittleEndian" : "BigEndian";

/// @brief Escapes XML attributes, preserving quoted names and whitespace.
std::string xml(std::string_view value)
{
    std::string result;
    for (const char c : value) {
        // XML 1.0 cannot represent arbitrary control characters.
        switch (c) {
        case '&': result += "&amp;"; break;
        case '<': result += "&lt;"; break;
        case '>': result += "&gt;"; break;
        case '\"': result += "&quot;"; break;
        case '\'': result += "&apos;"; break;
        case '\n': result += "&#10;"; break;
        case '\r': result += "&#13;"; break;
        case '\t': result += "&#9;"; break;
        default:
            mpi::require(c >= 32, "VTK names cannot contain XML control characters");
            result.push_back(static_cast<char>(c));
        }
    }
    return result;
}

/// @brief Opens binary output with stable XML numeric formatting.
std::ofstream open(const std::filesystem::path& path)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    mpi::require(stream.is_open(), std::string{"Cannot create VTK file: "} + path.string());
    stream.imbue(std::locale::classic());
    return stream;
}

/// @brief Closes the stream and catches buffered write failures.
void finish(std::ofstream& stream)
{
    // Disk-full errors may be delayed until flushing or closing.
    stream.close();
    mpi::require(!stream.fail(), "Cannot finish writing a VTK file");
}

/// @brief Computes a field's raw payload size with overflow protection.
std::uint64_t payload_bytes(const Field& f)
{
    mpi::require(f.components > 0 && static_cast<std::size_t>(f.type) < type_sizes.size(),
            "Invalid VTK field type or component count");
    const auto width = type_sizes[static_cast<std::size_t>(f.type)] * static_cast<std::uint64_t>(f.components);
    mpi::require(f.tuples <= std::numeric_limits<std::uint64_t>::max() / width, "VTK payload size overflow");
    return static_cast<std::uint64_t>(f.tuples) * width;
}

/// @brief Validates borrowed field shapes and rejects duplicate array names within one association.
void validate_fields(std::span<const Field> fields, std::size_t tuples)
{
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const auto& f = fields[i];
        (void)payload_bytes(f);
        mpi::require(f.tuples == tuples && !f.name.empty() && (!f.soa || f.components == 3),
                "VTK field does not match its grid association");
        mpi::require(!f.indices || (!f.soa && f.components == 1 && f.type == scalar_type<LocalIndex>()),
                "Invalid VTK index sequence");
        if (tuples != 0 && !f.indices) {
            mpi::require(f.data[0] != nullptr && (!f.soa || (f.data[1] && f.data[2])),
                    "VTK field points to missing storage");
        }
        for (std::size_t j = 0; j < i; ++j)
            mpi::require(f.name != fields[j].name, "Duplicate VTK field name");
    }
}

/// @brief Emits one XML array declaration and advances the appended-binary offset.
void declaration(std::ofstream& out, const Field& f, std::uint64_t& offset, bool parallel = false)
{
    out << "      <" << (parallel ? "PDataArray" : "DataArray") << " type=\""
        << type_names[static_cast<std::size_t>(f.type)] << "\" Name=\"" << xml(f.name)
        << "\" NumberOfComponents=\"" << f.components << '"';
    if (!parallel) {
        out << " format=\"appended\" offset=\"" << offset << '"';
        const auto bytes = payload_bytes(f);
        mpi::require(offset <= std::numeric_limits<std::uint64_t>::max() - sizeof(std::uint64_t) &&
                bytes <= std::numeric_limits<std::uint64_t>::max() - offset - sizeof(std::uint64_t),
                "VTK appended offset overflow");
        offset += sizeof(std::uint64_t) + bytes;
    }
    out << "/>\n";
}

/// @brief Writes a large raw span in bounded streamsize chunks.
void raw(std::ofstream& out, const void* data, std::uint64_t bytes)
{
    auto* position = static_cast<const char*>(data);
    constexpr std::uint64_t chunk = 64 * 1024 * 1024;
    while (bytes != 0) {
        const auto n = std::min(bytes, chunk);
        out.write(position, static_cast<std::streamsize>(n));
        mpi::require(static_cast<bool>(out), "VTK binary write failed");
        position += n;
        bytes -= n;
    }
}

/// @brief Streams generated/interleaved numeric values through a fixed 64-KiB buffer.
template<class T, class Value> void generated(std::ofstream& out, std::size_t count, Value&& value)
{
    std::array<T, 65536 / sizeof(T)> buffer{};
    for (std::size_t first = 0; first < count;) {
        const auto n = std::min(buffer.size(), count - first);
        for (std::size_t i = 0; i < n; ++i) buffer[i] = value(first + i);
        raw(out, buffer.data(), n * sizeof(T));
        first += n;
    }
}

/// @brief Writes a typed field directly or streams generated IDs and SoA/constant values.
template<class T> void field_values(std::ofstream& out, const Field& f)
{
    const auto components = static_cast<std::size_t>(f.components);
    const auto* values = static_cast<const T*>(f.data[0]);
    if (f.indices) {
        // Local IDs follow the current mesh order and need no cell-sized scratch array.
        generated<T>(out, f.tuples, [&](std::size_t i) { return static_cast<T>(f.first_tuple + i); });
    } else if (!f.soa && f.stride == components) {
        // Contiguous scalar/vector arrays need no numeric conversion or packing.
        if (f.tuples != 0) raw(out, values + f.first_tuple * components, payload_bytes(f));
    } else {
        mpi::require(f.tuples <= std::numeric_limits<std::size_t>::max() / components, "VTK field index overflow");
        generated<T>(out, f.tuples * components, [&](std::size_t i) {
            const auto tuple = f.first_tuple + i / components, component = i % components;
            const auto* column = static_cast<const T*>(f.data[f.soa ? component : 0]);
            return column[tuple * f.stride + (f.soa ? 0 : component)];
        });
    }
}

/// @brief Writes a UInt64 block header followed by the field's native-endian bytes.
void field_block(std::ofstream& out, const Field& f)
{
    const auto bytes = payload_bytes(f);
    raw(out, &bytes, sizeof(bytes));
    switch (f.type) {
    case ScalarType::i8: field_values<std::int8_t>(out, f); break;
    case ScalarType::u8: field_values<std::uint8_t>(out, f); break;
    case ScalarType::i16: field_values<std::int16_t>(out, f); break;
    case ScalarType::u16: field_values<std::uint16_t>(out, f); break;
    case ScalarType::i32: field_values<std::int32_t>(out, f); break;
    case ScalarType::u32: field_values<std::uint32_t>(out, f); break;
    case ScalarType::i64: field_values<std::int64_t>(out, f); break;
    case ScalarType::u64: field_values<std::uint64_t>(out, f); break;
    case ScalarType::f32: field_values<float>(out, f); break;
    case ScalarType::f64: field_values<double>(out, f); break;
    }
}

/// @brief Converts a supported linear CGNS cell type to its VTK cell number.
std::uint8_t vtk_type(mesh::CellType type)
{
    // These linear element families use the same vertex positions in CGNS and VTK.
    constexpr std::array<std::uint8_t, 8> types{10, 14, 13, 12, 5, 9, 3, 1};
    const auto index = static_cast<std::size_t>(type);
    mpi::require(index < types.size(), "Unsupported VTK cell type");
    return types[index];
}

/// @brief Returns a deterministic piece filename without imposing a five-digit rank limit.
std::string piece_name(std::string_view stem, int rank)
{
    return std::string{stem} + "_" + std::to_string(rank) + ".vtu";
}

/// @brief Creates the output directory once and synchronizes before per-rank file creation.
void prepare_directory(const std::filesystem::path& directory, std::string_view stem, mpi::CommView comm)
{
    mpi::require(mpi::is_active() && !comm.is_null(), "Parallel VTK export requires active MPI");
    mpi::require_main_thread();
    mpi::require(!directory.empty() && !stem.empty() && stem.find_first_of("/\\") == std::string_view::npos &&
            stem != "." && stem != "..", "Invalid VTK directory or filename stem");
    if (comm.rank() == 0) {
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        mpi::require(!error, std::string{"Cannot create VTK directory: "} + error.message());
    }
    // This barrier coordinates filesystem creation, not mesh validation.
    mpi::barrier(comm);
}

/// @brief Writes a PVTU index referencing all completed pieces with a matching field schema.
void parallel_index(const std::filesystem::path& directory, std::string_view stem,
                    int ranks, Fields fields)
{
    auto out = open(directory / (std::string{stem} + ".pvtu"));
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\""
        << byte_order << "\" header_type=\"UInt64\">\n  <PUnstructuredGrid GhostLevel=\"0\">\n";
    std::uint64_t unused = 0;
    out << "    <PPointData>\n";
    for (const auto& f : fields.point) declaration(out, f, unused, true);
    out << "    </PPointData>\n    <PCellData>\n";
    for (const auto& f : fields.cell) declaration(out, f, unused, true);
    out << "    </PCellData>\n    <PPoints>\n";
    const Field points{"Points", scalar_type<GeometryReal>(), {}, 0, 0, 1, 3, true};
    declaration(out, points, unused, true);
    out << "    </PPoints>\n";
    for (int rank = 0; rank < ranks; ++rank)
        out << "    <Piece Source=\"" << xml(piece_name(stem, rank)) << "\"/>\n";
    out << "  </PUnstructuredGrid>\n</VTKFile>\n";
    finish(out);
}

/// @brief Appends borrowed fields, trimming a full mesh association to the exported subset.
void append_fields(std::vector<Field>& target, std::span<const Field> extra,
                   std::size_t full_count, std::size_t first, std::size_t count)
{
    for (const auto& f : extra) {
        mpi::require(f.tuples == full_count || f.tuples == count, "MeshPart VTK field has the wrong length");
        target.push_back(f.tuples == full_count ? f.slice(first, count) : f);
    }
}

} // namespace

/// @brief Describes a representable LocalIndex sequence for buffered binary output.
Field index_field(std::string_view name, std::size_t tuples, std::size_t first)
{
    // Check the last generated ID once, without scanning or allocating a sequence.
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<LocalIndex>::max());
    mpi::require(first <= limit && (tuples == 0 || tuples - 1 <= limit - first),
            "VTK local ID exceeds LocalIndex");
    return {name, scalar_type<LocalIndex>(), {}, tuples, first, 1, 1, false, true};
}

/// @brief Selects a valid tuple interval while retaining the original borrowed storage.
Field Field::slice(std::size_t first, std::size_t count) const
{
    mpi::require(first <= tuples && count <= tuples - first, "VTK field slice is out of range");
    Field result = *this;
    // Keep the storage borrowed; only the tuple interval changes.
    result.first_tuple += first;
    result.tuples = count;
    return result;
}

/// @brief Writes topology and fields as appended raw binary with UInt64 block lengths.
void write_vtu_piece(GridView grid, const std::filesystem::path& path, Fields fields)
{
    const auto points = grid.x.size(), cells = grid.types.size();
    mpi::require(grid.y.size() == points && grid.z.size() == points && grid.offsets.size() == cells + 1 &&
            grid.offsets.front() <= grid.offsets.back() && grid.offsets.back() <= grid.connectivity.size(),
            "Invalid VTK grid view");
    validate_fields(fields.point, points);
    validate_fields(fields.cell, cells);
    const auto first = grid.offsets.front();
    const auto connectivity = grid.connectivity.subspan(first, grid.offsets.back() - first);
    const auto coordinates = vector_field("Points", grid.x, grid.y, grid.z);
    const auto node_ids = field("connectivity", connectivity);
    const Field ends{"offsets", ScalarType::u64, {}, cells};
    const Field types{"types", ScalarType::u8, {}, cells};
    auto out = open(path);
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\""
        << byte_order << "\" header_type=\"UInt64\">\n  <UnstructuredGrid>\n    <Piece NumberOfPoints=\""
        << points << "\" NumberOfCells=\"" << cells << "\">\n      <PointData>\n";
    std::uint64_t offset = 0;
    for (const auto& f : fields.point) declaration(out, f, offset);
    out << "      </PointData>\n      <CellData>\n";
    for (const auto& f : fields.cell) declaration(out, f, offset);
    out << "      </CellData>\n      <Points>\n";
    declaration(out, coordinates, offset);
    out << "      </Points>\n      <Cells>\n";
    declaration(out, node_ids, offset);
    declaration(out, ends, offset);
    declaration(out, types, offset);
    out << "      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n  <AppendedData encoding=\"raw\">\n_";

    // Stream SoA fields and points through a fixed-size buffer; contiguous arrays go directly to disk.
    for (const auto& f : fields.point) field_block(out, f);
    for (const auto& f : fields.cell) field_block(out, f);
    field_block(out, coordinates);
    field_block(out, node_ids);
    auto bytes = payload_bytes(ends);
    raw(out, &bytes, sizeof(bytes));
    generated<std::uint64_t>(out, cells, [&](std::size_t i) {
        return static_cast<std::uint64_t>(grid.offsets[i + 1] - first);
    });
    bytes = payload_bytes(types);
    raw(out, &bytes, sizeof(bytes));
    generated<std::uint8_t>(out, cells, [&](std::size_t i) { return vtk_type(grid.types[i]); });
    out << "\n  </AppendedData>\n</VTKFile>\n";
    finish(out);
}

/// @brief Coordinates independent piece writes and publishes the parallel index after completion.
void write_parallel_vtu(GridView grid, const std::filesystem::path& directory,
                        std::string_view stem, mpi::CommView comm, Fields fields)
{
    prepare_directory(directory, stem, comm);
    write_vtu_piece(grid, directory / piece_name(stem, comm.rank()), fields);
    // Rank zero publishes an index only after all referenced files are closed.
    mpi::barrier(comm);
    if (comm.rank() == 0) parallel_index(directory, stem, comm.size(), fields);
}

/// @brief Exports owned cells and boundary faces using borrowed contiguous views of MeshPart arrays.
void write_vtu(const mesh::host::MeshPart& m, const std::filesystem::path& directory,
               std::string_view stem, mpi::CommView comm, MeshFields extra)
{
    mpi::require(m.faces_grouped && m.rank == comm.rank() && m.nprocs == comm.size(),
            "MeshPart VTK export requires grouped faces and the matching communicator");
    const auto nodes = static_cast<std::size_t>(m.n_nodes_own);
    const auto owned = static_cast<std::size_t>(m.n_own);
    const auto boundary_begin = static_cast<std::size_t>(m.n_inner_faces);
    const auto boundary_count = static_cast<std::size_t>(m.n_faces) - boundary_begin;
    const auto x = std::span{m.node_x}.first(nodes);
    const auto y = std::span{m.node_y}.first(nodes);
    const auto z = std::span{m.node_z}.first(nodes);
    const GridView volume{x, y, z, std::span{m.cell_nodes_offsets}.first(owned + 1),
                          m.cell_nodes, std::span{m.cell_type}.first(owned)};
    const GridView boundary{x, y, z, std::span{m.face_nodes_offsets}.subspan(boundary_begin),
                            m.face_nodes, std::span{m.face_type}.subspan(boundary_begin)};

    // Node ownership here means used by owned cells; duplicated points across ranks are intentional.
    std::vector<Field> point{field("global_node_id", std::span{m.node_gid}.first(nodes))};
    append_fields(point, extra.point, static_cast<std::size_t>(m.n_nodes), 0, nodes);
    // Emit the same field schema on empty ranks as on non-empty ranks.
    std::vector<Field> cell{constant_field("rank", m.rank, owned),
                            field("global_id", std::span{m.cell_gid}.first(owned)),
                            index_field("local_id", owned),
                            field("volume", std::span{m.cell_volume}.first(owned))};
    append_fields(cell, extra.cell, static_cast<std::size_t>(m.n_cells), 0, owned);
    std::vector<Field> surface{constant_field("rank", m.rank, boundary_count),
        // Boundary IDs retain their indices in the full MeshPart face array.
        index_field("local_id", boundary_count, boundary_begin),
        field("patch_id", std::span{m.face_patch}.subspan(boundary_begin)),
        field("area", std::span{m.face_area}.subspan(boundary_begin)),
        vector_field("normal", std::span{m.face_normal_x}.subspan(boundary_begin),
                     std::span{m.face_normal_y}.subspan(boundary_begin),
                     std::span{m.face_normal_z}.subspan(boundary_begin))};
    append_fields(surface, extra.boundary, static_cast<std::size_t>(m.n_faces), boundary_begin, boundary_count);

    // Both datasets share just two filesystem barriers; no mesh data is gathered on rank zero.
    prepare_directory(directory, stem, comm);
    const std::string boundary_stem = std::string{stem} + "_bnd";
    write_vtu_piece(volume, directory / piece_name(stem, comm.rank()), {point, cell});
    write_vtu_piece(boundary, directory / piece_name(boundary_stem, comm.rank()), {point, surface});
    mpi::barrier(comm);
    if (comm.rank() == 0) {
        parallel_index(directory, stem, comm.size(), {point, cell});
        parallel_index(directory, boundary_stem, comm.size(), {point, surface});
    }
}

} // namespace krona::io::vtk