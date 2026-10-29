#include <krona/io/cgns/reader.hpp>
#include <krona/mpi/collectives.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <cgnslib.h>
#include <cgns_io.h>
#include <pcgnslib.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace krona::io::cgns {
namespace {

using ElementType = CGNS_ENUMT(ElementType_t);
using PointSetType = CGNS_ENUMT(PointSetType_t);
using DataType = CGNS_ENUMT(DataType_t);
using GridLocation = CGNS_ENUMT(GridLocation_t);
using Block = distribution::BlockDistribution<GlobalIndex>;
using WideBlock = distribution::BlockDistribution<std::int64_t>;
static_assert(sizeof(cgsize_t) == 8, "Build CGNS with 64-bit cgsize_t for large meshes");
static_assert(std::is_signed_v<GlobalIndex> && std::is_signed_v<LocalIndex>);
static_assert(std::is_same_v<GeometryReal, float> || std::is_same_v<GeometryReal, double>);

/// @brief Reports a failed CGNS call without allocating an error string.
void check_cgns(int code, const char* operation,
                std::source_location where = std::source_location::current()) noexcept
{
    if (code == CG_OK) return;
    // Preserve the library's explanation before terminating WORLD.
    char message[1024]{};
    std::snprintf(message, sizeof(message), "%s: %s", operation, cg_get_error());
    mpi::fatal(message, where);
}

/// @brief Reports a failed low-level CGIO operation.
void check_cgio(int code, const char* operation,
                std::source_location where = std::source_location::current()) noexcept
{
    if (code == 0) return;
    // CGIO has its own error reporting API.
    char error[81]{}, message[1024]{};
    cgio_error_message(error);
    std::snprintf(message, sizeof(message), "%s: CGIO error %d: %s", operation, code, error);
    mpi::fatal(message, where);
}

/// @brief Converts a source count or ID to the project's global index type.
[[nodiscard]] GlobalIndex global_index(cgsize_t value)
{
    // Narrow global indices remain an explicit project configuration choice.
    mpi::require(value >= 0 && std::in_range<GlobalIndex>(value),
           "CGNS index exceeds GlobalIndex; use 64-bit global indices");
    return static_cast<GlobalIndex>(value);
}

/// @brief Converts an allocation size without wrapping size_t.
[[nodiscard]] std::size_t host_size(std::int64_t value)
{
    mpi::require(value >= 0 && std::in_range<std::size_t>(value), "Import size exceeds host address space");
    return static_cast<std::size_t>(value);
}

/// @brief Checks that owned entity ordinals fit LocalIndex.
void check_local_size(GlobalIndex count)
{
    // Check once at a distribution boundary, not in element loops.
    mpi::require(std::in_range<LocalIndex>(count),
           "Owned entity count exceeds LocalIndex; use more ranks or 64-bit local indices");
}

/// @brief Checks legacy section types without reading their connectivity.
void check_legacy_sections(int file, double parent, int version, std::size_t level = 0)
{
    constexpr std::array<std::string_view, 3> labels{"CGNSBase_t", "Zone_t", "Elements_t"};
    int children = 0;
    check_cgio(cgio_number_children(file, parent, &children), "Count CGNS probe children");
    std::array<double, 64> ids{};
    for (int first = 0; first < children;) {
        const int count = std::min(static_cast<int>(ids.size()), children - first);
        int returned = 0;
        check_cgio(cgio_children_ids(file, parent, first + 1, count, &returned, ids.data()),
                   "Read CGNS probe children");
        assert(returned == count);
        for (int i = 0; i < returned; ++i) {
            char label[CGIO_MAX_LABEL_LENGTH + 1]{};
            const auto id = ids[static_cast<std::size_t>(i)];
            check_cgio(cgio_get_label(file, id, label), "Read CGNS probe label");
            if (label != labels[level]) continue;
            if (level + 1 < labels.size()) {
                // cgp_open parses every base and zone, including ones we will not import.
                check_legacy_sections(file, id, version, level + 1);
                continue;
            }
            int dimensions = 0;
            std::array<cgsize_t, CGIO_MAX_DIMENSIONS> shape{};
            check_cgio(cgio_get_dimensions(file, id, &dimensions, shape.data()),
                       "Read Elements_t header dimensions");
            mpi::require(dimensions == 1 && shape[0] == 2, "Invalid Elements_t header");
            std::array<int, 2> header{};
            check_cgio(cgio_read_all_data_type(file, id, "I4", header.data()),
                       "Read Elements_t header");
            auto type = header[0];
            // Match CGNS's enum renumbering for files written with version 3.0.
            if (version >= 3000 && version < 3100 &&
                type > CGNS_ENUMV(PYRA_5) && type < CGNS_ENUMV(NGON_n))
                type = type == CGNS_ENUMV(PYRA_14) ? CGNS_ENUMV(PYRA_13) : type - 1;
            mpi::require(type != CGNS_ENUMV(MIXED) && type != CGNS_ENUMV(NGON_n) &&
                       type != CGNS_ENUMV(NFACE_n),
                   "Update legacy CGNS to 4.x first: implicit MIXED/NGON_n/NFACE_n "
                   "conversion violates distributed import");
        }
        first += count;
    }
}

/// @brief Probes the file version and rejects only legacy bulk-conversion paths.
void check_source_format(const std::string& path)
{
    // Read the version first; inspect section headers only for legacy formats.
    int probe = -1;
    double root = 0, version_id = 0;
    check_cgio(cgio_open_file(path.c_str(), CGIO_MODE_READ, CGIO_FILE_HDF5, &probe),
               "Open CGNS format probe");
    check_cgio(cgio_get_root_id(probe, &root), "cgio_get_root_id");
    check_cgio(cgio_get_node_id(probe, root, "CGNSLibraryVersion", &version_id),
               "Find CGNSLibraryVersion");
    int dimensions = 0;
    std::array<cgsize_t, CGIO_MAX_DIMENSIONS> shape{};
    check_cgio(cgio_get_dimensions(probe, version_id, &dimensions, shape.data()),
               "CGNSLibraryVersion dimensions");
    mpi::require(dimensions == 1 && shape[0] == 1, "Invalid CGNSLibraryVersion array");
    float version = 0;
    check_cgio(cgio_read_all_data_type(probe, version_id, "R4", &version),
               "Read CGNSLibraryVersion");
    mpi::require(std::isfinite(version) && version > 0, "Invalid CGNSLibraryVersion value");
    if (version < 4.0F) {
        const int source_version = static_cast<int>(version * 1000.0F + 0.5F);
        // CGNS 3.4 already uses modern offsets; uniform legacy sections need no bulk conversion.
        if (source_version != 3400) check_legacy_sections(probe, root, source_version);
    }
    check_cgio(cgio_close_file(probe), "Close CGNS format probe");
}

/// @brief Owns the open file until its explicit collective close.
struct File final 
{
    mpi::CommView comm;
    int id = -1;
    int cgio = -1;

    /// @brief Opens an HDF5-backed CGNS file on the supplied communicator.
    File(const std::string& path, mpi::CommView view, const ReaderOptions& options) : comm(view)
    {
        check_source_format(path);
        // CGIO typed reads below inherit this parallel transfer mode.
        check_cgns(cgp_mpi_comm(comm.native()), "cgp_mpi_comm");
        check_cgns(cgp_mpi_info(options.io_info), "cgp_mpi_info");
        check_cgns(cgp_pio_mode(options.collective_io ? CGP_COLLECTIVE : CGP_INDEPENDENT), "cgp_pio_mode");
        check_cgns(cgp_open(path.c_str(), CG_MODE_READ, &id), "cgp_open");
        check_cgns(cg_get_cgio(id, &cgio), "cg_get_cgio");
        int type = 0;
        check_cgns(cg_get_file_type(id, &type), "cg_get_file_type");
        mpi::require(type == CG_FILE_HDF5, "Parallel import requires HDF5-backed CGNS");
    }

    /// @brief Prevents accidental implicit collective close during exception unwinding.
    ~File() noexcept
    {
        // read_parallel catches exceptions while this file is still alive.
        if (id >= 0) mpi::fatal("CGNS file escaped its explicit collective close");
    }

    /// @brief An open parallel file cannot be copied.
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    /// @brief Closes the file collectively and releases borrowed CGNS configuration.
    void close()
    {
        check_cgns(cgp_close(id), "cgp_close");
        id = -1;
        // Do not leave process-global CGNS state referring to a borrowed communicator/info.
        check_cgns(cgp_mpi_comm(MPI_COMM_WORLD), "Reset CGNS communicator");
        check_cgns(cgp_mpi_info(MPI_INFO_NULL), "Reset CGNS MPI info");
        check_cgns(cgp_pio_mode(CGP_COLLECTIVE), "Reset CGNS IO mode");
    }

    /// @brief Resolves one named child in the file's CGIO tree.
    [[nodiscard]] double child(double parent, const char* name) const
    {
        double node = 0;
        check_cgio(cgio_get_node_id(cgio, parent, name, &node), name);
        return node;
    }
};

struct IntegerArray 
{
    double id = 0;
    int dimensions = 0;
    std::array<cgsize_t, CGIO_MAX_DIMENSIONS> shape{};
};

/// @brief Resolves an I4/I8 connectivity, offset or point-set array.
[[nodiscard]] IntegerArray integer_array(
    const File& file, double parent, const char* name, int dimensions)
{
    IntegerArray array;
    array.id = file.child(parent, name);
    check_cgio(cgio_get_dimensions(file.cgio, array.id, &array.dimensions, array.shape.data()),
               "cgio_get_dimensions");
    mpi::require(array.dimensions == dimensions, "Unexpected CGNS integer array rank");
    char type[3]{};
    check_cgio(cgio_get_data_type(file.cgio, array.id, type), "cgio_get_data_type");
    // Inspect each array independently: I4 connectivity may have I8 MIXED offsets.
    mpi::require(std::string_view{type} == "I4" || std::string_view{type} == "I8",
           "Expected an I4/I8 CGNS integer array");
    if (dimensions == 2) mpi::require(array.shape[0] == 1, "Expected an unstructured point set");
    return array;
}

/// @brief Reads a distributed integer hyperslab with explicit memory-type conversion.
template<class Index>
void read_integer(const File& file, const IntegerArray& array, std::int64_t first,
                  std::span<Index> output)
{
    static_assert(std::is_signed_v<Index> && (sizeof(Index) == 4 || sizeof(Index) == 8));
    Index dummy{};
    if (output.empty()) {
        // CGIO lacks a null selection API: keep the HDF5 call matched with one discarded value.
        output = std::span<Index>{&dummy, 1};
        first = 0;
    }
    const auto axis = static_cast<std::size_t>(array.dimensions - 1);
    mpi::require(first >= 0 && first <= array.shape[axis] &&
               output.size() <= static_cast<std::uint64_t>(array.shape[axis] - first),
           "CGNS integer slice exceeds its array extent");
    std::array<cgsize_t, 2> start{1, 1}, end{1, 1}, stride{1, 1};
    start[axis] = static_cast<cgsize_t>(first) + 1;
    end[axis] = static_cast<cgsize_t>(first) + static_cast<cgsize_t>(output.size());
    const cgsize_t memory = static_cast<cgsize_t>(output.size()), one = 1;
    // CGNS 4.5.2's ADFH path uses the selected CGP mode and HDF5's I4/I8 conversion.
    check_cgio(cgio_read_data_type(file.cgio, array.id, start.data(), end.data(), stride.data(),
                                   sizeof(Index) == 8 ? "I8" : "I4", 1, &memory, &one,
                                   &memory, &one, output.data()), "cgio_read_data_type");
}

/// @brief Maps the supported linear CGNS types to mesh topology types.
[[nodiscard]] mesh::CellType cell_type(ElementType type)
{
    switch (type) {
    case CGNS_ENUMV(TETRA_4): return mesh::CellType::TET;
    case CGNS_ENUMV(PYRA_5):  return mesh::CellType::PYRA;
    case CGNS_ENUMV(PENTA_6): return mesh::CellType::PRISM;
    case CGNS_ENUMV(HEXA_8):  return mesh::CellType::HEXA;
    case CGNS_ENUMV(TRI_3):   return mesh::CellType::TRI;
    case CGNS_ENUMV(QUAD_4):  return mesh::CellType::QUAD;
    case CGNS_ENUMV(BAR_2):   return mesh::CellType::BAR;
    case CGNS_ENUMV(NODE):   return mesh::CellType::NODE;
    default: mpi::fatal("Unsupported CGNS element: require linear fixed-arity types");
    }
}

struct Section 
{
    GlobalIndex begin = 0, end = 0;
    GlobalIndex source_begin = 0, source_end = 0;
    ElementType type{};
    IntegerArray connectivity, offsets;
    bool volume = false, boundary = false;
    GlobalIndex boundary_offset = 0;
};
struct BoundaryCondition 
{
    IntegerArray points;
    PointSetType type{};
    std::int64_t count = 0;
    mesh::PatchId patch = mesh::invalid_patch_id;
};
struct Metadata 
{
    std::vector<Section> sections;
    std::vector<BoundaryCondition> bcs;
    std::array<int, 3> coordinate_indices{};
};
/// @brief Resolves section arrays and establishes cell and dense boundary ownership.
void read_sections(const File& file, const ReaderOptions& options, ReadResult& result,
                   Metadata& metadata)
{
    // get file index, base index and zone index
    const int f = file.id, b = options.base_index, z = options.zone_index;
    
    auto& mesh = result.mesh;
    char name[33]{};
    const auto cells = mesh.distribution.cells.global_size();
    
    // Resolve arrays once; payload reads later select only owned slices.
    double zone_id = 0;
    check_cgns(cg_zone_id(f, b, z, &zone_id), "cg_zone_id");

    // read number of sections
    int sections = 0;
    check_cgns(cg_nsections(f, b, z, &sections), "cg_nsections");
    metadata.sections.reserve(static_cast<std::size_t>(sections) * 2);

    // loop over all sections
    for (int s = 1; s <= sections; ++s)
    {
        // read section metadata
        Section section;
        cgsize_t first = 0, last = 0;
        int boundary_count = 0, parent_data = 0;
        check_cgns(cg_section_read(f, b, z, s, name, 
                                        &section.type, &first, &last, &boundary_count,
                                        &parent_data), "cg_section_read");

        // fill section metadata
        mpi::require(first >= 1 && last >= first, "Invalid CGNS section range");
        section.begin = global_index(first - 1);
        section.end = global_index(last);
        section.source_begin = section.begin;
        section.source_end = section.end;
        section.volume = section.end <= cells;

        // mixed section can contain elements of different dims (volume + boundary) ???
        mpi::require(section.volume || section.begin >= cells || section.type == CGNS_ENUMV(MIXED),
               "A section crosses the volume/boundary ID boundary");

        if (section.type == CGNS_ENUMV(MIXED)) {
            section.boundary = !section.volume; // pre-flag
        } else {
            // get element dimension
            const int dimension = mesh::dimension(cell_type(section.type));
            mpi::require(!section.volume || dimension == mesh.cell_dimension, "Volume section has the wrong dimension");
            mpi::require(section.volume || dimension < mesh.cell_dimension, "A volume section lies outside the zone cell range");
            section.boundary = dimension == mesh.cell_dimension - 1;
        }

        if (section.volume || section.boundary) {
            const double section_id = file.child(zone_id, name);
            section.connectivity = integer_array(file, section_id, "ElementConnectivity", 1);
            if (section.type == CGNS_ENUMV(MIXED)) {
                section.offsets = integer_array(file, section_id, "ElementStartOffset", 1);
                const auto element_count = last - first + 1;
                mpi::require(element_count < std::numeric_limits<cgsize_t>::max(),
                       "MIXED offset array size overflow");
                mpi::require(section.offsets.shape[0] == element_count + 1,
                       "Invalid MIXED offset array extent");
            }
            else {
                const auto arity =
                    static_cast<cgsize_t>(mesh::nodes_per_element(cell_type(section.type)));
                mpi::require(last - first + 1 <= std::numeric_limits<cgsize_t>::max() / arity,
                       "Section connectivity size overflow");
                mpi::require(section.connectivity.shape[0] == (last - first + 1) * arity,
                       "Invalid uniform connectivity extent");
            }
        }
        if (section.type == CGNS_ENUMV(MIXED) && section.begin < cells && section.end > cells) {
            auto volume = section;
            volume.end = cells;
            volume.volume = true;
            volume.boundary = false;
            metadata.sections.push_back(volume);
            section.begin = cells;
        }
        metadata.sections.push_back(section);
    }
    // Volume IDs must cover the dense CGNS cell range.
    std::sort(metadata.sections.begin(), metadata.sections.end(),
              [](const Section &left, const Section &right) { return left.begin < right.begin; });
    GlobalIndex next_cell = 0, previous_end = 0;
    GlobalIndex boundary_total = 0;

    // loop over all sections
    for (auto &section : metadata.sections) {
        mpi::require(section.begin >= previous_end, "Overlapping CGNS section ranges");
        previous_end = section.end;

        // volume section check
        if (section.volume) {
            mpi::require(section.begin == next_cell, "Volume sections must cover CGNS cell IDs without gaps");
            next_cell = section.end;
        }

        // boundary section check
        if (section.boundary) {
            section.boundary_offset = boundary_total;
            mpi::require(section.end - section.begin <= std::numeric_limits<GlobalIndex>::max() - boundary_total,
                   "Global boundary count overflow");
            boundary_total += section.end - section.begin;
        }
    }

    mpi::require(next_cell == cells, "Volume sections do not cover the zone cell range");

    // distribute boundary elements among ranks
    mesh.distribution.boundary_ordinals = Block{boundary_total, file.comm.size()};
    const std::size_t boundary_local = host_size(mesh.distribution.boundary_ordinals.local_size(file.comm.rank()));

    mpi::require(boundary_local <= static_cast<std::uint64_t>(std::numeric_limits<LocalIndex>::max()),
           "Local boundary count exceeds LocalIndex");
    mesh.boundary.global_ids.reserve(boundary_local);
    mesh.boundary.patch_ids.assign(boundary_local, mesh::invalid_patch_id);
}

/// @brief Reads patch labels and resolves supported boundary point-set arrays.
void read_patch_metadata(const File& file, const ReaderOptions& options, ReadResult& result,
                         Metadata& metadata)
{
    // get file index, base index and zone index
    const int f = file.id, b = options.base_index, z = options.zone_index;

    auto& mesh = result.mesh;
    char name[33]{};
    
    // read number of boundary zones
    int bcs = 0;
    check_cgns(cg_nbocos(f, b, z, &bcs), "cg_nbocos");
    metadata.bcs.reserve(static_cast<std::size_t>(bcs));
    mesh.patches.reserve(static_cast<std::size_t>(bcs));
    result.source.ignored_boundary_conditions.reserve(static_cast<std::size_t>(bcs));

    // loop over all boundary zones
    for (int bc = 1; bc <= bcs; ++bc)
    {
        // read boundary zone metadata
        BoundaryCondition condition;
        CGNS_ENUMT(BCType_t) type{};
        DataType normal_type{};
        cgsize_t count = 0, normal_size = 0;
        int normal_indices[3]{}, datasets = 0;
        check_cgns(cg_boco_info(f, b, z, bc, name, &type, 
                   &condition.type, &count, normal_indices,
                   &normal_size, &normal_type, &datasets), "cg_boco_info");
        GridLocation location{};
        check_cgns(cg_boco_gridlocation_read(f, b, z, bc, &location), "cg_boco_gridlocation_read");

        // check supported boundary zone
        const bool legacy_elements = condition.type == CGNS_ENUMV(ElementList) ||
                                     condition.type == CGNS_ENUMV(ElementRange);
        const bool supported = legacy_elements ||
            ((condition.type == CGNS_ENUMV(PointList) || condition.type == CGNS_ENUMV(PointRange)) &&
             ((mesh.cell_dimension == 3 && location == CGNS_ENUMV(FaceCenter)) ||
              (mesh.cell_dimension == 2 && location == CGNS_ENUMV(EdgeCenter))));
        if (!supported) {
            mpi::require(options.unsupported_boundary == UnsupportedBoundary::ignore,
                   "Unsupported BC: require PointList/PointRange at FaceCenter (3D) or EdgeCenter " "(2D); absent GridLocation means Vertex");
            result.source.ignored_boundary_conditions.emplace_back(name);
            continue;
        }

        mpi::require(count >= 0, "Negative BC point count");
        condition.count = count;
        mpi::require(mesh.patches.size() < static_cast<std::size_t>(std::numeric_limits<mesh::PatchId>::max()),
               "Too many boundary patches");

        // set patch id
        condition.patch = static_cast<mesh::PatchId>(mesh.patches.size());
        check_cgns(cg_goto(f, b, "Zone_t", z, "ZoneBC_t", 1, "BC_t", bc, "end"), "cg_goto(BC)");
        char family[33]{};
        const int family_status = cg_famname_read(family);
        if (family_status != CG_OK && family_status != CG_NODE_NOT_FOUND) {
            check_cgns(family_status, "cg_famname_read");
        }
        mesh.patches.push_back({name, family_status == CG_OK ? family : "", cg_BCTypeName(type)});

        // get bc id
        double bc_id = 0;
        check_cgns(cg_boco_id(f, b, z, bc, &bc_id), "cg_boco_id");
        const char* point_name = cg_PointSetTypeName(condition.type);
        double point_id = 0;
        const auto point_status = cgio_get_node_id(file.cgio, bc_id, point_name, &point_id);
        if (point_status != 0) {
            // CGNS may normalize legacy ElementList/Range enums without renaming a read-only file.
            point_name = condition.type == CGNS_ENUMV(PointRange) ? "ElementRange" : "ElementList";
        }
        condition.points = integer_array(file, bc_id, point_name, 2);
        mpi::require(condition.points.shape[1] == count, "BC point set extent mismatch");
        mpi::require((condition.type != CGNS_ENUMV(PointRange) &&
                condition.type != CGNS_ENUMV(ElementRange)) || count == 2,
               "PointRange must contain two endpoints");
        metadata.bcs.push_back(condition);
    }
}

/// @brief Reads small replicated metadata and establishes contiguous import ownership.
void read_metadata(const File &file, const ReaderOptions &options, ReadResult &result, Metadata &metadata)
{
    // get file index, base index and zone index
    const int f = file.id, b = options.base_index, z = options.zone_index;

    auto &mesh = result.mesh;

    // read cgns file version (can be different from library version)
    char name[33]{};
    check_cgns(cg_version(f, &result.source.version), "cg_version");

    // read base metadata
    check_cgns(cg_base_read(f, b, name, &mesh.cell_dimension, &mesh.physical_dimension),
               "cg_base_read");
    result.source.base_name = name;
    result.source.cell_dimension = mesh.cell_dimension;
    result.source.physical_dimension = mesh.physical_dimension;

    // check if mesh is 2D/3D
    mpi::require((mesh.cell_dimension == 2 || mesh.cell_dimension == 3) &&
           mesh.physical_dimension >= mesh.cell_dimension && mesh.physical_dimension <= 3,
           "Expected a 2D/3D CGNS base");

    // read zone metadata
    CGNS_ENUMT(ZoneType_t) zone_type{};
    check_cgns(cg_zone_type(f, b, z, &zone_type), "cg_zone_type");
    mpi::require(zone_type == CGNS_ENUMV(Unstructured), "Only unstructured zones are implemented");
    std::array<cgsize_t, 9> zone_size{};
    check_cgns(cg_zone_read(f, b, z, name, zone_size.data()), "cg_zone_read");
    result.source.zone_name = name;

    const GlobalIndex nodes = global_index(zone_size[0]), cells = global_index(zone_size[1]);
    mpi::require(nodes > 0 && cells > 0, "Empty CGNS zones are not supported");

    // create contiguous nodes and cells distributions
    mesh.distribution = {
        Block{cells, file.comm.size()}, 
        Block{nodes, file.comm.size()}, 
        {}, 
        file.comm.rank()};
    check_local_size(mesh.distribution.cells.local_size(file.comm.rank()));
    check_local_size(mesh.distribution.nodes.local_size(file.comm.rank()));

    // read number of cooridnate axes
    int coordinates = 0;
    check_cgns(cg_ncoords(f, b, z, &coordinates), "cg_ncoords");
    constexpr std::array<std::string_view, 3> names{"CoordinateX", "CoordinateY", "CoordinateZ"};

    // loop over all coordinate axes
    for (int c = 1; c <= coordinates; ++c)
    {
        DataType type{};
        check_cgns(cg_coord_info(f, b, z, c, &type, name), "cg_coord_info");
        for (std::size_t axis = 0; axis < names.size(); ++axis)
        {
            if (name != names[axis])
            {
                continue;
            }
            mpi::require(type == CGNS_ENUMV(RealSingle) || type == CGNS_ENUMV(RealDouble), "Expected floating-point Cartesian coordinates");
            mpi::require(metadata.coordinate_indices[axis] == 0, "Duplicate Cartesian coordinate array");
            metadata.coordinate_indices[axis] = c;
        }
    }

    for (int axis = 0; axis < mesh.physical_dimension; ++axis)
    {
        mpi::require(metadata.coordinate_indices[static_cast<std::size_t>(axis)] != 0, "Missing named Cartesian coordinate array, SEE IMPLEMENTATION");
    }

    read_sections(file, options, result, metadata);
    read_patch_metadata(file, options, result, metadata);
}

/// @brief Maps a section's owned dense boundary interval back to source IDs.
[[nodiscard]] std::pair<GlobalIndex, GlobalIndex> owned_boundary_range(
    const mesh::host::RawMesh& mesh, const Section& section)
{
    const auto& distribution = mesh.distribution.boundary_ordinals;
    const auto first = std::max(section.boundary_offset, distribution.begin(mesh.distribution.rank));
    const auto last = std::min(section.boundary_offset + section.end - section.begin,
                               distribution.end(mesh.distribution.rank));
    // Source boundary IDs may have gaps; ownership uses dense ordinals.
    if (first >= last) return {section.begin, section.begin};
    return {section.begin + first - section.boundary_offset,
            section.begin + last - section.boundary_offset};
}

/// @brief Converts a section source ID to its dense boundary ordinal.
[[nodiscard]] GlobalIndex boundary_ordinal(const Section& section, GlobalIndex source_id) noexcept
{
    return section.boundary_offset + source_id - section.begin;
}

/// @brief Computes the exact largest rank intersection from replicated block metadata.
[[nodiscard]] GlobalIndex maximum_overlap(const Block& distribution,
                                         GlobalIndex first, GlobalIndex last)
{
    if (first == last) return 0;
    const int left = distribution.owner(first), right = distribution.owner(last - 1);
    if (left == right) return last - first;
    // Interior blocks are non-increasing in size; only the first one can be largest.
    auto maximum = std::max(distribution.end(left) - first, last - distribution.begin(right));
    if (right > left + 1) maximum = std::max(maximum, distribution.local_size(left + 1));
    return maximum;
}

/// @brief Reads each Cartesian coordinate directly into its final owned array.
void read_coordinates(const File& file, const ReaderOptions& options,
                      const Metadata& metadata, mesh::host::RawMesh& mesh)
{
    const auto first = mesh.distribution.nodes.begin(file.comm.rank());
    const std::size_t n = host_size(mesh.distribution.nodes.local_size(file.comm.rank()));
    mesh.coordinates.x.resize(n);
    mesh.coordinates.y.resize(n);
    mesh.coordinates.z.assign(n, GeometryReal{0});
    std::array<std::vector<GeometryReal>*, 3> arrays{
        &mesh.coordinates.x, &mesh.coordinates.y, &mesh.coordinates.z};
    constexpr auto type = std::is_same_v<GeometryReal, float> ? CGNS_ENUMV(RealSingle) : CGNS_ENUMV(RealDouble);
    const cgsize_t start = n == 0 ? 1 : static_cast<cgsize_t>(first) + 1;
    const cgsize_t end = n == 0 ? 1 : start + static_cast<cgsize_t>(n) - 1;
    const cgsize_t one = 1, memory = n == 0 ? 1 : static_cast<cgsize_t>(n);
    for (int axis = 0; axis < mesh.physical_dimension; ++axis) {
        auto& values = *arrays[static_cast<std::size_t>(axis)];
        // PCGNS accepts nullptr as an empty HDF5 selection on ranks with no nodes.
        check_cgns(cgp_coord_general_read_data(
            file.id, options.base_index, options.zone_index, metadata.coordinate_indices[static_cast<std::size_t>(axis)],
            &start, &end, type, 1, &memory, &one, &memory, n == 0 ? nullptr : values.data()),
            "cgp_coord_general_read_data");
    }
}

struct LocalSection {
    const Section* section = nullptr;
    GlobalIndex first = 0, count = 0, maximum = 0;
    std::size_t element_begin = 0, node_begin = 0, node_count = 0;
};

/// @brief Plans exact local connectivity storage using only MIXED offset endpoints.
[[nodiscard]] std::vector<LocalSection> plan_elements(
    const File& file, const Metadata& metadata, const mesh::host::RawMesh& mesh)
{
    std::vector<LocalSection> slices;
    slices.reserve(metadata.sections.size());
    std::array<std::size_t, 2> elements{}, nodes{};
    for (const auto& section : metadata.sections) {
        if (!section.volume && !section.boundary) continue;
        LocalSection slice;
        slice.section = &section;
        const auto& ownership = section.volume ? mesh.distribution.cells
                                              : mesh.distribution.boundary_ordinals;
        if (section.volume) {
            slice.first = std::max(section.begin, ownership.begin(file.comm.rank()));
            const auto last = std::min(section.end, ownership.end(file.comm.rank()));
            slice.count = std::max(GlobalIndex{0}, last - slice.first);
            slice.maximum = maximum_overlap(ownership, section.begin, section.end);
        } else {
            const auto [first, last] = owned_boundary_range(mesh, section);
            slice.first = first;
            slice.count = last - first;
            slice.maximum = maximum_overlap(ownership, section.boundary_offset,
                                            section.boundary_offset + section.end - section.begin);
        }
        if (slice.count == 0) slice.first = section.begin;
        const auto n = host_size(slice.count);
        mpi::require(n <= std::numeric_limits<std::size_t>::max() / 9,
               "Local linear connectivity size exceeds host address space");
        if (section.type == CGNS_ENUMV(MIXED)) {
            std::int64_t first = 0, last = 0;
            // Two tiny matched reads avoid reallocating a potentially huge final CSR array.
            read_integer(file, section.offsets, slice.first - section.source_begin,
                         std::span<std::int64_t>{&first, 1});
            read_integer(file, section.offsets, slice.first + slice.count - section.source_begin,
                         std::span<std::int64_t>{&last, 1});
            mpi::require(first >= 0 && last >= first && last <= section.connectivity.shape[0],
                   "MIXED offset endpoints exceed connectivity extent");
            const auto records = host_size(last - first);
            mpi::require(records >= 2 * n && records - n <= 8 * n,
                   "Invalid or unsupported MIXED connectivity size");
            slice.node_count = records - n; // One element-type tag per record.
        } else {
            const auto arity = static_cast<std::size_t>(mesh::nodes_per_element(cell_type(section.type)));
            slice.node_count = n * arity;
        }
        const auto target = section.volume ? 0U : 1U;
        slice.element_begin = elements[target];
        slice.node_begin = nodes[target];
        mpi::require(slice.node_count <= std::numeric_limits<std::size_t>::max() - nodes[target],
               "Local connectivity allocation overflows size_t");
        elements[target] += n;
        nodes[target] += slice.node_count;
        slices.push_back(slice);
    }
    return slices;
}

/// @brief Reads owned element batches directly into preallocated final CSR storage.
void read_elements(const File& file, const ReaderOptions& options,
                   const Metadata& metadata, mesh::host::RawMesh& mesh)
{
    const auto slices = plan_elements(file, metadata, mesh);
    const auto allocate = [&](mesh::host::ElementTable& table, bool volume) {
        std::size_t count = 0, nodes = 0;
        for (const auto& slice : slices) if (slice.section->volume == volume) {
            count += host_size(slice.count);
            nodes += slice.node_count;
        }
        table.types.resize(count);
        table.offsets.resize(count + 1);
        table.node_ids.resize(nodes);
        table.offsets.front() = 0;
    };
    // The large arrays are allocated once, without vector growth during parsing.
    allocate(mesh.cells, true);
    allocate(mesh.boundary.elements, false);
    mesh.boundary.global_ids.resize(mesh.boundary.size());
    std::vector<std::int64_t> offsets, records;
    std::size_t mixed_batch = 0;
    for (const auto& slice : slices) if (slice.section->type == CGNS_ENUMV(MIXED))
        mixed_batch = std::max(mixed_batch, std::min(options.batch_elements, host_size(slice.count)));
    if (mixed_batch != 0) {
        offsets.reserve(mixed_batch + 1);
        records.reserve(9 * mixed_batch);
    }
    const auto total_nodes = mesh.distribution.nodes.global_size();
    for (const auto& slice : slices) {
        const auto& section = *slice.section;
        auto& table = section.volume ? mesh.cells : mesh.boundary.elements;
        const auto count_local = host_size(slice.count), maximum = host_size(slice.maximum);
        auto node_at = slice.node_begin;
        for (std::size_t round = 0; round < maximum;) {
            const auto n = round < count_local ? std::min(options.batch_elements, count_local - round) : 0;
            const auto at = n == 0 ? section.begin : slice.first + static_cast<GlobalIndex>(round);
            const auto element_at = slice.element_begin + std::min(round, count_local);
            if (section.type != CGNS_ENUMV(MIXED)) {
                const auto type = cell_type(section.type);
                const auto arity = static_cast<std::size_t>(mesh::nodes_per_element(type));
                auto destination = std::span<GlobalIndex>{table.node_ids}.subspan(node_at, n * arity);
                // Explicit conversion reads I4/I8 source nodes into the configured GlobalIndex.
                read_integer(file, section.connectivity,
                             static_cast<std::int64_t>(at - section.source_begin) * static_cast<std::int64_t>(arity),
                             destination);
                for (auto& id : destination) {
                    mpi::require(id > 0 && id <= total_nodes, "Connectivity references an invalid node ID");
                    --id;
                }
                for (std::size_t i = 0; i < n; ++i) {
                    table.types[element_at + i] = type;
                    table.offsets[element_at + i] = node_at + i * arity;
                }
                node_at += n * arity;
            } else {
                offsets.resize(n + 1);
                read_integer(file, section.offsets, at - section.source_begin,
                             std::span<std::int64_t>{offsets});
                mpi::require(offsets.front() >= 0 && offsets.back() >= offsets.front() &&
                           offsets.back() <= section.connectivity.shape[0],
                       "MIXED offsets exceed connectivity extent");
                const auto size = host_size(offsets.back() - offsets.front());
                mpi::require(size <= 9 * n, "MIXED records exceed supported linear arity");
                records.resize(size);
                read_integer(file, section.connectivity, offsets.front(),
                             std::span<std::int64_t>{records});
                for (std::size_t i = 0; i < n; ++i) {
                    const auto first = offsets[i], last = offsets[i + 1];
                    mpi::require(first >= offsets.front() && last > first && last <= offsets.back() &&
                               last - first >= 2 && last - first <= 9,
                           "Invalid MIXED record offsets");
                    const auto start = host_size(first - offsets.front());
                    mpi::require(std::in_range<int>(records[start]), "Invalid MIXED element type");
                    const auto type = cell_type(static_cast<ElementType>(records[start]));
                    const auto arity = static_cast<std::size_t>(mesh::nodes_per_element(type));
                    mpi::require(last - first == static_cast<std::int64_t>(arity + 1) &&
                               mesh::dimension(type) == (section.volume ? mesh.cell_dimension
                                                                       : mesh.cell_dimension - 1),
                           "MIXED section has an unsupported type, arity or dimension");
                    mpi::require(arity <= slice.node_begin + slice.node_count - node_at,
                           "MIXED offset plan does not match its records");
                    table.types[element_at + i] = type;
                    table.offsets[element_at + i] = node_at;
                    for (std::size_t j = 0; j < arity; ++j) {
                        const auto id = records[start + 1 + j];
                        mpi::require(id > 0 && id <= total_nodes, "MIXED connectivity references an invalid node ID");
                        table.node_ids[node_at++] = static_cast<GlobalIndex>(id - 1);
                    }
                }
            }
            if (section.boundary) {
                for (std::size_t i = 0; i < n; ++i)
                    mesh.boundary.global_ids[element_at + i] = at + static_cast<GlobalIndex>(i);
            }
            round += std::min(options.batch_elements, maximum - round);
        }
        mpi::require(node_at == slice.node_begin + slice.node_count, "Connectivity does not match its allocation plan");
        table.offsets[slice.element_begin + count_local] = node_at;
    }
    mpi::require(mesh.cells.size() == host_size(mesh.distribution.cells.local_size(file.comm.rank())) &&
               mesh.boundary.size() == mesh.boundary.patch_ids.size(), "Imported entity counts do not match ownership");
}

/// @brief Finds the boundary section containing a zero-based source element ID.
[[nodiscard]] const Section& boundary_section(const Metadata& metadata, GlobalIndex id)
{
    const auto found = std::upper_bound(metadata.sections.begin(), metadata.sections.end(), id,
        [](GlobalIndex value, const Section& section) { return value < section.begin; });
    mpi::require(found != metadata.sections.begin(), "BC references a missing boundary element");
    const auto& section = *std::prev(found);
    mpi::require(section.boundary && id < section.end, "BC references a non-boundary or missing element");
    return section;
}

/// @brief Assigns a patch while rejecting conflicting source BC definitions.
void assign_patch(mesh::host::RawMesh& mesh, std::size_t local, mesh::PatchId patch)
{
    assert(local < mesh.boundary.size());
    auto& current = mesh.boundary.patch_ids[local];
    mpi::require(current == mesh::invalid_patch_id || current == patch,
           "Boundary element belongs to conflicting BC patches");
    current = patch;
}

/// @brief Applies ranges locally and routes distributed BC lists to boundary owners.
void read_boundaries(const File& file, const ReaderOptions& options,
                     const Metadata& metadata, mesh::host::RawMesh& mesh)
{
    const auto comm = file.comm;
    const auto& ownership = mesh.distribution.boundary_ordinals;
    mpi::AlltoallvWorkspace workspace;
    std::vector<int> counts(static_cast<std::size_t>(comm.size()));
    std::vector<std::size_t> cursor(counts.size());
    std::vector<std::int64_t> points;
    std::vector<GlobalIndex> send, receive;
    // Even if every rank targets one receiver, its MPI int displacements remain representable.
    const auto batch = std::min(options.batch_elements,
        static_cast<std::size_t>(std::numeric_limits<int>::max() / comm.size()));
    for (const auto& bc : metadata.bcs) {
        if (bc.type == CGNS_ENUMV(PointRange) || bc.type == CGNS_ENUMV(ElementRange)) {
            std::array<std::int64_t, 2> range{};
            read_integer(file, bc.points, 0, std::span<std::int64_t>{range});
            mpi::require(range[0] >= 1 && range[1] >= 1, "Invalid BC range endpoint");
            const auto low = std::min(range[0], range[1]) - 1, high = std::max(range[0], range[1]);
            mpi::require(std::in_range<GlobalIndex>(high), "BC range exceeds GlobalIndex");
            std::int64_t covered = 0;
            for (const auto& section : metadata.sections) {
                const auto first = std::max(low, static_cast<std::int64_t>(section.begin));
                const auto last = std::min(high, static_cast<std::int64_t>(section.end));
                if (first >= last) continue;
                mpi::require(section.boundary, "BC range references non-boundary elements");
                covered += last - first;
                const auto [own_begin, own_end] = owned_boundary_range(mesh, section);
                for (auto id = std::max(first, static_cast<std::int64_t>(own_begin));
                     id < std::min(last, static_cast<std::int64_t>(own_end)); ++id) {
                    const auto ordinal = boundary_ordinal(section, static_cast<GlobalIndex>(id));
                    assign_patch(mesh, host_size(ordinal - ownership.begin(comm.rank())), bc.patch);
                }
            }
            mpi::require(covered == high - low, "BC range contains missing boundary elements");
            continue;
        }
        const WideBlock points_owner{bc.count, comm.size()};
        const auto first = points_owner.begin(comm.rank()), local = points_owner.local_size(comm.rank());
        const auto maximum = host_size(points_owner.local_size(0));
        for (std::size_t round = 0; round < maximum;) {
            const auto n = round < host_size(local) ? std::min(batch, host_size(local) - round) : 0;
            points.resize(n);
            read_integer(file, bc.points, first + static_cast<std::int64_t>(std::min(round, host_size(local))),
                         std::span<std::int64_t>{points});
            std::fill(counts.begin(), counts.end(), 0);
            for (auto& point : points) {
                mpi::require(point >= 1 && std::in_range<GlobalIndex>(point - 1), "Invalid BC point ID");
                const auto id = static_cast<GlobalIndex>(point - 1);
                point = boundary_ordinal(boundary_section(metadata, id), id);
                ++counts[static_cast<std::size_t>(ownership.owner(static_cast<GlobalIndex>(point)))];
            }
            std::size_t prefix = 0;
            for (std::size_t peer = 0; peer < counts.size(); ++peer) {
                cursor[peer] = prefix;
                prefix += static_cast<std::size_t>(counts[peer]);
            }
            send.resize(n);
            for (const auto ordinal : points) {
                const auto peer = static_cast<std::size_t>(ownership.owner(static_cast<GlobalIndex>(ordinal)));
                send[cursor[peer]++] = static_cast<GlobalIndex>(ordinal);
            }
            mpi::alltoallv_packed(comm, std::span<const int>{counts}, std::span<const GlobalIndex>{send},
                                 receive, workspace);
            for (const auto ordinal : receive) {
                assert(ownership.owner(ordinal) == comm.rank());
                assign_patch(mesh, host_size(ordinal - ownership.begin(comm.rank())), bc.patch);
            }
            round += std::min(batch, maximum - round);
        }
    }
}

} // namespace

/// @brief Reads owned source slices and automatically extrudes planar 2D imports.
ReadResult read_parallel(const std::string& path, mpi::CommView comm, const ReaderOptions& options)
{
    // check if MPI context is active and communicator is not MPI_COMM_NULL
    mpi::require(!comm.is_null() && mpi::is_active(), "CGNS import requires active MPI and a live communicator");

    // only main thread can perform parallel read
    mpi::require_main_thread();

    // check input parameters
    mpi::require(!path.empty() && options.base_index > 0 && options.zone_index > 0,
           "Invalid CGNS path or one-based base/zone index");
    mpi::require(options.batch_elements > 0 && options.batch_elements <= static_cast<std::size_t>(std::numeric_limits<int>::max() / 9),
           "Invalid CGNS temporary-buffer batch size");

    // open CGNS file 
    File file(path, comm, options);

    return mpi::guarded([&]() {
        ReadResult result;
        Metadata metadata;

        read_metadata(file, options, result, metadata);
        read_coordinates(file, options, metadata, result.mesh);
        read_elements(file, options, metadata, result.mesh);
        read_boundaries(file, options, metadata, result.mesh);

        // Close before communication-driven 2D conversion.
        file.close();
        if (result.mesh.cell_dimension == 2)
            result.extrusion =
                mesh::host::extrude_one_layer(result.mesh, comm, options.extrusion);

        return result;
    });
}

} // namespace krona::io::cgns