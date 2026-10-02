#include "krona/io/cgns/reader.hpp"

#include "krona/mpi/collective_status.hpp"
#include "krona/mpi/collectives.hpp"
#include "krona/mpi/error.hpp"

#include <cgnslib.h>
#include <cgns_io.h>
#include <pcgnslib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace krona::io::cgns
{
namespace
{

using ElementType = CGNS_ENUMT(ElementType_t);
using PointSetType = CGNS_ENUMT(PointSetType_t);
using DataType = CGNS_ENUMT(DataType_t);
using GridLocation = CGNS_ENUMT(GridLocation_t);
using Block = distribution::BlockDistribution<GlobalIndex>;
using WideBlock = distribution::BlockDistribution<std::int64_t>;
static_assert(sizeof(cgsize_t) == 8, "KRONA CGNS import requires a 64-bit CGNS build");
static_assert(std::is_signed_v<GlobalIndex> && std::is_signed_v<LocalIndex>);
static_assert(std::is_same_v<GeometryReal, float> || std::is_same_v<GeometryReal, double>);
std::mutex cgns_mutex; // CGNS/PCGNS maintain process-global state.

void check_cgns(int code, const char *operation, mpi::CommView comm) noexcept
{
    if (code == CG_OK)
    {
        return;
    }
    std::array<char, 1024> message{};
    std::snprintf(message.data(), message.size(), "%s: %s", operation, cg_get_error());
    mpi::fatal(comm.abort_scope(), message.data());
}
void check_cgio(int code, const char *operation, mpi::CommView comm) noexcept
{
    if (code == 0)
    {
        return;
    }
    std::array<char, 81> error{};
    static_cast<void>(cgio_error_message(error.data()));
    std::array<char, 1024> message{};
    std::snprintf(message.data(), message.size(), "%s: CGIO error %d: %s", operation, code,
                  error.data());
    mpi::fatal(comm.abort_scope(), message.data());
}
void ensure(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}
[[nodiscard]] GlobalIndex global_index(cgsize_t value)
{
    ensure(value >= 0 && std::in_range<GlobalIndex>(value),
           "CGNS index exceeds GlobalIndex; use 64-bit indices");
    return static_cast<GlobalIndex>(value);
}
[[nodiscard]] std::size_t host_size(std::int64_t value)
{
    ensure(value >= 0 && std::in_range<std::size_t>(value), "CGNS size exceeds host address space");
    return static_cast<std::size_t>(value);
}
void check_local_size(GlobalIndex count)
{
    ensure(std::in_range<LocalIndex>(count),
           "Owned entity count exceeds LocalIndex; use more ranks or 64-bit local indices");
}

// Metadata operations run in identical order; allocations agree before more IO.
template <class Work> void metadata_phase(mpi::CommView comm, const char *stage, Work &&work)
{
    mpi::LocalStatus status;
    try
    {
        std::forward<Work>(work)();
    }
    catch (const std::bad_alloc &)
    {
        status.fail("Memory allocation failed during CGNS import");
    }
    catch (const std::exception &error)
    {
        status.fail(error.what());
    }
    catch (...)
    {
        status.fail("Unknown exception during CGNS import");
    }
    mpi::require_collective_success(comm, status, stage);
}

template <class Work> void prepare(mpi::CommView comm, const char *stage, Work &&work)
{
    mpi::require_collective_success(comm, mpi::LocalStatus::capture(std::forward<Work>(work)),
                                    stage);
}

class File final
{
  public:
    File(const std::string &path, mpi::CommView comm) : comm_(comm)
    {
        check_cgns(cgp_mpi_comm(comm.native()), "cgp_mpi_comm", comm);
        check_cgns(cgp_mpi_info(MPI_INFO_NULL), "cgp_mpi_info", comm);
        check_cgns(cgp_pio_mode(CGP_INDEPENDENT), "cgp_pio_mode", comm);
        check_cgns(cgp_open(path.c_str(), CG_MODE_READ, &id_), "cgp_open", comm);
        check_cgns(cg_get_cgio(id_, &cgio_), "cg_get_cgio", comm);
        int type = 0;
        check_cgns(cg_get_file_type(id_, &type), "cg_get_file_type", comm);
        if (type != CG_FILE_HDF5)
        {
            mpi::fatal(comm.abort_scope(), "Parallel import requires HDF5-backed CGNS");
        }
    }
    ~File() noexcept
    {
        if (id_ >= 0)
        {
            mpi::fatal(comm_.abort_scope(), "CGNS file escaped collective close");
        }
    }
    File(const File &) = delete;
    File &operator=(const File &) = delete;
    [[nodiscard]] int id() const noexcept
    {
        return id_;
    }
    [[nodiscard]] int cgio() const noexcept
    {
        return cgio_;
    }
    [[nodiscard]] mpi::CommView comm() const noexcept
    {
        return comm_;
    }
    void close() noexcept
    {
        check_cgns(cgp_close(id_), "cgp_close", comm_);
        id_ = -1;
        // Do not leave CGNS holding a borrowed communicator past the import.
        check_cgns(cgp_mpi_comm(MPI_COMM_WORLD), "Reset CGNS communicator", comm_);
        check_cgns(cgp_pio_mode(CGP_COLLECTIVE), "Reset CGNS IO mode", comm_);
    }
    [[nodiscard]] double child(double parent, const char *name) const
    {
        double id = 0;
        check_cgio(cgio_get_node_id(cgio_, parent, name, &id), name, comm_);
        return id;
    }

  private:
    mpi::CommView comm_;
    int id_ = -1;
    int cgio_ = -1;
};

struct IntegerArray
{
    double id = 0;
    int dimensions = 0;
    std::array<cgsize_t, 12> shape{};
};
[[nodiscard]] IntegerArray integer_array(const File &file, double parent, const char *name,
                                         int dimensions)
{
    IntegerArray array;
    if (std::string_view{name} == "ElementStartOffset")
    {
        check_cgio(cgio_get_node_id(file.cgio(), parent, name, &array.id),
                   "ElementStartOffset is required; rewrite legacy MIXED sections with modern CGNS",
                   file.comm());
    }
    else
    {
        array.id = file.child(parent, name);
    }
    check_cgio(cgio_get_dimensions(file.cgio(), array.id, &array.dimensions, array.shape.data()),
               "cgio_get_dimensions", file.comm());
    ensure(array.dimensions == dimensions, "Unexpected CGNS integer array rank");
    std::array<char, 3> type{};
    check_cgio(cgio_get_data_type(file.cgio(), array.id, type.data()), "cgio_get_data_type",
               file.comm());
    ensure(std::string_view{type.data()} == "I4" || std::string_view{type.data()} == "I8",
           "Expected I4/I8 CGNS integer array");
    for (int d = 0; d < dimensions - 1; ++d)
    {
        ensure(array.shape[static_cast<std::size_t>(d)] == 1,
               "Expected an unstructured CGNS point set");
    }
    return array;
}

// Read a contiguous 1D payload, or [1,n] point set, with exact I4/I8 conversion.
void read_integer(const File &file, const IntegerArray &array, std::int64_t first,
                  std::span<std::int64_t> output) noexcept
{
    if (output.empty())
    {
        // CGNS enables collective HDF5 metadata even in independent data mode.
        // Keep dataset access matched with a discarded single-value read.
        std::int64_t dummy = 0;
        read_integer(file, array, 0, std::span<std::int64_t>{&dummy, 1});
        return;
    }
    const auto last_dim = static_cast<std::size_t>(array.dimensions - 1);
    if (first < 0 || first > array.shape[last_dim] ||
        output.size() > static_cast<std::uint64_t>(array.shape[last_dim] - first))
    {
        mpi::fatal(file.comm().abort_scope(), "CGNS integer slice exceeds array extent");
    }
    std::array<cgsize_t, 2> start{1, 1}, end{1, 1}, stride{1, 1}, memory{1, 1};
    start[last_dim] = static_cast<cgsize_t>(first + 1);
    end[last_dim] = static_cast<cgsize_t>(first) + static_cast<cgsize_t>(output.size());
    memory[last_dim] = static_cast<cgsize_t>(output.size());
    const std::array<cgsize_t, 2> memory_start{1, 1};
    check_cgio(cgio_read_data_type(file.cgio(), array.id, start.data(), end.data(), stride.data(),
                                   "I8", array.dimensions, memory.data(), memory_start.data(),
                                   memory.data(), stride.data(), output.data()),
               "cgio_read_data_type(I8)", file.comm());
}
[[nodiscard]] mesh::CellType cell_type(ElementType type)
{
    switch (type)
    {
    case CGNS_ENUMV(TETRA_4):
        return mesh::CellType::TET;
    case CGNS_ENUMV(PYRA_5):
        return mesh::CellType::PYRA;
    case CGNS_ENUMV(PENTA_6):
        return mesh::CellType::PRISM;
    case CGNS_ENUMV(HEXA_8):
        return mesh::CellType::HEXA;
    case CGNS_ENUMV(TRI_3):
        return mesh::CellType::TRI;
    case CGNS_ENUMV(QUAD_4):
        return mesh::CellType::QUAD;
    case CGNS_ENUMV(BAR_2):
        return mesh::CellType::BAR;
    case CGNS_ENUMV(NODE):
        return mesh::CellType::NODE;
    default:
        throw std::runtime_error(
            "Unsupported CGNS element: only linear fixed-arity types are implemented");
    }
}
struct Section
{
    GlobalIndex begin = 0, end = 0;
    GlobalIndex source_begin = 0, source_end = 0;
    ElementType type{};
    IntegerArray connectivity, offsets;
    bool volume = false;
    bool boundary = false;
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

void read_metadata(const File &file, const ReaderOptions &options, ReadResult &result,
                   Metadata &metadata)
{
    const int f = file.id(), b = options.base_index, z = options.zone_index;
    auto &mesh = result.mesh;
    char name[33]{};
    check_cgns(cg_version(f, &result.source.version), "cg_version", file.comm());
    check_cgns(cg_base_read(f, b, name, &mesh.cell_dimension, &mesh.physical_dimension),
               "cg_base_read", file.comm());
    prepare(file.comm(), "Store base name", [&] { result.source.base_name = name; });
    ensure((mesh.cell_dimension == 2 || mesh.cell_dimension == 3) &&
               mesh.physical_dimension >= mesh.cell_dimension && mesh.physical_dimension <= 3,
           "Expected a 2D/3D CGNS base");
    CGNS_ENUMT(ZoneType_t) zone_type{};
    check_cgns(cg_zone_type(f, b, z, &zone_type), "cg_zone_type", file.comm());
    ensure(zone_type == CGNS_ENUMV(Unstructured), "Only unstructured zones are implemented");
    std::array<cgsize_t, 9> zone_size{};
    check_cgns(cg_zone_read(f, b, z, name, zone_size.data()), "cg_zone_read", file.comm());
    prepare(file.comm(), "Store zone name", [&] { result.source.zone_name = name; });
    const auto nodes = global_index(zone_size[0]), cells = global_index(zone_size[1]);
    ensure(nodes > 0 && cells > 0, "Empty CGNS zones are not supported");
    mesh.distribution = {
        Block{cells, file.comm().size()}, Block{nodes, file.comm().size()}, {}, file.comm().rank()};
    prepare(file.comm(), "Validate local sizes",
            [&]
            {
                check_local_size(mesh.distribution.cells.local_size(file.comm().rank()));
                check_local_size(mesh.distribution.nodes.local_size(file.comm().rank()));
            });

    int coordinates = 0;
    check_cgns(cg_ncoords(f, b, z, &coordinates), "cg_ncoords", file.comm());
    constexpr std::array<std::string_view, 3> names{"CoordinateX", "CoordinateY", "CoordinateZ"};
    for (int c = 1; c <= coordinates; ++c)
    {
        DataType type{};
        check_cgns(cg_coord_info(f, b, z, c, &type, name), "cg_coord_info", file.comm());
        for (std::size_t axis = 0; axis < names.size(); ++axis)
        {
            if (name != names[axis])
            {
                continue;
            }
            ensure(type == CGNS_ENUMV(RealSingle) || type == CGNS_ENUMV(RealDouble),
                   "Expected floating-point Cartesian coordinates");
            ensure(metadata.coordinate_indices[axis] == 0, "Duplicate Cartesian coordinate array");
            metadata.coordinate_indices[axis] = c;
        }
    }
    for (int axis = 0; axis < mesh.physical_dimension; ++axis)
    {
        ensure(metadata.coordinate_indices[static_cast<std::size_t>(axis)] != 0,
               "Missing named Cartesian coordinate array");
    }

    double zone_id = 0;
    check_cgns(cg_zone_id(f, b, z, &zone_id), "cg_zone_id", file.comm());
    int sections = 0;
    check_cgns(cg_nsections(f, b, z, &sections), "cg_nsections", file.comm());
    prepare(file.comm(), "Allocate section metadata",
            [&] { metadata.sections.reserve(static_cast<std::size_t>(sections) * 2); });
    for (int s = 1; s <= sections; ++s)
    {
        Section section;
        cgsize_t first = 0, last = 0;
        int boundary_count = 0, parent_data = 0;
        check_cgns(cg_section_read(f, b, z, s, name, &section.type, &first, &last, &boundary_count,
                                   &parent_data),
                   "cg_section_read", file.comm());
        ensure(first >= 1 && last >= first, "Invalid CGNS section range");
        section.begin = global_index(first - 1);
        section.end = global_index(last);
        section.source_begin = section.begin;
        section.source_end = section.end;
        section.volume = section.end <= cells;
        ensure(section.volume || section.begin >= cells || section.type == CGNS_ENUMV(MIXED),
               "A section crosses the volume/boundary ID boundary");
        if (section.type == CGNS_ENUMV(MIXED))
        {
            section.boundary = !section.volume;
        }
        else
        {
            const auto dimension = mesh::dimension(cell_type(section.type));
            ensure(!section.volume || dimension == mesh.cell_dimension,
                   "Volume section has the wrong dimension");
            ensure(section.volume || dimension < mesh.cell_dimension,
                   "A volume section lies outside the zone cell range");
            section.boundary = dimension == mesh.cell_dimension - 1;
        }
        if (section.volume || section.boundary)
        {
            const double section_id = file.child(zone_id, name);
            section.connectivity = integer_array(file, section_id, "ElementConnectivity", 1);
            if (section.type == CGNS_ENUMV(MIXED))
            {
                // Modern MIXED requires offsets; never replicate legacy connectivity.
                section.offsets = integer_array(file, section_id, "ElementStartOffset", 1);
                const auto element_count = last - first + 1;
                ensure(element_count < std::numeric_limits<cgsize_t>::max(),
                       "MIXED offset array size overflow");
                ensure(section.offsets.shape[0] == element_count + 1,
                       "Invalid MIXED offset array extent");
            }
            else
            {
                const auto arity =
                    static_cast<cgsize_t>(mesh::nodes_per_element(cell_type(section.type)));
                ensure(last - first + 1 <= std::numeric_limits<cgsize_t>::max() / arity,
                       "Section connectivity size overflow");
                ensure(section.connectivity.shape[0] == (last - first + 1) * arity,
                       "Invalid uniform connectivity extent");
            }
        }
        if (section.type == CGNS_ENUMV(MIXED) && section.begin < cells && section.end > cells)
        {
            auto volume = section;
            volume.end = cells;
            volume.volume = true;
            volume.boundary = false;
            metadata.sections.push_back(volume);
            section.begin = cells;
        }
        metadata.sections.push_back(section);
    }
    std::sort(metadata.sections.begin(), metadata.sections.end(),
              [](const Section &left, const Section &right) { return left.begin < right.begin; });
    GlobalIndex next_cell = 0, previous_end = 0;
    GlobalIndex boundary_total = 0;
    for (auto &section : metadata.sections)
    {
        ensure(section.begin >= previous_end, "Overlapping CGNS section ranges");
        previous_end = section.end;
        if (section.volume)
        {
            ensure(section.begin == next_cell,
                   "Volume sections must cover CGNS cell IDs without gaps");
            next_cell = section.end;
        }
        if (section.boundary)
        {
            section.boundary_offset = boundary_total;
            ensure(section.end - section.begin <=
                       std::numeric_limits<GlobalIndex>::max() - boundary_total,
                   "Global boundary count overflow");
            boundary_total += section.end - section.begin;
        }
    }
    ensure(next_cell == cells, "Volume sections do not cover the zone cell range");
    mesh.distribution.boundary_ordinals = Block{boundary_total, file.comm().size()};
    const auto boundary_local =
        host_size(mesh.distribution.boundary_ordinals.local_size(file.comm().rank()));
    prepare(file.comm(), "Allocate local boundary metadata",
            [&]
            {
                ensure(boundary_local <=
                           static_cast<std::uint64_t>(std::numeric_limits<LocalIndex>::max()),
                       "Local boundary count exceeds LocalIndex");
                mesh.boundary.global_ids.reserve(boundary_local);
                mesh.boundary.patch_ids.assign(boundary_local, mesh::invalid_patch_id);
            });

    int bcs = 0;
    check_cgns(cg_nbocos(f, b, z, &bcs), "cg_nbocos", file.comm());
    prepare(file.comm(), "Allocate BC metadata",
            [&]
            {
                metadata.bcs.reserve(static_cast<std::size_t>(bcs));
                mesh.patches.reserve(static_cast<std::size_t>(bcs));
                result.source.ignored_boundary_conditions.reserve(static_cast<std::size_t>(bcs));
            });
    for (int bc = 1; bc <= bcs; ++bc)
    {
        BoundaryCondition condition;
        CGNS_ENUMT(BCType_t) type{};
        DataType normal_type{};
        cgsize_t count = 0, normal_size = 0;
        int normal_indices[3]{}, datasets = 0;
        check_cgns(cg_boco_info(f, b, z, bc, name, &type, &condition.type, &count, normal_indices,
                                &normal_size, &normal_type, &datasets),
                   "cg_boco_info", file.comm());
        GridLocation location{};
        check_cgns(cg_boco_gridlocation_read(f, b, z, bc, &location), "cg_boco_gridlocation_read",
                   file.comm());
        const bool supported =
            (condition.type == CGNS_ENUMV(PointList) || condition.type == CGNS_ENUMV(PointRange)) &&
            ((mesh.cell_dimension == 3 && location == CGNS_ENUMV(FaceCenter)) ||
             (mesh.cell_dimension == 2 && location == CGNS_ENUMV(EdgeCenter)));
        if (!supported)
        {
            ensure(options.unsupported_boundary == UnsupportedBoundary::ignore,
                   "Unsupported BC: require PointList/PointRange at FaceCenter (3D) or EdgeCenter "
                   "(2D); absent GridLocation means Vertex");
            prepare(file.comm(), "Record ignored BC",
                    [&] { result.source.ignored_boundary_conditions.emplace_back(name); });
            continue;
        }
        ensure(count >= 0, "Negative BC point count");
        condition.count = count;
        ensure(mesh.patches.size() <
                   static_cast<std::size_t>(std::numeric_limits<mesh::PatchId>::max()),
               "Too many boundary patches");
        condition.patch = static_cast<mesh::PatchId>(mesh.patches.size());
        check_cgns(cg_goto(f, b, "Zone_t", z, "ZoneBC_t", 1, "BC_t", bc, "end"), "cg_goto(BC)",
                   file.comm());
        char family[33]{};
        const int family_status = cg_famname_read(family);
        if (family_status != CG_OK && family_status != CG_NODE_NOT_FOUND)
        {
            check_cgns(family_status, "cg_famname_read", file.comm());
        }
        prepare(file.comm(), "Store BC labels",
                [&]
                {
                    mesh.patches.push_back(
                        {name, family_status == CG_OK ? family : "", cg_BCTypeName(type)});
                });
        double bc_id = 0;
        check_cgns(cg_boco_id(f, b, z, bc, &bc_id), "cg_boco_id", file.comm());
        condition.points = integer_array(
            file, bc_id, condition.type == CGNS_ENUMV(PointRange) ? "PointRange" : "PointList", 2);
        ensure(condition.points.shape[1] == count, "BC point set extent mismatch");
        ensure(condition.type != CGNS_ENUMV(PointRange) || count == 2,
               "PointRange must contain two endpoints");
        metadata.bcs.push_back(condition);
    }
}

[[nodiscard]] std::pair<GlobalIndex, GlobalIndex> owned_boundary_range(const mesh::RawMesh &mesh,
                                                                       const Section &section)
{
    const auto &distribution = mesh.distribution.boundary_ordinals;
    const auto first =
        std::max(section.boundary_offset, distribution.begin(mesh.distribution.rank));
    const auto last = std::min(section.boundary_offset + (section.end - section.begin),
                               distribution.end(mesh.distribution.rank));
    if (first >= last)
    {
        return {section.begin, section.begin};
    }
    return {section.begin + (first - section.boundary_offset),
            section.begin + (last - section.boundary_offset)};
}
[[nodiscard]] GlobalIndex boundary_ordinal(const Section &section, GlobalIndex source_id) noexcept
{
    return section.boundary_offset + (source_id - section.begin);
}
[[nodiscard]] std::size_t local_boundary_index(const mesh::RawMesh &mesh, const Section &section,
                                               GlobalIndex source_id)
{
    return host_size(boundary_ordinal(section, source_id) -
                     mesh.distribution.boundary_ordinals.begin(mesh.distribution.rank));
}

void read_coordinates(const File &file, const ReaderOptions &options, const Metadata &metadata,
                      mesh::RawMesh &mesh)
{
    const auto first = mesh.distribution.nodes.begin(file.comm().rank());
    const auto n = host_size(mesh.distribution.nodes.local_size(file.comm().rank()));
    prepare(file.comm(), "Allocate owned coordinates",
            [&]
            {
                mesh.coordinates.x.resize(n);
                mesh.coordinates.y.resize(n);
                mesh.coordinates.z.assign(n, GeometryReal{0});
            });
    const auto maximum = host_size(mesh.distribution.nodes.local_size(0));
    std::array<std::vector<GeometryReal> *, 3> values{&mesh.coordinates.x, &mesh.coordinates.y,
                                                      &mesh.coordinates.z};
    constexpr auto type =
        std::is_same_v<GeometryReal, float> ? CGNS_ENUMV(RealSingle) : CGNS_ENUMV(RealDouble);
    for (int axis = 0; axis < mesh.physical_dimension; ++axis)
    {
        auto &coordinates = *values[static_cast<std::size_t>(axis)];
        for (std::size_t at = 0; at < maximum;)
        {
            const auto count = at < n ? std::min(options.batch_elements, n - at) : std::size_t{0};
            const cgsize_t start = count == 0 ? 1 : static_cast<cgsize_t>(first) + static_cast<cgsize_t>(at) + 1;
            const cgsize_t end = count == 0 ? 1 : start + static_cast<cgsize_t>(count) - 1;
            const cgsize_t one = 1, memory = count == 0 ? 1 : static_cast<cgsize_t>(count);
            check_cgns(cgp_coord_general_read_data(
                           file.id(), options.base_index, options.zone_index,
                           metadata.coordinate_indices[static_cast<std::size_t>(axis)], &start,
                           &end, type, 1, &memory, &one, &memory,
                           count == 0 ? nullptr : coordinates.data() + at),
                       "cgp_coord_general_read_data", file.comm());
            prepare(file.comm(), "Validate coordinate batch",
                    [&]
                    {
                        for (std::size_t i = 0; i < count; ++i)
                        {
                            ensure(std::isfinite(coordinates[at + i]),
                                   "Non-finite coordinate after conversion to GeometryReal");
                        }
                    });
            at += std::min(options.batch_elements, maximum - at);
        }
    }
}

void validate_nodes(std::span<GlobalIndex> nodes, GlobalIndex total)
{
    for (auto &id : nodes)
    {
        ensure(id > 0 && id <= total, "Connectivity references an invalid node ID");
        --id;
    }
}
template <class Index>
void read_uniform_nodes(const File &file, const IntegerArray &array, std::int64_t offset,
                        std::span<Index> destination, std::vector<std::int64_t> &scratch) noexcept
{
    if constexpr (std::is_same_v<Index, std::int64_t>)
    {
        read_integer(file, array, offset, destination);
    }
    else
    {
        read_integer(file, array, offset, scratch);
    }
}
void read_elements(const File &file, const ReaderOptions &options, const Metadata &metadata,
                   mesh::RawMesh &mesh)
{
    const auto comm = file.comm();
    prepare(comm, "Allocate local element tables",
            [&]
            {
                mesh.cells.types.reserve(
                    host_size(mesh.distribution.cells.local_size(comm.rank())));
                mesh.cells.offsets.reserve(mesh.cells.types.capacity() + 1);
                mesh.boundary.elements.types.reserve(mesh.boundary.patch_ids.size());
                mesh.boundary.elements.offsets.reserve(mesh.boundary.patch_ids.size() + 1);
            });
    std::vector<std::int64_t> offsets, records;
    const GlobalIndex total_nodes = mesh.distribution.nodes.global_size();
    for (const auto &section : metadata.sections)
    {
        if (!section.volume && !section.boundary)
        {
            continue;
        }
        GlobalIndex begin = 0, end = 0;
        if (section.volume)
        {
            begin = std::max(section.begin, mesh.distribution.cells.begin(comm.rank()));
            end = std::min(section.end, mesh.distribution.cells.end(comm.rank()));
        }
        else
        {
            const auto range = owned_boundary_range(mesh, section);
            begin = range.first;
            end = range.second;
        }
        const auto local_count = begin < end ? end - begin : GlobalIndex{0};
        const auto maximum = mpi::allreduce<mpi::Reduction::max>(comm, local_count);
        auto &table = section.volume ? mesh.cells : mesh.boundary.elements;
        for (GlobalIndex round = 0; round < maximum;)
        {
            const auto count = round < local_count ? std::min(options.batch_elements,
                                                              host_size(local_count - round))
                                                   : std::size_t{0};
            const auto at = count == 0 ? section.begin : begin + round;
            auto store_boundary_ids = [&]
            {
                if (section.boundary)
                {
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        mesh.boundary.global_ids.push_back(at + static_cast<GlobalIndex>(i));
                    }
                }
            };
            if (section.type != CGNS_ENUMV(MIXED))
            {
                const auto type = cell_type(section.type);
                const auto arity = static_cast<std::size_t>(mesh::nodes_per_element(type));
                const auto size = count * arity;
                const auto old_size = table.node_ids.size();
                prepare(comm, "Allocate uniform connectivity batch",
                        [&]
                        {
                            ensure(size <= table.node_ids.max_size() - old_size,
                                   "Connectivity buffer size overflow");
                            table.node_ids.resize(old_size + size);
                            if constexpr (!std::is_same_v<GlobalIndex, std::int64_t>)
                            {
                                records.resize(size);
                            }
                        });
                auto destination = std::span<GlobalIndex>{table.node_ids}.subspan(old_size);
                const std::int64_t offset = static_cast<std::int64_t>(at - section.source_begin) *
                                            static_cast<std::int64_t>(arity);
                read_uniform_nodes(file, section.connectivity, offset, destination, records);
                prepare(comm, "Validate uniform connectivity batch",
                        [&]
                        {
                            if constexpr (std::is_same_v<GlobalIndex, std::int64_t>)
                            {
                                validate_nodes(destination, total_nodes);
                            }
                            else
                            {
                                for (std::size_t i = 0; i < size; ++i)
                                {
                                    ensure(records[i] > 0 && records[i] <= total_nodes,
                                           "Connectivity references an invalid node ID");
                                    destination[i] = static_cast<GlobalIndex>(records[i] - 1);
                                }
                            }
                            for (std::size_t i = 0; i < count; ++i)
                            {
                                table.types.push_back(type);
                                table.offsets.push_back(old_size + (i + 1) * arity);
                            }
                            store_boundary_ids();
                        });
            }
            else
            {
                prepare(comm, "Allocate MIXED offsets", [&] { offsets.resize(count + 1); });
                read_integer(file, section.offsets, at - section.source_begin, offsets);
                prepare(comm, "Validate MIXED offsets and allocate records",
                        [&]
                        {
                            ensure(offsets.front() >= 0 &&
                                       offsets.back() <= section.connectivity.shape[0],
                                   "MIXED offsets exceed connectivity extent");
                            if (count != 0)
                            {
                                ensure(at != section.source_begin || offsets.front() == 0,
                                       "MIXED offsets must start at zero");
                                ensure(at + static_cast<GlobalIndex>(count) != section.source_end ||
                                           offsets.back() == section.connectivity.shape[0],
                                       "MIXED offsets must cover connectivity");
                            }
                            for (std::size_t i = 0; i < count; ++i)
                            {
                                ensure(offsets[i + 1] >= offsets[i] &&
                                           offsets[i + 1] - offsets[i] >= 2 &&
                                           offsets[i + 1] - offsets[i] <= 9,
                                       "Invalid or unsupported MIXED record length");
                            }
                            records.resize(host_size(offsets.back() - offsets.front()));
                        });
                read_integer(file, section.connectivity, offsets.front(), records);
                prepare(comm, "Validate and append MIXED records",
                        [&]
                        {
                            for (std::size_t i = 0; i < count; ++i)
                            {
                                const auto start = host_size(offsets[i] - offsets.front());
                                ensure(std::in_range<int>(records[start]),
                                       "Invalid MIXED element type");
                                const auto type =
                                    cell_type(static_cast<ElementType>(records[start]));
                                const auto arity =
                                    static_cast<std::size_t>(mesh::nodes_per_element(type));
                                ensure(offsets[i + 1] - offsets[i] ==
                                           static_cast<std::int64_t>(arity + 1),
                                       "MIXED record length/type mismatch");
                                ensure(mesh::dimension(type) == (section.volume
                                                                     ? mesh.cell_dimension
                                                                     : mesh.cell_dimension - 1),
                                       "MIXED section contains an element of the wrong dimension");
                                for (std::size_t node = 0; node < arity; ++node)
                                {
                                    const auto id = records[start + 1 + node];
                                    ensure(id > 0 && id <= total_nodes,
                                           "MIXED connectivity references an invalid node ID");
                                    table.node_ids.push_back(static_cast<GlobalIndex>(id - 1));
                                }
                                table.types.push_back(type);
                                table.offsets.push_back(table.node_ids.size());
                            }
                            store_boundary_ids();
                        });
            }
            round += static_cast<GlobalIndex>(
                std::min(options.batch_elements, host_size(maximum - round)));
        }
    }
    prepare(comm, "Validate local import counts",
            [&]
            {
                ensure(mesh.cells.size() ==
                           host_size(mesh.distribution.cells.local_size(comm.rank())),
                       "Imported local cell count mismatch");
                ensure(mesh.boundary.size() == mesh.boundary.patch_ids.size(),
                       "Imported local boundary count mismatch");
            });
}

[[nodiscard]] const Section &boundary_section(const Metadata &metadata, GlobalIndex id)
{
    auto it = std::upper_bound(metadata.sections.begin(), metadata.sections.end(), id,
                               [](GlobalIndex value, const Section &section)
                               { return value < section.begin; });
    ensure(it != metadata.sections.begin(), "BC references a missing boundary element");
    --it;
    ensure(it->boundary && id < it->end, "BC references a non-boundary or missing element");
    return *it;
}
void assign_patch(mesh::RawMesh &mesh, std::size_t local, mesh::PatchId patch)
{
    ensure(local < mesh.boundary.size(), "BC local boundary index out of bounds");
    auto &current = mesh.boundary.patch_ids[local];
    ensure(current == mesh::invalid_patch_id || current == patch,
           "Boundary element belongs to conflicting BC patches");
    current = patch;
}
void read_boundaries(const File &file, const ReaderOptions &options, const Metadata &metadata,
                     mesh::RawMesh &mesh)
{
    const auto comm = file.comm();
    mpi::AlltoallvWorkspace workspace;
    std::vector<int> counts;
    std::vector<std::size_t> positions;
    std::vector<std::int64_t> points;
    std::vector<GlobalIndex> send, receive;
    // Bound total messages in one round, including concentration on one receiver.
    const auto batch =
        std::min(options.batch_elements,
                 static_cast<std::size_t>(std::numeric_limits<int>::max() / comm.size()));
    for (const auto &bc : metadata.bcs)
    {
        if (bc.type == CGNS_ENUMV(PointRange))
        {
            std::array<std::int64_t, 2> range{};
            read_integer(file, bc.points, 0, range);
            prepare(
                comm, "Apply BC PointRange",
                [&]
                {
                    ensure(range[0] >= 1 && range[1] >= 1, "Invalid BC range endpoint");
                    const auto low = std::min(range[0], range[1]) - 1;
                    const auto high = std::max(range[0], range[1]);
                    ensure(std::in_range<GlobalIndex>(high), "BC range exceeds GlobalIndex");
                    std::int64_t covered = 0;
                    for (const auto &section : metadata.sections)
                    {
                        const auto begin = std::max(low, static_cast<std::int64_t>(section.begin));
                        const auto end = std::min(high, static_cast<std::int64_t>(section.end));
                        if (begin >= end)
                        {
                            continue;
                        }
                        ensure(section.boundary, "BC range references non-boundary elements");
                        covered += end - begin;
                        const auto [own_begin, own_end] = owned_boundary_range(mesh, section);
                        for (auto id = std::max(begin, static_cast<std::int64_t>(own_begin));
                             id < std::min(end, static_cast<std::int64_t>(own_end)); ++id)
                        {
                            assign_patch(
                                mesh,
                                local_boundary_index(mesh, section, static_cast<GlobalIndex>(id)),
                                bc.patch);
                        }
                    }
                    ensure(covered == high - low, "BC range contains missing boundary elements");
                });
            continue;
        }
        const WideBlock ownership{bc.count, comm.size()};
        const auto begin = ownership.begin(comm.rank()), end = ownership.end(comm.rank());
        const auto maximum_local = ownership.local_size(0);
        for (std::int64_t round = 0; round < maximum_local;)
        {
            const auto point_start = begin + std::min(round, end - begin);
            const auto n = std::min(batch, host_size(end - point_start));
            prepare(comm, "Allocate BC point batch", [&] { points.resize(n); });
            read_integer(file, bc.points, point_start, points);
            prepare(comm, "Pack BC PointList",
                    [&]
                    {
                        counts.assign(static_cast<std::size_t>(comm.size()), 0);
                        positions.resize(counts.size());
                        for (const auto point : points)
                        {
                            ensure(point >= 1 && std::in_range<GlobalIndex>(point - 1),
                                   "BC point ID exceeds GlobalIndex");
                            const auto id = static_cast<GlobalIndex>(point - 1);
                            const auto &section = boundary_section(metadata, id);
                            const auto target = mesh.distribution.boundary_ordinals.owner(
                                boundary_ordinal(section, id));
                            ++counts[static_cast<std::size_t>(target)];
                        }
                        std::size_t prefix = 0;
                        for (std::size_t peer = 0; peer < counts.size(); ++peer)
                        {
                            positions[peer] = prefix;
                            prefix += static_cast<std::size_t>(counts[peer]);
                        }
                        send.resize(n);
                        for (const auto point : points)
                        {
                            const auto id = static_cast<GlobalIndex>(point - 1);
                            const auto &section = boundary_section(metadata, id);
                            const auto target = mesh.distribution.boundary_ordinals.owner(
                                boundary_ordinal(section, id));
                            send[positions[static_cast<std::size_t>(target)]++] = id;
                        }
                    });
            mpi::alltoallv_packed(comm, std::span<const int>{counts},
                                  std::span<const GlobalIndex>{send}, receive, workspace);
            const auto status = mpi::LocalStatus::capture(
                [&]
                {
                    for (const auto id : receive)
                    {
                        const auto &section = boundary_section(metadata, id);
                        ensure(mesh.distribution.boundary_ordinals.owner(
                                   boundary_ordinal(section, id)) == comm.rank(),
                               "BC routed to the wrong rank");
                        assign_patch(mesh, local_boundary_index(mesh, section, id), bc.patch);
                    }
                });
            mpi::require_collective_success(comm, status, "Assign BC PointList");
            round += static_cast<std::int64_t>(std::min(batch, host_size(maximum_local - round)));
        }
    }
}

} // namespace

ReadResult read_parallel(const std::string &path, mpi::CommView comm, const ReaderOptions &options)
{
    if (comm.is_null() || comm.is_intercommunicator())
    {
        mpi::fatal(comm.abort_scope(), "CGNS import requires a live intracommunicator");
    }
    int main_thread = 0;
    mpi::check(MPI_Is_thread_main(&main_thread), comm.abort_scope(), "MPI_Is_thread_main");
    if (!main_thread)
    {
        mpi::fatal(comm.abort_scope(), "CGNS import must run on the MPI main thread");
    }
    std::unique_lock lock(cgns_mutex, std::defer_lock);
    const auto status = mpi::LocalStatus::capture(
        [&]
        {
            ensure(lock.try_lock(), "Concurrent CGNS imports are not supported in one process");
            ensure(!path.empty() && path.find('\0') == std::string::npos, "Invalid CGNS file path");
            ensure(options.base_index > 0 && options.zone_index > 0,
                   "CGNS base/zone indices are one-based");
            ensure(options.batch_elements > 0 &&
                       options.batch_elements <=
                           static_cast<std::size_t>(std::numeric_limits<int>::max() / 9),
                   "Invalid CGNS batch size");
            ensure(options.unsupported_boundary == UnsupportedBoundary::reject ||
                       options.unsupported_boundary == UnsupportedBoundary::ignore,
                   "Invalid unsupported BC policy");
        });
    mpi::require_collective_success(comm, status, "Prepare CGNS import");
    File file(path, comm);
    ReadResult result;
    Metadata metadata;
    metadata_phase(comm, "Read CGNS metadata",
                   [&] { read_metadata(file, options, result, metadata); });
    read_coordinates(file, options, metadata, result.mesh);
    read_elements(file, options, metadata, result.mesh);
    read_boundaries(file, options, metadata, result.mesh);
    file.close();
    return result;
}

} // namespace krona::io::cgns
