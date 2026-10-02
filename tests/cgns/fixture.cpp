#include <cgnslib.h>
#include <cgns_io.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

void check(int error)
{
    if (error)
    {
        std::fprintf(stderr, "CGNS fixture: %s\n", cg_get_error());
        std::exit(2);
    }
}
void io(int error)
{
    if (error)
    {
        std::fprintf(stderr, "CGIO fixture error %d\n", error);
        std::exit(2);
    }
}
int main(int argc, char **argv)
{
    if (argc != 3)
        return 2;
    const std::string mode = argv[2];
    const bool two = mode.find("2d") != std::string::npos;
    const bool mixed = mode.find("mixed") != std::string::npos;
    const bool no_bc = mode.find("no-bc") != std::string::npos || mode == "no-boundary";
    const bool bad_location = mode == "vertex";
    int file = 0, base = 0, zone = 0, coord = 0, section = 0, bc = 0;
    check(cg_set_file_type(CG_FILE_HDF5));
    check(cg_open(argv[1], CG_MODE_WRITE, &file));
    check(
        cg_base_write(file, "Base", two ? 2 : 3, mode == "2d-embedded" ? 3 : (two ? 2 : 3), &base));
    cgsize_t sizes[3] = {8, 4, 0};
    check(cg_zone_write(file, base, "Zone", sizes, CGNS_ENUMV(Unstructured), &zone));
    std::array<double, 8> x{};
    std::array<float, 8> y{}, z{};
    for (int i = 0; i < 8; ++i)
    {
        x[i] = double(i) + 0.125;
        y[i] = float(i) * 2.0f + 0.5f;
        z[i] = float(i) * 3.0f + 0.25f;
    }
    if (mode == "nan")
        x[0] = std::numeric_limits<double>::quiet_NaN();
    if (!two || mode == "2d-embedded")
        check(cg_coord_write(file, base, zone, CGNS_ENUMV(RealSingle), "CoordinateZ", z.data(),
                             &coord));
    check(
        cg_coord_write(file, base, zone, CGNS_ENUMV(RealSingle), "CoordinateY", y.data(), &coord));
    check(
        cg_coord_write(file, base, zone, CGNS_ENUMV(RealDouble), "CoordinateX", x.data(), &coord));
    const std::array<CGNS_ENUMT(ElementType_t), 4> volume_types =
        two ? std::array<CGNS_ENUMT(ElementType_t), 4>{CGNS_ENUMV(TRI_3), CGNS_ENUMV(QUAD_4),
                                                       CGNS_ENUMV(TRI_3), CGNS_ENUMV(QUAD_4)}
            : std::array<CGNS_ENUMT(ElementType_t), 4>{CGNS_ENUMV(TETRA_4), CGNS_ENUMV(PYRA_5),
                                                       CGNS_ENUMV(PENTA_6), CGNS_ENUMV(HEXA_8)};
    std::vector<cgsize_t> cross_boundary_records, cross_boundary_offsets;
    auto write_section = [&](bool boundary)
    {
        std::vector<cgsize_t> records, offsets{0};
        for (int i = 0; i < 4; ++i)
        {
            auto type = boundary ? (two ? CGNS_ENUMV(BAR_2)
                                        : (mixed && i % 2 ? CGNS_ENUMV(QUAD_4) : CGNS_ENUMV(TRI_3)))
                                 : (mixed ? volume_types[i]
                                          : (two ? CGNS_ENUMV(TRI_3) : CGNS_ENUMV(TETRA_4)));
            int arity = 0;
            check(cg_npe(type, &arity));
            if (mixed)
                records.push_back(type);
            // Non-sorted boundary node order verifies orientation is not discarded.
            for (int n = 0; n < arity; ++n)
                records.push_back(boundary ? (n == 0 ? 3 : n == 2 ? 1 : n + 1) : n + 1);
            offsets.push_back(records.size());
        }
        if (mode == "bad-node" && !boundary)
            records[0] = std::numeric_limits<std::int32_t>::max() + cgsize_t{4};
        if (mode == "many-sections" && boundary)
        {
            for (int i = 0; i < 4; ++i)
            {
                const auto name = "Boundary" + std::to_string(i);
                check(cg_section_write(file, base, zone, name.c_str(), CGNS_ENUMV(TRI_3), 5 + i,
                                       5 + i, 0, records.data() + i * 3, &section));
            }
            return;
        }
        if (mode == "cross-mixed")
        {
            if (boundary)
            {
                cross_boundary_records = records;
                cross_boundary_offsets = offsets;
                return;
            }
            const auto prefix = records.size();
            records.insert(records.end(), cross_boundary_records.begin(),
                           cross_boundary_records.end());
            for (std::size_t i = 1; i < cross_boundary_offsets.size(); ++i)
                offsets.push_back(prefix + cross_boundary_offsets[i]);
            check(cg_poly_section_write(file, base, zone, "Combined", CGNS_ENUMV(MIXED), 1, 8, 0,
                                        records.data(), offsets.data(), &section));
            return;
        }
        if (mixed)
            check(cg_poly_section_write(file, base, zone, boundary ? "Boundary" : "Cells",
                                        CGNS_ENUMV(MIXED), boundary ? 5 : 1, boundary ? 8 : 4, 0,
                                        records.data(), offsets.data(), &section));
        else
            check(cg_section_write(file, base, zone, boundary ? "Boundary" : "Cells",
                                   boundary ? (two ? CGNS_ENUMV(BAR_2) : CGNS_ENUMV(TRI_3))
                                            : (two ? CGNS_ENUMV(TRI_3) : CGNS_ENUMV(TETRA_4)),
                                   boundary ? 5 : 1, boundary ? 8 : 4, 0, records.data(),
                                   &section));
    };
    if (mode != "no-boundary")
        write_section(true);
    write_section(false);
    if (!no_bc)
    {
        cgsize_t range[2] = {6, 5};
        check(cg_boco_write(file, base, zone, "RangeWall", CGNS_ENUMV(BCWall),
                            CGNS_ENUMV(PointRange), 2, range, &bc));
        if (!bad_location)
            check(cg_boco_gridlocation_write(
                file, base, zone, bc, two ? CGNS_ENUMV(EdgeCenter) : CGNS_ENUMV(FaceCenter)));
        check(cg_goto(file, base, "Zone_t", zone, "ZoneBC_t", 1, "BC_t", bc, "end"));
        check(cg_famname_write("MovingWall"));
        cgsize_t list[9] = {8, 7, 8, 7, 8, 7, 8, 7, 8};
        if (mode == "missing-id")
            list[0] = 99;
        check(cg_boco_write(file, base, zone, "ListFarfield", CGNS_ENUMV(BCFarfield),
                            CGNS_ENUMV(PointList), 9, list, &bc));
        if (!bad_location)
            check(cg_boco_gridlocation_write(
                file, base, zone, bc, two ? CGNS_ENUMV(EdgeCenter) : CGNS_ENUMV(FaceCenter)));
        if (mode == "conflict")
        {
            cgsize_t one = 5;
            check(cg_boco_write(file, base, zone, "Conflict", CGNS_ENUMV(BCWall),
                                CGNS_ENUMV(PointList), 1, &one, &bc));
            check(cg_boco_gridlocation_write(file, base, zone, bc, CGNS_ENUMV(FaceCenter)));
        }
    }
    check(cg_close(file));
    // Convert integer payloads to I4 independently of the 64-bit CGNS API build.
    if (mode.find("i4") != std::string::npos || mode == "malformed-mixed")
    {
        check(cg_open(argv[1], CG_MODE_MODIFY, &file));
        int cgio = 0;
        check(cg_get_cgio(file, &cgio));
        double zone_id = 0;
        check(cg_zone_id(file, 1, 1, &zone_id));
        for (const char *section_name : {"Cells", "Boundary"})
        {
            double section_id = 0;
            io(cgio_get_node_id(cgio, zone_id, section_name, &section_id));
            for (const char *array_name : {"ElementConnectivity", "ElementStartOffset"})
            {
                double id = 0;
                if (cgio_get_node_id(cgio, section_id, array_name, &id))
                    continue;
                cgsize_t dims[12]{};
                int ndim = 0;
                io(cgio_get_dimensions(cgio, id, &ndim, dims));
                std::vector<cgsize_t> values(dims[0]);
                io(cgio_read_all_data_type(cgio, id, "I8", values.data()));
                if (mode == "malformed-mixed" && std::string(array_name) == "ElementStartOffset")
                    values[1] = 100;
                if (mode.find("i4") != std::string::npos)
                {
                    std::vector<std::int32_t> narrowed(values.begin(), values.end());
                    io(cgio_set_dimensions(cgio, id, "I4", ndim, dims));
                    io(cgio_write_all_data(cgio, id, narrowed.data()));
                }
                else
                    io(cgio_write_all_data(cgio, id, values.data()));
            }
        }
        check(cg_close(file));
    }
    return 0;
}
