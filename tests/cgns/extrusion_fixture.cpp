#include <cgnslib.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

void check(int result)
{
    if (result != CG_OK)
    {
        std::fprintf(stderr, "CGNS fixture: %s\n", cg_get_error());
        std::exit(EXIT_FAILURE);
    }
}
int main(int argc, char **argv)
{
    if (argc != 3)
        return EXIT_FAILURE;
    const std::string plane = argv[2];
    int file = 0, base = 0, zone = 0, coordinate = 0, section = 0, bc = 0;
    check(cg_set_file_type(CG_FILE_HDF5));
    check(cg_open(argv[1], CG_MODE_WRITE, &file));
    check(cg_base_write(file, "Base", 2, plane == "xy" ? 2 : 3, &base));
    const cgsize_t counts[3]{5, 2, 0};
    check(cg_zone_write(file, base, "Zone", counts, CGNS_ENUMV(Unstructured), &zone));
    std::array<double, 5> x{0, 2, 2, 0, 3}, y{0, 0, 1, 1, 0}, z{0, 0, 0, 0, 0};
    if (plane == "xz")
    {
        z = y;
        y.fill(5);
    }
    if (plane == "yz")
    {
        z = y;
        y = x;
        x.fill(5);
    }
    check(cg_coord_write(file, base, zone, CGNS_ENUMV(RealDouble), "CoordinateX", x.data(),
                         &coordinate));
    check(cg_coord_write(file, base, zone, CGNS_ENUMV(RealDouble), "CoordinateY", y.data(),
                         &coordinate));
    if (plane != "xy")
        check(cg_coord_write(file, base, zone, CGNS_ENUMV(RealDouble), "CoordinateZ", z.data(),
                             &coordinate));
    const cgsize_t connectivity[]{CGNS_ENUMV(QUAD_4), 1, 4, 3, 2, CGNS_ENUMV(TRI_3), 2, 3, 5};
    const cgsize_t offsets[]{0, 5, 9};
    check(cg_poly_section_write(file, base, zone, "Cells", CGNS_ENUMV(MIXED), 1, 2, 0, connectivity,
                                offsets, &section));
    const cgsize_t edges[]{1, 2, 2, 5, 5, 3, 3, 4, 4, 1};
    check(cg_section_write(file, base, zone, "Boundary", CGNS_ENUMV(BAR_2), 3, 7, 0, edges,
                           &section));
    const cgsize_t range[]{3, 7};
    check(cg_boco_write(file, base, zone, "Wall", CGNS_ENUMV(BCWall), CGNS_ENUMV(PointRange), 2,
                        range, &bc));
    check(cg_boco_gridlocation_write(file, base, zone, bc, CGNS_ENUMV(EdgeCenter)));
    check(cg_close(file));
}
