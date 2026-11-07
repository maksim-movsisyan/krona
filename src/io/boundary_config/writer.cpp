#include <krona/io/boundary_config/writer.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <cstdio>
#include <fstream>
#include <locale>
#include <ostream>
#include <string>
#include <string_view>

namespace krona::io::boundary_config {
namespace {

/// @brief Suggests a supported physical type; an empty result requires an explicit user choice.
[[nodiscard]] std::string_view suggested_type(std::string_view source) noexcept
{
    if (source == "BCInflowSupersonic") return "SUPERSONIC_INLET";
    if (source == "BCOutflowSupersonic") return "SUPERSONIC_OUTLET";
    if (source == "BCInflow" || source == "BCInflowSubsonic") return "TOTAL_CONDITIONS_INLET";
    if (source == "BCOutflow" || source == "BCOutflowSubsonic") return "PRESSURE_OUTLET";
    if (source == "BCWall" || source == "BCWallViscous") return "ADIABATIC_WALL";
    if (source == "BCWallViscousIsothermal") return "ISOTHERMAL_WALL";
    if (source == "BCWallInviscid" || source == "BCSymmetryPlane") return "SYMMETRY";
    // Farfield is not implemented; ambiguous labels must not silently select a physical condition.
    return {};
}

/// @brief Writes a TOML basic string, escaping quotes, backslashes and ASCII controls.
void quoted(std::ostream& out, std::string_view value)
{
    constexpr char hex[] = "0123456789abcdef";
    out.put('"');
    for (const char byte : value) {
        const auto c = static_cast<unsigned char>(byte);
        if (c == '"' || c == '\\') { out.put('\\'); out.put(byte); }
        else if (c < 0x20 || c == 0x7f) {
            out << "\\u00"; out.put(hex[c >> 4]); out.put(hex[c & 15]);
        } else out.put(byte); // Import metadata must already contain valid UTF-8.
    }
    out.put('"');
}

/// @brief Emits only the commented parameter examples relevant to the suggested BC type.
void parameter_examples(std::ostream& out, std::string_view type)
{
    if (type == "TOTAL_CONDITIONS_INLET")
        out << "# total_pressure = 101325.0    # Total pressure [Pa]\n"
               "# total_temperature = 300.0    # Total temperature [K]\n"
               "# direction = [1.0, 0.0, 0.0] # Replace with a direction into the domain.\n"
               "# allow_choking = false       # Optional; default is false.\n";
    else if (type == "SUPERSONIC_INLET")
        out << "# pressure = 101325.0          # Static pressure [Pa]\n"
               "# temperature = 300.0         # Static temperature [K]\n"
               "# mach = 2.0                  # Speed / sound speed.\n"
               "# direction = [1.0, 0.0, 0.0] # Must give supersonic inward normal velocity.\n"
               "# Alternatively, replace mach and direction with velocity = [u, v, w].\n";
    else if (type == "PRESSURE_OUTLET")
        out << "# pressure = 101325.0          # Static back pressure [Pa]\n"
               "# Reverse flow is rejected unless an explicit backflow table is supplied.\n";
    else if (type == "ISOTHERMAL_WALL")
        out << "# temperature = 300.0         # Wall temperature [K]\n"
               "# velocity = [0.0, 0.0, 0.0]  # Optional wall velocity; default is zero.\n";
    else if (type == "ADIABATIC_WALL")
        out << "# velocity = [0.0, 0.0, 0.0]  # Optional wall velocity; default is zero.\n";
    else out << "# No additional parameters in this template type.\n";
}

} // namespace

/// @brief Exports replicated patch metadata locally, checking both open and final close errors.
void generate_bc_template_config(const mesh::host::MeshPart& mesh, const std::filesystem::path& output,
                                 std::span<const std::uint64_t> global_patch_faces)
{
    if (output.empty() || mesh.rank != 0) return;
    if (mpi::is_active()) mpi::require_main_thread();
    mpi::require(global_patch_faces.size() == mesh.patches.size(), "BC template patch-count extent mismatch");
    mpi::require(mesh.n_labeled_bfaces_g <= mesh.n_bfaces_g, "BC template labelled boundary count exceeds total");
    const auto unlabelled_faces = mesh.n_bfaces_g - mesh.n_labeled_bfaces_g;
    std::string filename;
    mpi::guarded([&](){
        filename = output.string();
        std::ofstream out;
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.imbue(std::locale::classic());
        // Only the small optional configuration is written by rank 0; mesh data stays distributed.
        out.open(output, std::ios::out | std::ios::trunc);
        out << "# KRONA boundary-condition template\n"
               "# Edit before use: review types and uncomment/edit required parameters.\n"
               "# Import labels are hints, not complete physical boundary conditions.\n"
               "# Supported physical types: SYMMETRY, SLIP_WALL, ADIABATIC_WALL, ISOTHERMAL_WALL,\n"
               "# HEAT_FLUX_WALL, PRESSURE_OUTLET, TOTAL_CONDITIONS_INLET, VELOCITY_INLET,\n"
               "# SUPERSONIC_INLET, SUPERSONIC_OUTLET.\n"
               "# Total/supersonic inlets require a compressible model; use VELOCITY_INLET\n"
               "# for incompressible inflow. Isothermal and heat-flux walls require temperature.\n"
               "# Farfield and custom physical conditions cannot be selected by this parser.\n"
            << "# Patches: " << mesh.patches.size() << "\n"
            << "# Global boundary faces: " << mesh.n_bfaces_g << "\n"
            << "# Labelled faces: " << mesh.n_labeled_bfaces_g << "\n"
            << "# Unlabelled faces: " << unlabelled_faces
            << " (not represented by patch entries)\n\n";
        out << "# Additional scalars: add [boundary_condition.scalar.<field>] in EVERY patch entry.\n"
               "# Scalar types: FIXED_VALUE (value), NORMAL_GRADIENT (gradient),\n"
               "# FIXED_FLUX (flux), ZERO_GRADIENT (no parameter). Flux is positive outward.\n"
               "# For unlabelled faces, use [unlabelled.scalar.<field>]; physical conditions\n"
               "# use [unlabelled]. No condition is assigned automatically.\n"
               "# Do not add scalar.temperature when reading physical conditions: temperature\n"
               "# belongs to the physical model. Scalar-only runs may use scalar.temperature.\n"
               "# Keep physical parameters BEFORE nested scalar/backflow tables.\n\n";
        if (mesh.patches.empty()) out << "boundary_condition = []\n\n";
        for (std::size_t p = 0; p < mesh.patches.size(); ++p) {
            const auto& patch = mesh.patches[p];
            const auto type = suggested_type(patch.source_type);
            out << "[[boundary_condition]]\npatch_id = " << p << "\nname = ";
            quoted(out, patch.name); out << "\nfamily = "; quoted(out, patch.family);
            out << "\nsource_type = "; quoted(out, patch.source_type);
            out << "\nglobal_face_count = " << global_patch_faces[p] << "\n";
            if (type.empty()) {
                if (patch.source_type == "BCFarfield")
                    out << "# No built-in physical farfield condition is implemented.\n";
                out << "# TODO: set type to a supported physical condition and add its parameters,\n"
                       "# or assign a custom condition through the C++ manager.\n"
                       "# No physical type has been selected for this patch.\n";
            } else {
                out << "type = "; quoted(out, type); out << "\n";
                parameter_examples(out, type);
            }
            out << "# Optional additional scalar (repeat for each scalar; rename tracer as needed):\n"
                   "# [boundary_condition.scalar.tracer]\n"
                   "# type = \"ZERO_GRADIENT\"\n\n";
        }
        if (unlabelled_faces != 0) {
            out << "[unlabelled]\nglobal_face_count = " << unlabelled_faces << "\n"
                   "# TODO: choose a physical condition for these external faces.\n"
                   "# Example only: a stationary adiabatic wall; review before uncommenting.\n"
                   "# type = \"ADIABATIC_WALL\"\n"
                   "# velocity = [0.0, 0.0, 0.0]\n"
                   "# Each additional scalar also needs an explicit condition here.\n"
                   "# [unlabelled.scalar.tracer]\n"
                   "# type = \"ZERO_GRADIENT\"\n\n";
        }
        // close() also exposes buffered-write failures, such as a full filesystem.
        out.close();
    });
}

} // namespace krona::io::boundary_config