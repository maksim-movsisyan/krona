#include <krona/mesh/bc_config.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/error.hpp>

#include <cstdio>
#include <fstream>
#include <locale>
#include <ostream>
#include <string>
#include <string_view>

namespace krona::mesh {
namespace {

/// @brief Suggests a solver template type from an import label; physics still needs user review.
[[nodiscard]] std::string_view suggested_type(std::string_view source) noexcept
{
    if (source == "BCInflowSupersonic") return "INLET_SUPERSONIC";
    if (source == "BCOutflowSupersonic") return "OUTLET_SUPERSONIC";
    if (source == "BCInflow" || source == "BCInflowSubsonic") return "INLET_TOTAL_TP";
    if (source == "BCOutflow" || source == "BCOutflowSubsonic") return "OUTLET_BACK_PRESSURE";
    if (source == "BCWall" || source == "BCWallViscous") return "WALL_ADIABATIC";
    if (source == "BCWallViscousIsothermal") return "WALL_ISOTHERMAL";
    if (source == "BCWallInviscid" || source == "BCSymmetryPlane") return "SYMMETRY";
    if (source == "BCFarfield") return "FARFIELD";
    // A specified heat flux is not necessarily zero; ambiguous labels need an explicit choice.
    return "USER_DEFINED";
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
    if (type == "INLET_TOTAL_TP")
        out << "# p0 = 101325.0              # Total pressure [Pa]\n"
               "# T0 = 300.0                 # Total temperature [K]\n"
               "# direction = [1.0, 0.0, 0.0] # Unit flow direction\n";
    else if (type == "INLET_SUPERSONIC")
        out << "# p = 101325.0               # Static pressure [Pa]\n"
               "# T = 300.0                  # Static temperature [K]\n"
               "# M = 2.0                    # Mach number\n"
               "# direction = [1.0, 0.0, 0.0] # Unit flow direction\n";
    else if (type == "OUTLET_BACK_PRESSURE")
        out << "# p_back = 101325.0          # Static back pressure [Pa]\n";
    else if (type == "WALL_ISOTHERMAL")
        out << "# T_wall = 300.0             # Wall temperature [K]\n";
    else if (type == "FARFIELD")
        out << "# mach_inf = 0.8\n# p_inf = 101325.0          # Static pressure [Pa]\n"
               "# T_inf = 300.0              # Static temperature [K]\n"
               "# alpha = 0.0                # Angle of attack [deg]\n"
               "# beta = 0.0                 # Sideslip angle [deg]\n";
    else if (type == "USER_DEFINED")
        out << "# Select a solver BC type and add its required parameters.\n";
    else out << "# No additional parameters in this template type.\n";
}

} // namespace

/// @brief Exports replicated patch metadata locally, checking both open and final close errors.
void generate_bc_template_config(const MeshPart& mesh, const std::filesystem::path& output,
                                 std::span<const std::uint64_t> global_patch_faces)
{
    if (output.empty() || mesh.rank != 0) return;
    if (mpi::is_active()) mpi::require_main_thread();
    mpi::require(global_patch_faces.size() == mesh.patches.size(), "BC template patch-count extent mismatch");
    std::string filename;
    mpi::guarded([&](){
        filename = output.string();
        std::ofstream out;
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.imbue(std::locale::classic());
        // Only the small optional configuration is written by rank 0; mesh data stays distributed.
        out.open(output, std::ios::out | std::ios::trunc);
        out << "# KRONA boundary-condition template\n"
               "# Review every suggested type and uncomment/edit its required parameters.\n"
               "# Import labels are hints, not complete physical boundary conditions.\n"
               "# Solver type names: INLET_TOTAL_TP, INLET_SUPERSONIC, OUTLET_BACK_PRESSURE,\n"
               "# OUTLET_SUPERSONIC, WALL_ADIABATIC, WALL_ISOTHERMAL, SYMMETRY, FARFIELD, USER_DEFINED.\n"
            << "# Patches: " << mesh.patches.size() << "\n"
            << "# Global boundary faces: " << mesh.n_bfaces_g << "\n"
            << "# Labelled faces: " << mesh.n_labeled_bfaces_g << "\n"
            << "# Unlabelled faces: " << mesh.n_bfaces_g - mesh.n_labeled_bfaces_g
            << " (not represented by patch entries)\n\n";
        for (std::size_t p = 0; p < mesh.patches.size(); ++p) {
            const auto& patch = mesh.patches[p];
            const auto type = suggested_type(patch.source_type);
            out << "[[boundary_condition]]\npatch_id = " << p << "\nname = ";
            quoted(out, patch.name); out << "\nfamily = "; quoted(out, patch.family);
            out << "\nsource_type = "; quoted(out, patch.source_type);
            out << "\nglobal_face_count = " << global_patch_faces[p] << "\ntype = ";
            quoted(out, type); out << "\n";
            parameter_examples(out, type); out << '\n';
        }
        // close() also exposes buffered-write failures, such as a full filesystem.
        out.close();
    });
}

} // namespace krona::mesh
