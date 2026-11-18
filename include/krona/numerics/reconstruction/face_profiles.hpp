#pragma once

#include <krona/numerics/reconstruction/detail/validation.hpp>
#include <krona/numerics/reconstruction/evaluate.hpp>
#include <krona/numerics/reconstruction/schemes/limited.hpp>

namespace krona::numerics::reconstruction {

/// @brief Binds the formulas used at inner and physical faces to one field and stencil.
/// @tparam Stencil Provider of face-side geometry or explicitly prepared samples.
/// @tparam Values Provider of cell values in component order, including any samples read by the formulas.
/// @tparam Inner Formula used on both sides of a cell-to-cell face.
/// @tparam Boundary Formula that reconstructs the face value from the adjacent cell
/// at a boundary of the computational domain.
/// @details The same object can prepare cell factors and evaluate the face traces.
/// Boundary describes the interior trace. The transport operator applies the physical condition.
template <class Stencil, class Values, class Inner, class Boundary>
struct FaceProfiles {
    Stencil stencil;
    Values values;
    Inner inner;
    Boundary boundary;

    /// @brief Evaluates both sides of an inner face, for example before a Riemann solve.
    /// @tparam Real Arithmetic type of the traces.
    template <class Real>
    KOKKOS_INLINE_FUNCTION auto inner_values(std::size_t face) const {
        return evaluate<Real>(inner, inner_geometry(face), values);
    }

    /// @brief Evaluates just the cell-side trace at a physical face.
    /// @tparam Real Arithmetic type of the trace.
    template <class Real>
    KOKKOS_INLINE_FUNCTION auto boundary_values(std::size_t face) const {
        return evaluate_side<Real>(boundary, boundary_geometry(face), values);
    }

    /// @brief Returns the change relative to the selected cell value, as used by scalar upwinding.
    /// @tparam Real Arithmetic type of the increment.
    template <class Real>
    KOKKOS_INLINE_FUNCTION Real inner_increment(std::size_t face, bool from_owner,
                                                std::size_t component) const {
        if constexpr (requires { stencil.template inner_side<Inner::geometry>(face, from_owner); }) {
            const auto side = stencil.template inner_side<Inner::geometry>(face, from_owner);
            return inner.template increment<Real>(side, values, component);
        } else {
            const auto sides = inner_geometry(face);
            if (from_owner)
                return inner.template increment<Real>(sides.left, values, component);
            return inner.template increment<Real>(sides.right, values, component);
        }
    }

    /// @brief Returns the change relative to the adjacent cell value at a physical face.
    /// @tparam Real Arithmetic type of the increment.
    template <class Real>
    KOKKOS_INLINE_FUNCTION Real boundary_increment(std::size_t face, std::size_t component) const {
        return boundary.template increment<Real>(boundary_geometry(face), values, component);
    }

    /// @brief Number of bound samples around this owned cell.
    KOKKOS_INLINE_FUNCTION std::size_t sample_count(std::size_t cell) const {
        return stencil.size(cell);
    }

    /// @brief Number of face points to check for this owned cell.
    KOKKOS_INLINE_FUNCTION std::size_t point_count(std::size_t cell) const {
        return stencil.size(cell);
    }

    /// @brief Visits the exact side and formula used by the consuming flux at this incident face.
    template <class Visitor>
    KOKKOS_INLINE_FUNCTION void visit_point(std::size_t cell, std::size_t slot, const Visitor& visit) const {
        const auto face = stencil.incident_face(cell, slot);
        if (stencil.is_boundary(face)) {
            visit(boundary, boundary_geometry(face));
        } else {
            const bool from_owner = stencil.is_owner(face, cell);
            if constexpr (requires { stencil.template inner_side<Inner::geometry>(face, from_owner); }) {
                visit(inner, stencil.template inner_side<Inner::geometry>(face, from_owner));
            } else {
                const auto sides = inner_geometry(face);
                if (from_owner)
                    visit(inner, sides.left);
                else
                    visit(inner, sides.right);
            }
        }
    }

    /// @brief Checks the host-visible field extents before a transport launch.
    /// @tparam Exec Execution space that will read the fields and stencil.
    /// @param cells Number of real cell entries, including MPI ghosts.
    /// @param value_extent Minimum value extent required by the transport operator itself.
    /// @param with_boundary Whether this launch can evaluate physical faces.
    /// @param diagnostic Error message used when any required extent is too short.
    template <class Exec>
    void validate(std::size_t cells, std::size_t value_extent, bool with_boundary,
                  const char* diagnostic) const {
        detail::require_accessible<Exec, Stencil>();
        if constexpr (Inner::geometry == Geometry::Explicit) {
            value_extent = Kokkos::max(value_extent, stencil.required_input_size());
        }
        if constexpr (Boundary::geometry == Geometry::Explicit || Boundary::geometry == Geometry::Line) {
            if (with_boundary)
                value_extent = Kokkos::max(value_extent, stencil.required_input_size());
        }
        detail::validate_extent<Exec>(values, Kokkos::max(cells, value_extent), diagnostic);
        const auto message = [diagnostic](const char*) { return diagnostic; };
        detail::validate_formula<Exec, Values::components>(inner, cells, message);
        if (with_boundary)
            detail::validate_formula<Exec, Values::components>(boundary, cells, message);
    }

    /// @brief Visits the borrowed formula fields so an operator can check output aliasing.
    template <class Visitor>
    void visit_inputs(Visitor&& visit, bool with_boundary) const {
        detail::visit_inputs(inner, visit);
        if (with_boundary)
            detail::visit_inputs(boundary, visit);
    }

private:
    KOKKOS_INLINE_FUNCTION auto inner_geometry(std::size_t face) const {
        if constexpr (requires { stencil.template face<Inner::geometry>(face); })
            return stencil.template face<Inner::geometry>(face);
        else
            return stencil.face(face);
    }

    KOKKOS_INLINE_FUNCTION auto boundary_geometry(std::size_t face) const {
        if constexpr (requires { stencil.template boundary_side<Boundary::geometry>(face); })
            return stencil.template boundary_side<Boundary::geometry>(face);
        else
            return stencil.boundary_side(face);
    }
};

/// @brief Uses the same formula at inner faces and on the cell side of physical faces.
/// @tparam Stencil Provider of both kinds of face-side geometry.
/// @tparam Values Provider of base cell values.
/// @tparam Formula Concrete formula with its borrowed fields, such as Linear{gradients}.
template <class Stencil, class Values, class Formula>
auto face_profiles(Stencil stencil, Values values, Formula formula) {
    return FaceProfiles{stencil, values, formula, formula};
}

/// @brief Uses a separate formula on the cell side of physical faces.
/// @tparam Stencil Provider of both kinds of face-side geometry.
/// @tparam Values Provider of base cell values.
/// @tparam Inner Formula used between cells.
/// @tparam Boundary Formula used at physical faces.
template <class Stencil, class Values, class Inner, class Boundary>
auto face_profiles(Stencil stencil, Values values, Inner inner, Boundary boundary) {
    return FaceProfiles{stencil, values, inner, boundary};
}

/// @brief Applies the factors prepared for these profiles to both formulas.
/// @tparam Stencil Type of the prepared face stencil.
/// @tparam Values Type of the shared base value provider.
/// @tparam Inner Bound inner-face formula.
/// @tparam Boundary Bound physical-face formula.
/// @tparam Factors Provider of one shared factor or one per variable in each local cell.
template <class Stencil, class Values, class Inner, class Boundary, class Factors>
auto limited(const FaceProfiles<Stencil, Values, Inner, Boundary>& profiles, Factors factors) {
    return face_profiles(profiles.stencil, profiles.values, Limited{profiles.inner, factors},
                         Limited{profiles.boundary, factors});
}

} // namespace krona::numerics::reconstruction
