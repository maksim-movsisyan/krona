#pragma once

#include <krona/transport/euler/boundary.hpp>
#include <krona/transport/euler/detail/stencil_data.hpp>
#include <krona/transport/euler/reconstruction.hpp>
#include <krona/transport/euler/state.hpp>
#include <krona/transport/euler/status.hpp>

namespace krona::transport::euler::detail {

/// @brief Keeps the input primitive and its prepared Riemann state together.
/// @tparam R Scalar type of both states.
template<class R>
struct CellState {
    boundary::physical::Primitive<R, true> primitive;
    numerics::riemann::State<R> state;
};

/// @brief Shared state preparation and face flux evaluation used by the assembly kernels.
/// @tparam R Scalar used to evaluate traces and fluxes.
/// @tparam Reconstruction Bound face formulas and primitive field pack.
/// @tparam StateLaw Converts a primitive to a state the Riemann solver can use.
/// @tparam Solver Computes flux per unit area for the supplied normal.
/// @tparam Flags Writable batch diagnostics.
template<class R, class Reconstruction, class StateLaw, class Solver, class Flags>
struct Physics {
    Reconstruction reconstruction;
    StateLaw state_law;
    Solver solver;
    Flags flags;

    KOKKOS_INLINE_FUNCTION auto prepare(boundary::physical::Primitive<R, true> q) const {
        return prepare_state(state_law, q);
    }

    KOKKOS_INLINE_FUNCTION auto base(std::size_t cell) const {
        const auto q = primitive<R>(reconstruction.values(), cell);
        return CellState<R>{q, prepare(q)};
    }

    // Fall back on the entire trace so the solver receives a consistent thermodynamic state.
    KOKKOS_INLINE_FUNCTION auto reconstructed(const Kokkos::Array<R, 5>& values, const CellState<R>& cell) const {
        const auto result = prepare(primitive(values));
        if (state_valid(result)) return result;
        count(flags, ReconstructionFallbacks);
        return cell.state;
    }

    /// @brief Evaluates the oriented flux through one inner face.
    /// @tparam Space Memory space of the borrowed mesh arrays.
    /// @param data Owner, neighbour and normal arrays.
    /// @param f Local face ID.
    /// @param cell Cell whose row is being assembled.
    /// @param current Prepared state of that row's cell, reused across all its faces.
    template<class Space>
    KOKKOS_INLINE_FUNCTION auto inner(const StencilData<Space>& data, std::size_t f, std::size_t cell,
                                      const CellState<R>& current) const {
        namespace ri = numerics::riemann;
        const std::size_t owner = static_cast<std::size_t>(data.owner(f));
        const std::size_t neighbour = static_cast<std::size_t>(data.neighbour(f));
        const auto left = owner == cell ? current : base(owner);
        const auto right = neighbour == cell ? current : base(neighbour);
        if (!state_valid(left.state) || !state_valid(right.state)) {
            flag(flags, BadState);
            return ri::Result<R>{{}, R{0}, ri::Status::InvalidState};
        }

        auto l = left.state, r = right.state;
        if constexpr (!std::is_same_v<typename Reconstruction::inner_formula, numerics::reconstruction::FirstOrder>) {
            const auto traces = reconstruction.template interior<R>(f);
            l = reconstructed(traces.left, left);
            r = reconstructed(traces.right, right);
        }
        const auto result = solver(l, r, ri::Vector<R>{static_cast<R>(data.nx(f)), static_cast<R>(data.ny(f)),
                                                        static_cast<R>(data.nz(f))});
        check_flux(result);
        return result;
    }

    KOKKOS_INLINE_FUNCTION void check_flux(const numerics::riemann::Result<R>& result) const {
        if (!result.valid()) flag(flags, BadFlux);
        else if (result.used_fallback()) count(flags, RiemannFallbacks);
    }

    /// @brief Builds the exterior primitive state for the boundary Riemann problem.
    /// @tparam Model Physical boundary model, or a provider of a model for each face.
    /// @tparam Condition Concrete boundary condition assigned to this face's patch.
    /// @tparam Closure Callable that supplies the exterior state, such as EvaluatedExterior.
    /// @param c Face geometry and time used to evaluate the boundary condition.
    /// @param condition Boundary condition to apply, such as SlipWall or PressureOutlet.
    /// @param model Shared model passed directly to the closure unless it provides at(c, condition).
    /// @param closure Chooses how to obtain the exterior state from the boundary condition.
    /// @param inside Owner state reconstructed to the face, or the cell-centre state after fallback.
    /// @param face Previously prepared boundary value, used by closures such as PreparedExterior.
    /// @return Exterior primitive state (p, u, v, w, T) for the right side of the Riemann problem.
    template<class Model, class Condition, class Closure>
    KOKKOS_INLINE_FUNCTION auto exterior(const boundary::FaceContext<R>& c, const Condition& condition,
                                        const Model& model, const Closure& closure,
                                        boundary::physical::Primitive<R, true> inside,
                                        boundary::physical::Primitive<R, true> face) const {
        if constexpr (requires { model.at(c, condition); })
            return closure(c, condition, inside, face, model.at(c, condition));
        else
            return closure(c, condition, inside, face, model);
    }

    /// @brief Selects the interior state for a boundary face.
    /// The residual and boundary Jacobian use the same selection and fallback.
    /// @param face Local mesh index of the boundary face.
    /// @param current Owner's cell-centre primitive values and prepared Riemann state.
    /// @return Primitive values and the corresponding Riemann state on the interior side.
    KOKKOS_INLINE_FUNCTION CellState<R> boundary_trace(std::size_t face, const CellState<R>& current) const {
        if constexpr (!std::is_same_v<typename Reconstruction::boundary_formula, numerics::reconstruction::FirstOrder>) {
            // Reconstruct the owner's primitive values at the face and prepare them for the solver.
            const auto q = primitive(reconstruction.template boundary<R>(face));
            const auto candidate = prepare(q);
            if (state_valid(candidate)) return {q, candidate};

            // Reject the whole reconstructed state if any of its values are unusable.
            count(flags, ReconstructionFallbacks);
        }

        // First order and rejected reconstructions both use the cell-centre state.
        return current;
    }

    /// @brief Loads a physical-face slot only when the consuming policy needs it.
    /// @tparam Read Whether the policy uses the prepared face argument.
    /// @param slot Index of the face value in the extended primitive pack.
    template<bool Read>
    KOKKOS_INLINE_FUNCTION auto prepared_face(std::size_t slot) const {
        if constexpr (Read) return primitive<R>(reconstruction.values(), slot);
        else return boundary::physical::Primitive<R, true>{};
    }

    /// @brief Evaluates the outward flux through one exterior boundary face.
    /// @tparam Model Physical boundary model, optionally with face-specific data.
    /// @tparam Condition Condition assigned to the face's patch.
    /// @tparam Closure Policy that produces the exterior state.
    /// @param c Face context with geometry and time.
    /// @param condition Typed condition to evaluate.
    /// @param model Model used by the closure.
    /// @param closure Policy used to build the exterior trace.
    /// @param current Owner's prepared cell-centre state.
    template<class Model, class Condition, class Closure>
    KOKKOS_INLINE_FUNCTION auto boundary_flux(const boundary::FaceContext<R>& c, const Condition& condition,
                                              const Model& model, const Closure& closure, const CellState<R>& current) const {
        namespace ri = numerics::riemann;
        const auto trace = boundary_trace(static_cast<std::size_t>(c.face), current);
        const auto face = prepared_face<uses_prepared_face<Closure>>(c.extended_index);
        const auto outside = exterior(c, condition, model, closure, trace.primitive, face);
        const auto right = prepare(outside);
        if (!state_valid(right)) {
            flag(flags, BadBoundary);
            return ri::Result<R>{{}, R{0}, ri::Status::InvalidState};
        }
        const auto result = solver(trace.state, right, ri::Vector<R>{c.nx, c.ny, c.nz});
        check_flux(result);
        return result;
    }
};

} // namespace krona::transport::euler::detail
