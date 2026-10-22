#pragma once

#include <krona/transport/scalar/convection/plan.hpp>
#include <krona/transport/scalar/convection/types.hpp>
#include <krona/numerics/reconstruction/mesh_stencil.hpp>

namespace krona::transport::scalar::convection {
 
/// @brief Borrowed high-order increment of one selected component of a reconstruction Data batch.
/// Stencil supplies face<Scheme>(f) (or face(f)) and boundary_side(f).
/// QUICK requires explicit ThreePointStencil sides, including an explicit outflow stencil.
/// Inputs include all accessed MPI ghosts/stencil samples and must not alias written targets.
/// Geometry and sample indices must satisfy the selected reconstruction's stencil contract.
template<class Scheme, class Data, class Stencil, class OutflowScheme = Scheme>
struct Reconstruction {
    Scheme scheme;
    Data data;
    Stencil stencil;
    OutflowScheme outflow;
    std::size_t component = 0;

    template<class Real>
    KOKKOS_INLINE_FUNCTION Real interior(const InteriorFaceContext<Real>& c, bool from_owner) const
    {
        const auto face = face_stencil(static_cast<std::size_t>(c.face));
        if (from_owner) return scheme.template increment<Real>(face.left, data, component);
        return scheme.template increment<Real>(face.right, data, component);
    }
    template<class Real>
    KOKKOS_INLINE_FUNCTION Real boundary(const boundary::FaceContext<Real>& c) const
    { return outflow.template increment<Real>(stencil.boundary_side(static_cast<std::size_t>(c.face)), data, component); }
private:
    KOKKOS_INLINE_FUNCTION auto face_stencil(std::size_t f) const
    {
        if constexpr (requires { stencil.template face<Scheme>(f); })
            return stencil.template face<Scheme>(f);
        else return stencil.face(f);
    }
};

/// @brief Explicit one-sided outflow scheme; it may not read nonexistent exterior samples.
template<class Exec, class Real, class Scheme, class Data, class OutflowScheme>
    requires requires { OutflowScheme::requirements; }
auto make_reconstruction(const ConvectionPlan<Exec, Real>& plan, Scheme scheme, Data data,
    OutflowScheme outflow, std::size_t component = 0)
{
    namespace rec = numerics::reconstruction;
    static_assert(!std::is_same_v<Scheme, rec::Quick>, "QUICK requires an explicit stencil provider");
    static_assert((OutflowScheme::requirements & rec::OtherValues) == 0,
        "A mesh outflow trace must not read a boundary slot; use an explicit one-sided stencil provider");
    using Stencil = rec::MeshStencil<typename Exec::memory_space, Real>;
    return Reconstruction<Scheme, Data, Stencil, OutflowScheme>{scheme, data,
        Stencil(plan.mesh_view(), plan.connectivity_view()), outflow, component};
}

/// @brief Geometry-only adapter. MUSCL/TVD use a one-sided linear outflow trace with Data's
/// cell factor (UnitFactors if absent); no FixedValue/extended boundary sample enters that trace.
/// Other schemes use the same polynomial at outflow. Override the outflow scheme explicitly above.
template<class Exec, class Real, class Scheme, class Data>
auto make_reconstruction(const ConvectionPlan<Exec, Real>& plan, Scheme scheme, Data data,
    std::size_t component = 0)
{
    namespace rec = numerics::reconstruction;
    if constexpr ((Scheme::requirements & rec::OtherValues) != 0)
        return make_reconstruction(plan, scheme, data, rec::LimitedLinear{}, component);
    else return make_reconstruction(plan, scheme, data, scheme, component);
}

/// @brief Explicit stencil extension, e.g. QUICK or a larger unstructured reconstruction stencil.
template<class Scheme, class Data, class Stencil>
    requires requires { Scheme::requirements; }
auto make_reconstruction(Scheme scheme, Data data, Stencil stencil, std::size_t component = 0)
{ return Reconstruction<Scheme, Data, Stencil>{scheme, data, stencil, scheme, component}; }

namespace detail {
template<class R> inline constexpr bool reconstructed = !std::is_same_v<R, NoReconstruction>;

template<class Exec, class Input>
void validate_reconstruction_input(const Input& input, std::size_t cells)
{
    if constexpr (requires { typename Input::memory_space; })
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Input::memory_space>::accessible);
    if constexpr (requires { input.size(); })
        mpi::require(input.size() >= cells, "Scalar convection: reconstruction input extent is too short");
}
template<class Exec, class R> void validate_reconstruction(const R&, std::size_t) {}
template<class Exec, class Scheme, class Data, class Stencil, class OutflowScheme>
void validate_reconstruction(const Reconstruction<Scheme, Data, Stencil, OutflowScheme>& r, std::size_t cells)
{
    namespace rec = numerics::reconstruction;
    mpi::require(r.component < Data::components, "Scalar convection: invalid reconstruction component");
    if constexpr (requires { typename Stencil::memory_space; })
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Stencil::memory_space>::accessible);
    validate_reconstruction_input<Exec>(r.data.values, cells);
    constexpr auto requirements = Scheme::requirements | OutflowScheme::requirements;
    if constexpr ((requirements & rec::Gradients) != 0)
        validate_reconstruction_input<Exec>(r.data.gradients, cells);
    if constexpr ((requirements & rec::Factors) != 0)
        validate_reconstruction_input<Exec>(r.data.factors, cells);
    if constexpr ((requirements & rec::Hessians) != 0)
        validate_reconstruction_input<Exec>(r.data.hessians, cells);
    if constexpr ((requirements & rec::Moments) != 0)
        validate_reconstruction_input<Exec>(r.data.moments, cells);
}
} // namespace detail

} // namespace krona::transport::scalar::convection
