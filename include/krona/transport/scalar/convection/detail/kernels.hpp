#pragma once

#include <krona/boundary/geometry_view.hpp>
#include <krona/transport/scalar/convection/reconstruction.hpp>
#include <krona/transport/scalar/status.hpp>

namespace krona::transport::scalar::convection::detail {
using scalar::detail::flag;
using scalar::detail::add_checked;
using scalar::detail::atomic_add_checked;
using scalar::detail::BadMassFlux;
using scalar::detail::BadInflow;
using scalar::detail::BadReconstruction;
using scalar::detail::BadSystem;

template<class Space>
struct RowData { 
    mesh::compute::MeshView<Space> mesh;
    mesh::compute::AuxConnectivityView<Space> connectivity;
};
template<class Real, class Space>
KOKKOS_INLINE_FUNCTION auto context(const RowData<Space>& d, LocalIndex face, Real time)
{
    const auto f = static_cast<std::size_t>(face);
    const auto& g = d.mesh.geometry;
    return InteriorFaceContext<Real>{face, d.mesh.connectivity.face_owner(f), d.mesh.connectivity.face_neigh(f),
        static_cast<Real>(g.face_centroid_x(f)), static_cast<Real>(g.face_centroid_y(f)), static_cast<Real>(g.face_centroid_z(f)),
        static_cast<Real>(g.face_normal_x(f)), static_cast<Real>(g.face_normal_y(f)), static_cast<Real>(g.face_normal_z(f)),
        static_cast<Real>(g.face_area(f)), time};
}
template<class Real, class Flux, class Context, class Flags>
KOKKOS_INLINE_FUNCTION Real mass_flux(const Flux& flux, const Context& c, Flags flags)
{
    const Real m = static_cast<Real>(flux(c));
    if (!Kokkos::isfinite(m)) { flag(flags, BadMassFlux); return Real{0}; }
    return m;
}
template<class Value, class Real, class Flags>
KOKKOS_INLINE_FUNCTION bool representable_flux(Real m, Flags flags)
{
    const auto value = static_cast<Value>(m);
    if (!Kokkos::isfinite(value) || value == Value{0}) { flag(flags, BadSystem); return false; }
    return true;
}
template<class Real, class R, class Context, class Flags>
KOKKOS_INLINE_FUNCTION Real interior_increment(const R& reconstruction, const Context& c, Real m, Flags flags)
{
    if constexpr (!reconstructed<R>) return Real{0};
    else {
        const Real delta = static_cast<Real>(reconstruction.interior(c, m > Real{0}));
        if (!Kokkos::isfinite(delta)) { flag(flags, BadReconstruction); return Real{0}; }
        return delta;
    }
}
template<class Real, class R, class Context, class Flags>
KOKKOS_INLINE_FUNCTION Real boundary_increment(const R& reconstruction, const Context& c, Flags flags)
{
    if constexpr (!reconstructed<R>) return Real{0};
    else {
        const Real delta = static_cast<Real>(reconstruction.boundary(c));
        if (!Kokkos::isfinite(delta)) { flag(flags, BadReconstruction); return Real{0}; }
        return delta;
    }
}
template<class Real, class Inflow, class Context, class Condition, class Flags>
KOKKOS_INLINE_FUNCTION Real inflow_value(const Inflow& inflow, const Context& c, const Condition& condition, Flags flags)
{
    const Real value = static_cast<Real>(inflow(c, condition));
    if (!Kokkos::isfinite(value)) { flag(flags, BadInflow); return Real{0}; }
    return value;
}

// One iteration owns one row. F is outward from this row; m keeps canonical face orientation.
template<class Space, class Real, class Target, class Flux, class Reconstruction, class Flags>
struct AddRows {
    RowData<Space> data; Target target; Flux flux; Reconstruction reconstruction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const auto i = static_cast<std::size_t>(cell);
        using Value = typename decltype(target.values)::non_const_value_type;
        Real diagonal = 0, rhs = 0;
        const auto& conn = data.connectivity;
        for (auto k = conn.cell_faces_offsets(i); k < conn.cell_faces_offsets(i+1); ++k) {
            const auto face = conn.cell_faces(k);
            if (face >= data.mesh.sizes.n_inner_faces) continue;
            const auto c = context<Real>(data, face, time);
            const Real m = mass_flux<Real>(flux, c, flags);
            if (m == Real{0} || !representable_flux<Value>(m, flags)) continue;
            const bool owner = c.owner == cell;
            const Real outward = owner ? m : -m;
            if (outward > Real{0}) diagonal += outward;
            else {
                const auto f = static_cast<std::size_t>(face);
                const auto entry = owner ? target.entries.owner_entries(f) : target.entries.neighbour_entries(f);
                add_checked(target.values, entry, outward, flags);
            }
            if constexpr (reconstructed<Reconstruction>)
                rhs -= outward*interior_increment<Real>(reconstruction, c, m, flags);
        }
        add_checked(target.values, target.entries.diagonal(i), diagonal, flags);
        if constexpr (reconstructed<Reconstruction>) add_checked(target.rhs, i, rhs, flags);
    }
};

template<class Space, class Real, class Target, class Condition, class Flux, class Reconstruction, class Inflow, class Flags>
struct AddBoundary {
    boundary::GeometryView<Space> geometry; Target target; Condition condition; Flux flux;
    Reconstruction reconstruction; Inflow inflow; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const Real m = mass_flux<Real>(flux, c, flags);
        using Value = typename decltype(target.values)::non_const_value_type;
        if (m == Real{0} || !representable_flux<Value>(m, flags)) return;
        const auto i = static_cast<std::size_t>(c.owner);
        if (m > Real{0}) {
            atomic_add_checked(target.values, target.entries.diagonal(i), m, flags);
            if constexpr (reconstructed<Reconstruction>)
                atomic_add_checked(target.rhs, i, -m*boundary_increment<Real>(reconstruction, c, flags), flags);
        } else atomic_add_checked(target.rhs, i, -m*inflow_value<Real>(inflow, c, condition, flags), flags);
    }
};

template<class Space, class Real, class Field, class Output, class Flux, class Reconstruction, class Flags>
struct ResidualRows {
    RowData<Space> data; Field field; Output output; Flux flux; Reconstruction reconstruction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const auto i = static_cast<std::size_t>(cell);
        Real residual = 0;
        const auto& conn = data.connectivity;
        for (auto k = conn.cell_faces_offsets(i); k < conn.cell_faces_offsets(i+1); ++k) {
            const auto face = conn.cell_faces(k);
            if (face >= data.mesh.sizes.n_inner_faces) continue;
            const auto c = context<Real>(data, face, time);
            const Real m = mass_flux<Real>(flux, c, flags);
            if (m == Real{0}) continue;
            const auto donor = m > Real{0} ? c.owner : c.neighbour;
            const Real value = static_cast<Real>(field(static_cast<std::size_t>(donor))) +
                interior_increment<Real>(reconstruction, c, m, flags);
            residual -= (c.owner == cell ? m : -m)*value;
        }
        add_checked(output, i, residual, flags);
    }
};
template<class Space, class Real, class Field, class Output, class Condition, class Flux, class Reconstruction, class Inflow, class Flags>
struct ResidualBoundary {
    boundary::GeometryView<Space> geometry; Field field; Output output; Condition condition; Flux flux;
    Reconstruction reconstruction; Inflow inflow; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const Real m = mass_flux<Real>(flux, c, flags);
        if (m == Real{0}) return;
        const auto i = static_cast<std::size_t>(c.owner);
        const Real value = m > Real{0} ? static_cast<Real>(field(i))+boundary_increment<Real>(reconstruction, c, flags) :
            inflow_value<Real>(inflow, c, condition, flags);
        atomic_add_checked(output, i, -m*value, flags);
    }
};

template<class Space, class Real, class Output, class Flux, class Reconstruction, class Flags>
struct CorrectionRows {
    RowData<Space> data; Output output; Flux flux; Reconstruction reconstruction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const
    {
        const auto i = static_cast<std::size_t>(cell);
        Real rhs = 0;
        const auto& conn = data.connectivity;
        for (auto k = conn.cell_faces_offsets(i); k < conn.cell_faces_offsets(i+1); ++k) {
            const auto face = conn.cell_faces(k);
            if (face >= data.mesh.sizes.n_inner_faces) continue;
            const auto c = context<Real>(data, face, time);
            const Real m = mass_flux<Real>(flux, c, flags);
            if (m != Real{0}) rhs -= (c.owner == cell ? m : -m)*interior_increment<Real>(reconstruction, c, m, flags);
        }
        add_checked(output, i, rhs, flags);
    }
};
template<class Space, class Real, class Output, class Flux, class Reconstruction, class Flags>
struct CorrectionBoundary {
    boundary::GeometryView<Space> geometry; Output output; Flux flux; Reconstruction reconstruction; Flags flags; Real time;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex face) const
    {
        const auto c = geometry.template context<Real>(face, time);
        const Real m = mass_flux<Real>(flux, c, flags);
        if (m > Real{0}) atomic_add_checked(output, static_cast<std::size_t>(c.owner),
            -m*boundary_increment<Real>(reconstruction, c, flags), flags);
    }
};
} // namespace krona::transport::scalar::convection::detail
