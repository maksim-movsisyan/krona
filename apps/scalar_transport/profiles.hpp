#pragma once
#include "config.hpp"
#include <krona/execution/types.hpp>
#include <krona/mesh/host/mesh_part.hpp>
#include <krona/mpi/error.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <variant>

namespace krona::apps::scalar_transport {
struct ConstantProfile {
    double initial = 0, rate = 0;
    template<class R>
    KOKKOS_INLINE_FUNCTION R point(R, R, R, R time) const
    { return static_cast<R>(initial)+static_cast<R>(rate)*time; }
    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const
    { return point(c.x, c.y, c.z, c.time); }
    double average(const std::array<double, 3>&, const std::array<double, 3>&, double time) const
    { return point(0.0, 0.0, 0.0, time); }
};

// width is the initial standard deviation. This is an exact continuum solution
// only for the application's constant coefficients, uniform velocity and source.
struct GaussianProfile {
    double background = 0, amplitude = 1, width = 0.1, diffusivity = 0, rate = 0;
    Kokkos::Array<double, 3> centre{}, velocity{};
    template<class R>
    KOKKOS_INLINE_FUNCTION R point(R x, R y, R z, R time) const
    {
        const R sigma = static_cast<R>(width);
        const R variance = sigma*sigma+R{2}*static_cast<R>(diffusivity)*time;
        const R spread = Kokkos::sqrt(variance);
        const R dx = (x-static_cast<R>(centre[0])-static_cast<R>(velocity[0])*time)/spread;
        const R dy = (y-static_cast<R>(centre[1])-static_cast<R>(velocity[1])*time)/spread;
        const R dz = (z-static_cast<R>(centre[2])-static_cast<R>(velocity[2])*time)/spread;
        const R scale = sigma/spread;
        return static_cast<R>(background)+static_cast<R>(rate)*time+
            static_cast<R>(amplitude)*scale*scale*scale*Kokkos::exp(-R{0.5}*(dx*dx+dy*dy+dz*dz));
    }
    template<class Context>
    KOKKOS_INLINE_FUNCTION auto operator()(const Context& c) const
    { return point(c.x, c.y, c.z, c.time); }
    double average(const std::array<double, 3>& lo, const std::array<double, 3>& hi, double time) const
    {
        const double spread = std::sqrt(2.0*(width*width+2.0*diffusivity*time));
        double product = amplitude;
        for (std::size_t d = 0; d < 3; ++d) {
            const double location = centre[d]+velocity[d]*time;
            const double a = (lo[d]-location)/spread, b = (hi[d]-location)/spread;
            // Avoid subtracting two erf values close to the same +/-1 limit.
            const double difference = a >= 0 ? std::erfc(a)-std::erfc(b)
                : b <= 0 ? std::erfc(-b)-std::erfc(-a) : std::erf(b)-std::erf(a);
            product *= width*std::sqrt(std::acos(-1.0)/2.0)*difference/(hi[d]-lo[d]);
        }
        return background+rate*time+product;
    }
};
struct LinearProfile {
    double initial = 0, rate = 0;
    Kokkos::Array<double,3> gradient{}, origin{};
    template<class R> KOKKOS_INLINE_FUNCTION R point(R x,R y,R z,R time) const
    { return static_cast<R>(initial)+static_cast<R>(rate)*time+
        static_cast<R>(gradient[0])*(x-static_cast<R>(origin[0]))+
        static_cast<R>(gradient[1])*(y-static_cast<R>(origin[1]))+
        static_cast<R>(gradient[2])*(z-static_cast<R>(origin[2])); }
    template<class C> KOKKOS_INLINE_FUNCTION auto operator()(const C& c) const
    { return point(c.x,c.y,c.z,c.time); }
    double average(const std::array<double,3>& lo,const std::array<double,3>& hi,double time) const
    { return point((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2,time); }
};
using Profile = std::variant<ConstantProfile, LinearProfile, GaussianProfile>;

template<class R, class P>
KOKKOS_INLINE_FUNCTION Kokkos::Array<R,3> reference_gradient(const P& p,R x,R y,R z,R t)
{
    if constexpr (std::is_same_v<P,ConstantProfile>) return {R{0},R{0},R{0}};
    else if constexpr (std::is_same_v<P,LinearProfile>)
        return {static_cast<R>(p.gradient[0]),static_cast<R>(p.gradient[1]),static_cast<R>(p.gradient[2])};
    else {
        const R variance=static_cast<R>(p.width*p.width)+R{2}*static_cast<R>(p.diffusivity)*t;
        const R excess=p.point(x,y,z,t)-static_cast<R>(p.background)-static_cast<R>(p.rate)*t;
        return {-excess*(x-static_cast<R>(p.centre[0])-static_cast<R>(p.velocity[0])*t)/variance,
                -excess*(y-static_cast<R>(p.centre[1])-static_cast<R>(p.velocity[1])*t)/variance,
                -excess*(z-static_cast<R>(p.centre[2])-static_cast<R>(p.velocity[2])*t)/variance};
    }
}

inline Profile make_profile(const Configuration& c,bool reference=false)
{
    // Used as initial data at t=0. Analytic evolution is consumed only by explicit
    // profile references and/or verification, never implicitly installed as a BC.
    const double rate = c.mode == Mode::Transient ? c.source/c.density : 0;
    mpi::require(std::isfinite(rate), "Scalar transport: unrepresentable source/density");
    return std::visit([&](const auto& initial) -> Profile {
        using I = std::remove_cvref_t<decltype(initial)>;
        if constexpr (std::is_same_v<I, ConstantInitial>) return ConstantProfile{initial.value, rate};
        else if constexpr (std::is_same_v<I, LinearInitial>) {
            LinearProfile p; p.initial=initial.value; p.rate=rate;
            for(std::size_t d=0;d<3;++d) {
                p.gradient[d]=initial.gradient[d];p.origin[d]=initial.origin[d];
                if(c.convection && c.mode==Mode::Transient)p.rate-=c.velocity[d]*initial.gradient[d];
            }
            mpi::require(std::isfinite(p.rate),"Scalar transport: unrepresentable linear profile rate");
            return p;
        } else {
            GaussianProfile p;
            p.background = initial.background; p.amplitude = initial.amplitude; p.width = initial.width; p.rate = rate;
            p.diffusivity = c.diffusion_enabled ? c.diffusion/c.density : 0;
            const double time = c.mode == Mode::Transient ? c.end_time : 0;
            mpi::require(std::isfinite(p.diffusivity) &&
                std::isfinite(p.width*p.width+2*p.diffusivity*time), "Scalar transport: unrepresentable Gaussian spread");
            for (std::size_t d = 0; d < 3; ++d) {
                p.centre[d] = initial.centre[d]; p.velocity[d] = c.convection ? c.velocity[d] : 0;
                mpi::require(std::isfinite(p.centre[d]+p.velocity[d]*time), "Scalar transport: unrepresentable Gaussian centre");
            }
            return p;
        }
    }, reference && c.reference ? *c.reference : c.initial);
}

struct CellBox { std::array<double, 3> lo{}, hi{}; };

// Gaussian verification uses true cell averages. Restrict this helper explicitly
// to Cartesian hexahedra instead of silently integrating their bounding boxes.
inline CellBox cartesian_box(const mesh::host::MeshPart& m, std::size_t cell)
{
    mpi::require(m.cell_type[cell] == mesh::CellType::HEXA, "Scalar transport: Gaussian initialization requires Cartesian HEXA cells");
    const auto begin = m.cell_nodes_offsets[cell], end = m.cell_nodes_offsets[cell+1];
    mpi::require(end-begin == 8, "Scalar transport: Gaussian initialization expects eight vertices");
    CellBox b;
    b.lo.fill(std::numeric_limits<double>::infinity()); b.hi.fill(-std::numeric_limits<double>::infinity());
    for (auto k = begin; k < end; ++k) {
        const auto node = static_cast<std::size_t>(m.cell_nodes[k]);
        const std::array<double, 3> p{static_cast<double>(m.node_x[node]), static_cast<double>(m.node_y[node]), static_cast<double>(m.node_z[node])};
        for (std::size_t d = 0; d < 3; ++d) { b.lo[d] = std::min(b.lo[d], p[d]); b.hi[d] = std::max(b.hi[d], p[d]); }
    }
    const double tolerance = 256.0*static_cast<double>(std::numeric_limits<GeometryReal>::epsilon());
    unsigned mask = 0;
    double volume = 1;
    for (std::size_t d = 0; d < 3; ++d) {
        const double width = b.hi[d]-b.lo[d];
        mpi::require(std::isfinite(width) && width > 0, "Scalar transport: degenerate Gaussian cell box");
        volume *= width;
    }
    for (auto k = begin; k < end; ++k) {
        const auto node = static_cast<std::size_t>(m.cell_nodes[k]); unsigned corner = 0;
        const std::array<double, 3> p{static_cast<double>(m.node_x[node]), static_cast<double>(m.node_y[node]), static_cast<double>(m.node_z[node])};
        for (std::size_t d = 0; d < 3; ++d) {
            const double tol = tolerance*(b.hi[d]-b.lo[d]);
            if (std::abs(p[d]-b.hi[d]) <= tol) corner |= 1U << d;
            else mpi::require(std::abs(p[d]-b.lo[d]) <= tol, "Scalar transport: Gaussian cell is not axis-aligned");
        }
        mask |= 1U << corner;
    }
    mpi::require(mask == 255 && std::isfinite(volume) &&
        std::abs(volume-static_cast<double>(m.cell_volume[cell])) <= tolerance*volume,
        "Scalar transport: Gaussian reference requires a Cartesian cell, not its bounding box");
    return b;
}
} // namespace krona::apps::scalar_transport
