#pragma once
#include "dispatch.hpp"
#include "profiles.hpp"
#include <krona/boundary/scalar/manager.hpp>
#include <krona/transport/scalar/convection/operator.hpp>
#include <krona/transport/scalar/diffusion/operator.hpp>
#include <krona/transport/scalar/source.hpp>
#include <optional>

namespace krona::apps::scalar_transport {
using Exec = execution::ExecutionSpace;
using Real = ComputeReal;
using Space = typename Exec::memory_space;
using Storage = StorageReal;
namespace cv = ts::convection;
namespace df = ts::diffusion;
namespace bc = boundary::scalar;
namespace gr = numerics::gradient;
using ConstantBoundary = bc::FixedValue<Real, ConstantProfile>;
using LinearBoundary = bc::FixedValue<Real, LinearProfile>;
using GaussianBoundary = bc::FixedValue<Real, GaussianProfile>;
using Boundary = bc::Manager<Exec, Real, ConstantBoundary, LinearBoundary, GaussianBoundary>;
using Law = transport::IsotropicDiffusion<transport::UniformCoefficient<Real>>;
struct UniformFlow {
    Real rho = 1, ux = 0, uy = 0, uz = 0;
    template<class C> KOKKOS_INLINE_FUNCTION Real operator()(const C& c) const
    { return rho*(ux*c.nx+uy*c.ny+uz*c.nz)*c.area; }
};
inline Real computation_value(double value)
{
    const Real converted = static_cast<Real>(value);
    mpi::require(std::isfinite(converted) && (value == 0 || converted != Real{0}),
        "Scalar transport: parameter is not representable in computation precision");
    return converted;
}
inline void configure_boundary(Boundary& boundary,const bc::Configuration& config,const Profile& profile)
{
    boundary.configure(config,[&](const bc::ConditionConfig& spec,const auto& assign) {
        mpi::require(spec.profile=="reference" && spec.kind==bc::Kind::FixedValue,
            "Scalar transport: built-in profile reference is FIXED_VALUE with profile='reference'");
        std::visit([&](const auto& p){assign(bc::fixed_value<Real>(p));},profile);
    });
    boundary.finalize();
}

// Stage-local dispatch: no gradient/time/solver variant surrounds the reconstruction visitor.
class SpatialEquation {
    std::optional<cv::ConvectionOperator<Exec, Real>> convection_;
    std::optional<df::DiffusionOperator<Exec, Real>> diffusion_;
    ts::SourceOperator<Exec, Real> source_;
    UniformFlow flow_;
    Law law_;
    ts::UniformSource<Real> source_value_;
    Configuration config_;
    SchemeChoice scheme_;
public:
    SpatialEquation(const Configuration& c, mesh::device::MeshView<Space> mesh,
        mesh::device::AuxConnectivityView<Space> connectivity, mesh::device::AuxGeometryView<Space> geometry,
        mpi::CommView comm, const Exec& exec)
        : source_(mesh,exec),flow_{computation_value(c.density),computation_value(c.velocity[0]),
            computation_value(c.velocity[1]),computation_value(c.velocity[2])},
          law_{transport::UniformCoefficient<Real>{computation_value(c.diffusion)}},
          source_value_{computation_value(c.source)},config_(c),scheme_(choose_scheme(c.reconstruction))
    {
        if(c.convection)convection_.emplace(cv::ConvectionPlan<Exec,Real>{mesh,connectivity,comm,exec});
        if(c.diffusion_enabled)diffusion_.emplace(df::DiffusionPlan<Exec,Real>{mesh,connectivity,geometry,comm,exec,
            {.require_orthogonal=c.require_orthogonal,.orthogonality_tolerance=c.orthogonality_tolerance}});
    }
    [[nodiscard]] const Law& law() const noexcept {return law_;}
    [[nodiscard]] const SchemeChoice& scheme() const noexcept {return scheme_;}
    template<class Target>
    void add_base_async(Target target,const Boundary& boundary,ts::AssemblyStatus<Exec>& status,Real time) const
    {
        if(convection_)convection_->add_system_async(target,boundary,status,flow_,cv::NoReconstruction{},cv::FixedValueInflow{},time);
        if(diffusion_)diffusion_->add_system_async(target,boundary,status,law_,time);
        source_.add_async(target.rhs,status,source_value_);
    }
    template<class Values,class Gradients,class Factors,class F>
    void visit_reconstruction(Values values,Gradients gradients,Factors factors,F&& f) const
    {
        std::visit([&](auto scheme) {
            using S=decltype(scheme);
            if constexpr(std::is_same_v<S,rec::FirstOrder>) f(cv::NoReconstruction{});
            else {
                if(config_.reconstruction.cell_limiter==ts::CellLimiter::None)
                    f(cv::make_reconstruction(convection_->plan(),scheme,rec::data(values,gradients)));
                else f(cv::make_reconstruction(convection_->plan(),rec::Limited<S>{scheme},rec::data(values,gradients,factors)));
            }
        },scheme_);
    }
    template<class Output,class Values,class Gradients,class Factors,class Correction>
    void add_correction_async(Output output,const Boundary& boundary,ts::AssemblyStatus<Exec>& status,
        Values values,Gradients gradients,Factors factors,Correction correction,Real time) const
    {
        if(convection_)visit_reconstruction(values,gradients,factors,[&](auto r) {
            convection_->add_correction_async(output,boundary,status,flow_,r,time);
        });
        if(diffusion_ && config_.nonorthogonal_correction)
            diffusion_->add_correction_async(output,boundary,status,law_,correction,time);
    }
    template<class Field,class Output,class Values,class Gradients,class Factors,class Correction>
    void add_residual_async(Field field,Output output,const Boundary& boundary,ts::AssemblyStatus<Exec>& status,
        Values values,Gradients gradients,Factors factors,Correction correction,Real time) const
    {
        if(convection_)visit_reconstruction(values,gradients,factors,[&](auto r) {
            convection_->add_residual_async(field,output,boundary,status,flow_,r,cv::FixedValueInflow{},time);
        });
        if(diffusion_) {
            if(config_.nonorthogonal_correction)diffusion_->add_residual_async(field,output,boundary,status,law_,correction,time);
            else diffusion_->add_residual_async(field,output,boundary,status,law_,time);
        }
        source_.add_async(output,status,source_value_);
    }
};
} // namespace krona::apps::scalar_transport
