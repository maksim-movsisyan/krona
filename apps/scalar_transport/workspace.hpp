#pragma once
#include "runtime.hpp"
#include <krona/fields/field_storage.hpp>
#include <krona/halo/exchange.hpp>
#include <krona/numerics/gradient/gradient_operator.hpp>
#include <krona/numerics/reconstruction/gradient_pack.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace krona::apps::scalar_transport {
// Named functors avoid NVCC extended-lambda restrictions in constructors and generic visitors.
namespace workspace_detail {
struct BoundaryFeedback {
    boundary::GeometryView<Space> geometry;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t f, int& sum) const
    {
        const auto c=geometry.template context<Real>(static_cast<LocalIndex>(f));
        const Real tangent=Kokkos::sqrt(c.tx*c.tx+c.ty*c.ty+c.tz*c.tz);
        if (tangent > Real{64} * static_cast<Real>(Kokkos::epsilon_v<GeometryReal>) * c.distance)
            ++sum;
    }
};

template<class Volumes>
struct DomainVolume {
    Volumes volumes;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, double& sum) const
    { sum+=static_cast<double>(volumes(i)); }
};

template<class Current, class Previous>
struct SaveGradient {
    Current current; Previous previous;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    { for(int d=0;d<3;++d)previous(i,d)=current(i,d); }
};

template<class Current, class Previous>
struct GradientDifference {
    Current current; Previous previous;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, double& sum) const
    {
        for(int d=0;d<3;++d) {
            const double value=static_cast<double>(current(i,d))-static_cast<double>(previous(i,d));
            sum+=value*value;
        }
    }
};

template<class Gradients>
struct GradientScale {
    Gradients gradients;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, double& sum) const
    {
        for(int d=0;d<3;++d) {
            const double value=static_cast<double>(gradients(i,d));
            sum+=value*value;
        }
    }
};

template<class Field>
struct ValueScale {
    Field field;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, double& sum) const
    {
        const double value=static_cast<double>(field(i,0));
        sum+=value*value;
    }
};

template<class Condition, class ProfileType>
struct VerifyBoundary {
    boundary::GeometryView<Space> geometry;
    Condition condition;
    ProfileType profile;
    Real time, coefficient, tolerance;
    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex f, std::size_t& count) const
    {
        const auto c=geometry.template context<Real>(f,time);
        Real actual=0,expected=0;
        if constexpr(bc::is_fixed_value<Condition>) {
            actual=boundary::profile::sample(condition.value,c);
            expected=profile(c);
        } else {
            const auto gradient=reference_gradient(profile,c.x,c.y,c.z,time);
            const Real normal=gradient[0]*c.nx+gradient[1]*c.ny+gradient[2]*c.nz;
            if constexpr(bc::is_normal_gradient<Condition>) {
                actual=boundary::profile::sample(condition.gradient,c);
                expected=normal;
            } else {
                actual=boundary::profile::sample(condition.flux,c);
                expected=-coefficient*normal;
            }
        }
        if(!Kokkos::isfinite(actual)||!Kokkos::isfinite(expected)||
           Kokkos::abs(actual-expected)>tolerance*Kokkos::max(Real{1},Kokkos::max(Kokkos::abs(actual),Kokkos::abs(expected))))++count;
    }
};
} // namespace workspace_detail

// Match the convection adapter's one-sided linear outflow increment during factor preparation.
template<class Scheme>
struct FactorIncrement {
    static constexpr unsigned requirements=Scheme::requirements;
    Scheme scheme; std::size_t real_cells;
    template<class R,class Side,class D>
    KOKKOS_INLINE_FUNCTION R increment(const Side& side,const D& data,std::size_t component) const
    {
        if constexpr((requirements&rec::OtherValues)!=0)
            if(side.other>=real_cells)return rec::SecondOrderUpwind{}.template increment<R>(side,data,component);
        return scheme.template increment<R>(side,data,component);
    }
};
// Persistent high-order storage. All borrowed descriptors die before these owners.
class NumericsWorkspace {
    using Policy=Kokkos::RangePolicy<Exec,Kokkos::IndexType<std::size_t>>;
    Exec exec_;
    mpi::CommView comm_;
    Configuration config_;
    fields::FieldStorage<Storage,3,Exec> gradients_;
    Kokkos::View<Storage*[3],Kokkos::LayoutLeft,Space> old_gradients_;
    fields::FieldStorage<Storage,1,Exec> factors_;
    std::optional<gr::GradientOperator<Exec>> gradient_;
    std::optional<halo::Exchange<Storage,Exec>> exchange_;
    rec::MeshStencil<Space,Real> stencil_;
    bc::Status<Exec> status_;
    bool feedback_=false;
    double length_=1;
    std::size_t real_cells_=0;
public:
    NumericsWorkspace(const Configuration& c,mesh::device::MeshView<Space> mesh,
        mesh::device::AuxConnectivityView<Space> connectivity,mesh::device::AuxGeometryView<Space> geometry,
        mesh::host::HaloPlanView halo,const Boundary& boundary,mpi::CommView comm,const Exec& exec)
        :exec_(exec),comm_(comm),config_(c),
         gradients_("scalar/gradient",c.needs_gradients()?static_cast<std::size_t>(mesh.sizes.n_cells):0,fields::FieldLocation::Cell,exec),
         factors_("scalar/limiter",c.reconstruction.cell_limiter!=ts::CellLimiter::None?static_cast<std::size_t>(mesh.sizes.n_cells):0,fields::FieldLocation::Cell,exec),
         stencil_(mesh,connectivity),status_(exec)
    {
        real_cells_=static_cast<std::size_t>(mesh.sizes.n_cells);
        if(!c.needs_gradients())return;
        gradients_.fill_async(exec,Storage{0});
        gradient_.emplace(mesh,connectivity,geometry,c.gradient,exec);
        exchange_.emplace(mesh,halo,comm,72,exec,halo::Transfer::HostStaged);
        exchange_->add_field(gradients_.view());
        if(factors_.size()!=0)exchange_->add_field(factors_.view());
        exchange_->prepare();
        int local=0;
        const auto geom=boundary.geometry_view();
        boundary.for_each_patch([&](auto range,const auto& condition) {
            using C=std::remove_cvref_t<decltype(condition)>;
            if constexpr(!bc::is_fixed_value<C>) {
                int count=0;
                Kokkos::parallel_reduce("scalar/boundary_feedback",Policy(exec_,static_cast<std::size_t>(range.begin),static_cast<std::size_t>(range.end)),
                    workspace_detail::BoundaryFeedback{geom},count);
                if(count!=0)local=1;
            }
        });
        int global=0;mpi::check(MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MAX,comm.native()),"MPI_Allreduce(boundary feedback)");
        feedback_=global!=0;
        if(feedback_) {
            old_gradients_=decltype(old_gradients_)(Kokkos::view_alloc(exec_,Kokkos::WithoutInitializing,"scalar/old_gradient"),static_cast<std::size_t>(mesh.sizes.n_own));
            double volume=0,total=0;const auto volumes=mesh.geometry.cell_volume;
            Kokkos::parallel_reduce("scalar/domain_volume",Policy(exec_,0,static_cast<std::size_t>(mesh.sizes.n_own)),
                workspace_detail::DomainVolume{volumes},volume);
            mpi::check(MPI_Allreduce(&volume,&total,1,MPI_DOUBLE,MPI_SUM,comm.native()),"MPI_Allreduce(domain volume)");
            length_=std::cbrt(total);mpi::require(std::isfinite(length_)&&length_>0,"Scalar transport: invalid domain volume");
        }
    }
    auto gradient_pack() const {return rec::gradient_pack(gradients_.view());}
    auto factor_pack() {return fields::pack(factors_.view());}
    auto correction() const {return gr::GradientCorrection{gr::cell_gradients(gradients_.view())};}
    template<class Field>
    double prepare(Field scalar,const Boundary& boundary,const Law& law,const SchemeChoice& schemes,Real time)
    {
        status_.begin_async();
        if(gradient_)boundary.apply_async(scalar,status_,law,correction(),time);
        else boundary.apply_async(scalar,status_,law,gr::NoCorrection{},time);
        status_.check();
        if(!gradient_)return 0;
        const auto owned=stencil_.size();
        const auto g=gradients_.view();const auto old=old_gradients_;
        if(feedback_)Kokkos::parallel_for("scalar/save_gradient",Policy(exec_,0,owned),
            workspace_detail::SaveGradient{g,old});
        gradient_->apply_async(scalar,g);
        if(config_.reconstruction.cell_limiter!=ts::CellLimiter::None) {
            const auto input=rec::data(fields::pack(scalar),gradient_pack());
            std::visit([&](auto scheme) {
                // Use the raw increment being limited, before any old factor is applied.
                const FactorIncrement raw{scheme,real_cells_};
                if(config_.reconstruction.cell_limiter==ts::CellLimiter::BarthJespersen)
                    rec::compute_limiters_async<Real>(exec_,stencil_,input,factor_pack(),rec::BarthJespersen{},raw);
                else rec::compute_limiters_async<Real>(exec_,stencil_,input,factor_pack(),
                    rec::Venkatakrishnan{rec::ConstantRegularization<Real>{computation_value(config_.reconstruction.regularization)}},raw);
            },schemes);
        }
        exchange_->exchange();
        if(!feedback_)return 0;
        double difference=0,scale=0,values=0;
        Kokkos::parallel_reduce("scalar/gradient_difference",Policy(exec_,0,owned),
            workspace_detail::GradientDifference{g,old},difference);
        Kokkos::parallel_reduce("scalar/gradient_scale",Policy(exec_,0,owned),
            workspace_detail::GradientScale{g},scale);
        Kokkos::parallel_reduce("scalar/value_scale",Policy(exec_,0,owned),
            workspace_detail::ValueScale{scalar},values);
        double local[2]{difference,scale+values/(length_*length_)},global[2]{};
        mpi::check(MPI_Allreduce(local,global,2,MPI_DOUBLE,MPI_SUM,comm_.native()),"MPI_Allreduce(gradient consistency)");
        mpi::require(std::isfinite(global[0])&&std::isfinite(global[1]),"Scalar transport: nonfinite gradient feedback");
        return global[0]==0?0:std::sqrt(global[0]/std::max(global[1],std::numeric_limits<double>::min()));
    }
};

// Verification checks the supplied conditions; it never replaces numeric BC data.
inline void verify_boundary(const Configuration& config,const Boundary& boundary,const Profile& profile,Real time)
{
    if(!config.verification)return;
    std::size_t bad=0;const auto geometry=boundary.geometry_view();const auto exec=boundary.execution_space();
    const Real coefficient=computation_value(config.diffusion);
    const Real tolerance=Real{256}*static_cast<Real>(std::max(std::numeric_limits<Storage>::epsilon(),static_cast<Storage>(std::numeric_limits<Real>::epsilon())));
    std::visit([&](auto p){boundary.for_each_patch([&](auto range,auto condition) {
        using C=decltype(condition);std::size_t local=0;
        Kokkos::parallel_reduce("scalar/verify_boundary",Kokkos::RangePolicy<Exec,Kokkos::IndexType<LocalIndex>>(exec,range.begin,range.end),
            workspace_detail::VerifyBoundary<C,decltype(p)>{geometry,condition,p,time,coefficient,tolerance},local);
        bad+=local;
    });},profile);
    mpi::require(bad==0,"Scalar transport: boundary data do not match the requested analytic reference");
}
} // namespace krona::apps::scalar_transport