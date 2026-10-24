#pragma once
#include <krona/numerics/reconstruction/pack.hpp>
#include <krona/mpi/error.hpp>

namespace krona::transport::euler {

/// @brief Primitive order: p,u,v,w,T. All arguments are rank-one Kokkos views.
template<class P,class U,class V,class W,class T>
auto primitive_components(P p,U u,V v,W w,T t)
{ return numerics::reconstruction::pack(p,u,v,w,t); }

template<class P,class U,class T>
auto primitive_fields(P p,U velocity,T t)
{
    static_assert(P::components==1 && U::components==3 && T::components==1);
    return primitive_components(p.component(0),velocity.component(0),velocity.component(1),velocity.component(2),t.component(0));
}

/// @brief Conservative/residual order: rho,rho*u,rho*v,rho*w,rho*E.
template<class D,class M,class E>
auto conservative_fields(D density,M momentum,E energy)
{
    static_assert(D::components==1 && M::components==3 && E::components==1);
    return numerics::reconstruction::pack(density.component(0),momentum.component(0),momentum.component(1),momentum.component(2),energy.component(0));
}

namespace detail {
template<class Exec,bool Write,class Pack>
void validate_pack(const Pack& pack,std::size_t count)
{
    static_assert(Pack::components==5,"Euler requires exactly five components");
    static_assert(Kokkos::SpaceAccessibility<Exec,typename Pack::memory_space>::accessible);
    static_assert(std::is_floating_point_v<std::remove_const_t<typename Pack::value_type>>);
    if constexpr(Write)static_assert(!std::is_const_v<typename Pack::value_type>);
    mpi::require(pack.size()>=count,"Euler: undersized component pack");
    if constexpr(Write && requires{pack.views[0].pointer;})
        for(std::size_t a=0;a<5;++a)for(std::size_t b=a+1;b<5;++b)
            mpi::require(count==0 || pack.views[a].pointer!=pack.views[b].pointer,"Euler: duplicate residual components");
    if constexpr(requires{pack.views[0].pointer;pack.views[0].stride;})
        for(const auto& view:pack.views)
            mpi::require(count==0 || (view.pointer && view.stride>0),"Euler: invalid component storage");
}

template<class Input,class Output>
void reject_exact_alias(const Input& input,const Output& output)
{
    if constexpr(requires{input.views[0].pointer;output.views[0].pointer;})
        for(const auto& a:input.views)for(const auto& b:output.views)
            mpi::require(!a.pointer || !b.pointer || static_cast<const void*>(a.pointer)!=static_cast<const void*>(b.pointer),
                "Euler: input/output alias is unsupported");
}
} // namespace detail
} // namespace krona::transport::euler
