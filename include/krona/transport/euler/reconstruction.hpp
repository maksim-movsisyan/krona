#pragma once
#include <krona/numerics/reconstruction/mesh_stencil.hpp>
#include <krona/transport/euler/bindings.hpp>
#include <utility>

namespace krona::transport::euler {

/// @brief Borrowed p,u,v,w,T batch. Stencil may be MeshStencil or an explicit custom face stencil (e.g. QUICK).
template<class Scheme,class Data,class Stencil,class BoundaryScheme=Scheme>
struct Reconstruction {
    using scheme_type=Scheme;
    using boundary_scheme_type=BoundaryScheme;
    Scheme scheme;
    Data data;
    Stencil stencil;
    BoundaryScheme boundary_scheme;
    Reconstruction(Scheme s,Data input,Stencil geometry) requires std::is_same_v<Scheme,BoundaryScheme>
        :scheme(std::move(s)),data(std::move(input)),stencil(std::move(geometry)),boundary_scheme(scheme) {}
    Reconstruction(Scheme s,Data input,Stencil geometry,BoundaryScheme boundary)
        :scheme(std::move(s)),data(std::move(input)),stencil(std::move(geometry)),boundary_scheme(std::move(boundary)) {}

    template<class R>
    KOKKOS_INLINE_FUNCTION auto interior(std::size_t f) const
    {
        if constexpr(requires{stencil.template face<Scheme>(f);})
            return numerics::reconstruction::evaluate<R>(scheme,stencil.template face<Scheme>(f),data);
        else return numerics::reconstruction::evaluate<R>(scheme,stencil.face(f),data);
    }
    template<class R>
    KOKKOS_INLINE_FUNCTION auto boundary(std::size_t f) const
    { return numerics::reconstruction::evaluate_side<R>(boundary_scheme,stencil.boundary_side(f),data); }

    template<class Exec>
    void validate(std::size_t cells,std::size_t extended,bool with_boundary) const
    {
        detail::validate_pack<Exec,false>(data.values,with_boundary?extended:cells);
        constexpr unsigned needs=Scheme::requirements|BoundaryScheme::requirements;
        namespace rec=numerics::reconstruction;
        auto check=[&]<class Provider>(const Provider& provider){
            if constexpr(requires{typename Provider::memory_space;})
                static_assert(Kokkos::SpaceAccessibility<Exec,typename Provider::memory_space>::accessible);
            if constexpr(requires{provider.size();})mpi::require(provider.size()>=cells,"Euler: undersized reconstruction provider");
        };
        if constexpr((needs&rec::Gradients)!=0)check(data.gradients);
        if constexpr((needs&rec::Factors)!=0)check(data.factors);
        if constexpr((needs&rec::Hessians)!=0)check(data.hessians);
        if constexpr((needs&rec::Moments)!=0)check(data.moments);
        if constexpr(requires{typename Stencil::memory_space;})
            static_assert(Kokkos::SpaceAccessibility<Exec,typename Stencil::memory_space>::accessible);
        if constexpr(requires{stencil.required_input_size();})
            if(with_boundary)mpi::require(data.values.size()>=stencil.required_input_size(),"Euler: stencil samples exceed primitive storage");
    }
};
template<class S,class D,class T>Reconstruction(S,D,T)->Reconstruction<S,D,T>;
template<class S,class D,class T,class B>Reconstruction(S,D,T,B)->Reconstruction<S,D,T,B>;

} // namespace krona::transport::euler
