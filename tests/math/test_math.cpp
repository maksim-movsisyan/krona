#include <common/test_support.hpp>
#include <krona/math/math.hpp>

#include <array>
#include <limits>
#include <type_traits>
#include <utility>

namespace {
using namespace krona::test;
namespace math = krona::math; 

template<class R> constexpr R tolerance() { return R{256}*std::numeric_limits<R>::epsilon(); }
template<class R> void vector_near(math::Vector3<R> a, math::Vector3<R> b)
{ for (std::size_t i = 0; i < 3; ++i) near(a[i], b[i], tolerance<R>(), "Vector component"); }
template<class R> void matrix_near(const math::Matrix3<R>& a, const math::Matrix3<R>& b)
{ for (std::size_t i = 0; i < 3; ++i) for (std::size_t j = 0; j < 3; ++j) near(a(i,j), b(i,j), tolerance<R>(), "Matrix entry"); }

template<class R> void vector_case()
{
    using V = math::Vector3<R>;
    static_assert(std::is_trivially_copyable_v<V> && std::is_aggregate_v<V>);
    constexpr V a{1,2,3}, b{4,-1,2};
    static_assert(math::dot(a,b) == R{8});
    static_assert(math::cross(a,b).z == R{-9});
    static_assert(std::is_same_v<decltype(std::declval<const V&>()[0]), const R&>);
    vector_near(a+b, V{5,1,5}); vector_near(a-b, V{-3,3,1});
    vector_near(-a, V{-1,-2,-3}); vector_near(a*R{2}, V{2,4,6});
    vector_near(R{2}*a, V{2,4,6}); vector_near(a/R{2}, V{R{0.5},1,R{1.5}});
    vector_near(a.cross(b), V{7,10,-9});
    equal(a.norm2(), R{14}, "Squared vector norm");
    near(a.norm(), Kokkos::sqrt(R{14}), tolerance<R>(), "Vector norm");
    auto v = a; v += b; v -= b; v *= R{2}; v /= R{2}; vector_near(v,a);
    v[1] = R{7}; equal(v.y,R{7},"Indexed access updates the named member");
    check(math::isfinite(v),"Finite vector");
    v.z = std::numeric_limits<R>::infinity(); check(!math::isfinite(v),"Nonfinite vector");
}
void vector(const Exec&) { vector_case<float>(); vector_case<double>(); }

template<class R> void matrix_case()
{
    using M = math::Matrix3<R>;
    static_assert(std::is_trivially_copyable_v<M> && std::is_aggregate_v<M>);
    constexpr M a{{{1,2,3},{0,1,4},{5,6,0}}}, b{{{2,0,1},{1,3,0},{4,-1,2}}};
    constexpr M expected{{{16,3,7},{17,-1,8},{16,18,5}}};
    static_assert(math::determinant(a) == R{1});
    static_assert((a*b).data[0][0] == R{16});
    matrix_near(a*b, expected);
    matrix_near(a*M::identity(),a); matrix_near(M::identity()*a,a);
    matrix_near(math::transpose(math::transpose(a)),a);
    equal(math::transpose(a)(2,1),R{4},"Transpose swaps row and column");
    equal(math::trace(a),R{2},"Matrix trace");
    vector_near(a*math::Vector3<R>{1,2,3},math::Vector3<R>{14,14,17});
    matrix_near((a+b)-b,a); matrix_near(-(-a),a);
    matrix_near((R{2}*a)/R{2},a);
    auto update=a; update+=b; update-=b; update*=R{2}; update/=R{2}; matrix_near(update,a);
    const math::Vector3<R> u{1,2,3}, v{4,-1,2};
    vector_near(math::outer(u,v)*math::Vector3<R>{1,0,0},R{4}*u);
    matrix_near(math::transpose(math::outer(u,v)),math::outer(v,u));
    check(math::isfinite(a),"Finite matrix");
    update(1,2)=std::numeric_limits<R>::quiet_NaN(); check(!math::isfinite(update),"Nonfinite matrix");
}
void matrix(const Exec&) { matrix_case<float>(); matrix_case<double>(); }

template<class R> void tensor_case()
{
    using S = math::SymmetricTensor3<R>;
    static_assert(std::is_trivially_copyable_v<S> && std::is_aggregate_v<S>);
    constexpr S s{2,3,4,R{0.5},R{-0.25},R{0.75}};
    static_assert(math::trace(s) == R{9});
    const math::Vector3<R> v{1,2,3};
    vector_near(s*v,math::Vector3<R>{R{2.25},R{8.75},R{13.25}});
    vector_near(s*v,math::to_matrix(s)*v);
    equal(math::double_contract(s,s),R{30.75},"Off-diagonal products are counted twice");
    equal(math::quadratic_form(s,v),R{59.5},"Quadratic form");
    matrix_near(math::to_matrix(math::symmetric_outer(v)),math::outer(v,v));
    matrix_near(math::to_matrix((s+s)/R{2}),math::to_matrix(s));
    matrix_near(math::to_matrix(-(-s)),math::to_matrix(s));
    auto value=s; value+=s; value-=s; value*=R{2}; value/=R{2};
    matrix_near(math::to_matrix(value),math::to_matrix(s));
    value(2,0)=R{7}; equal(value.xz,R{7},"Symmetric lower access aliases upper storage");
    equal(&value(0,2),&value(2,0),"Symmetric positions share one member");
    check(math::isfinite(value),"Finite tensor");
    value.xy=std::numeric_limits<R>::infinity(); check(!math::isfinite(value),"Nonfinite tensor");
}
void tensor(const Exec&) { tensor_case<float>(); tensor_case<double>(); }

template<class R> void inverse_case()
{
    using M=math::Matrix3<R>;
    const M a{{{4,1,0},{1,3,0},{0,0,2}}};
    M inverse;
    check(math::inverse_spd(a,inverse),"SPD matrix is accepted");
    const M expected{{{R{3}/R{11},-R{1}/R{11},0},{-R{1}/R{11},R{4}/R{11},0},{0,0,R{0.5}}}};
    matrix_near(inverse,expected); matrix_near(a*inverse,M::identity());
    auto in_place=a; check(math::inverse_spd(in_place,in_place),"Matrix in-place inverse"); matrix_near(in_place,inverse);
    R raw[9]{4,1,0,1,3,0,0,0,2};
    check(math::inverse_spd(raw,raw),"Raw in-place inverse");
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) near(raw[3*i+j],expected.data[i][j],tolerance<R>(),"Row-major raw inverse");
    math::SymmetricTensor3<R> s{4,3,2,1,0,0}, si;
    check(math::inverse_spd(s,si),"Compact symmetric inverse"); matrix_near(math::to_matrix(si),inverse);
    check(math::inverse_spd(s,s),"Compact in-place inverse"); matrix_near(math::to_matrix(s),inverse);
    check(math::inverse_spd(M::identity(),inverse,0.0009765625),"Tolerance does not alter scalar-type deduction");

    for (int k=0;k<24;++k) {
        M b;
        for (int i=0;i<3;++i) for (int j=0;j<3;++j)
            b.data[i][j]=static_cast<R>(((k+1)*(i+2)+(j+1)*3)%9-4)/R{4};
        const M spd=math::transpose(b)*b+R{0.5}*M::identity();
        check(math::inverse_spd(spd,inverse),"Constructed SPD family");
        matrix_near(spd*inverse,M::identity());
    }
}
void inverse(const Exec&) { inverse_case<float>(); inverse_case<double>(); }

template<class R> void inverse_failure_case()
{
    using M=math::Matrix3<R>;
    const M sentinel{{{7,8,9},{10,11,12},{13,14,15}}};
    auto output=sentinel;
    auto rejects=[&](M a,R tol=R{64}*std::numeric_limits<R>::epsilon()) {
        check(!math::inverse_spd(a,output,tol),"Invalid/ill-conditioned input is rejected");
        matrix_near(output,sentinel);
    };
    rejects(M{});
    rejects(M{{{1,0,0},{0,0,0},{0,0,1}}});
    rejects(M{{{1,0,0},{0,-1,0},{0,0,1}}});
    rejects(M{{{1,1,0},{1,1,0},{0,0,1}}});
    auto nonsymmetric=M::identity(); nonsymmetric(0,1)=R{1}; rejects(nonsymmetric);
    auto bad=M::identity(); bad(0,2)=std::numeric_limits<R>::quiet_NaN(); rejects(bad);
    bad=M::identity(); bad(1,1)=std::numeric_limits<R>::infinity(); rejects(bad);
    for (const R tol : {R{0},R{-1},R{1},std::numeric_limits<R>::quiet_NaN(),std::numeric_limits<R>::infinity()}) rejects(M::identity(),tol);
    auto small=M::identity(); small(2,2)=static_cast<R>(1.e-6); rejects(small,static_cast<R>(1.e-4));
    check(math::inverse_spd(small,output,static_cast<R>(1.e-10)),"Explicit tolerance controls pivot rejection");
    output=sentinel;
    auto tiny=M::identity()*std::numeric_limits<R>::denorm_min(); rejects(tiny);
    R raw[9]{1,0,0,0,1,0,0,0,1};
    check(!math::inverse_spd(static_cast<const R*>(nullptr),raw),"Null raw input returns failure");
    check(!math::inverse_spd(raw,static_cast<R*>(nullptr)),"Null raw output returns failure");
    math::SymmetricTensor3<R> s{1,-1,1,0,0,0}, target{7,8,9,10,11,12};
    check(!math::inverse_spd(s,target),"Compact indefinite matrix is rejected");
    equal(target.xy,R{10},"Failed compact inverse preserves output");
    equal(target.zz,R{9},"Failed compact inverse preserves diagonal");
}
void inverse_failure(const Exec&) { inverse_failure_case<float>(); inverse_failure_case<double>(); }

void precision(const Exec&)
{
    using F=math::Vector3<float>; using D=math::Vector3<double>;
    constexpr F f{1,2,3}; constexpr D d{0.5,1.5,2.5};
    static_assert(std::is_same_v<decltype(f+d),D>);
    static_assert(std::is_same_v<decltype(f*2.0),D>);
    static_assert(std::is_same_v<decltype(f*2),F>);
    static_assert(std::is_same_v<decltype(math::dot(f,d)),double>);
    static_assert(std::is_same_v<decltype(math::outer(f,d)),math::Matrix3<double>>);
    const F a{1.e8f,1,-1.e8f}, b{1,1,1};
    equal(math::dot_as<double>(a,b),1.0,"Explicit dot precision casts before accumulation");
    const F large{std::numeric_limits<float>::max(),0,0};
    equal(math::dot_as<double>(large,F{2,0,0}),2.0*static_cast<double>(large.x),"Explicit dot precision casts before multiplication");
    vector_near(math::cast<double>(f),D{1,2,3});
    matrix_near(math::cast<double>(math::Matrix3<float>::identity()),math::Matrix3<double>::identity());
    const auto s=math::cast<double>(math::SymmetricTensor3<float>{1,2,3,4,5,6});
    equal(s.yz,6.0,"Tensor cast preserves storage order");
}

template<class R> void scale_case()
{
    const R large=std::is_same_v<R,float> ? static_cast<R>(1.e30) : static_cast<R>(1.e200);
    const R small=std::is_same_v<R,float> ? static_cast<R>(1.e-30) : static_cast<R>(1.e-200);
    near(math::norm(math::Vector3<R>{large,large,large})/large,Kokkos::sqrt(R{3}),tolerance<R>(),"Norm avoids avoidable squared overflow");
    near(math::norm(math::Vector3<R>{small,small,small})/small,Kokkos::sqrt(R{3}),tolerance<R>(),"Norm avoids avoidable squared underflow");
    const R tiny=std::numeric_limits<R>::denorm_min()*R{16};
    vector_near(math::Vector3<R>{tiny,tiny,tiny}/tiny,math::Vector3<R>{1,1,1});
}
void scale(const Exec&) { scale_case<float>(); scale_case<double>(); }

template<class R> void kernel_case(const Exec& exec)
{
    Kokkos::View<R*[7],typename Exec::memory_space> results("math/kernel",128);
    auto host=Kokkos::create_mirror(results);
    complete(exec,[&] {
        Kokkos::parallel_for("test/math_kernel",Policy(exec,0,128),KOKKOS_LAMBDA(Index index) {
            const auto i=static_cast<std::size_t>(index);
            const math::Vector3<R> a{1,2,3}, b{4,-1,2};
            const math::Matrix3<R> m{{{4,1,0},{1,3,0},{0,0,2}}};
            math::Matrix3<R> inverse;
            const bool success=math::inverse_spd(m,inverse);
            results(i,0)=success ? R{1} : R{0};
            results(i,1)=math::dot(a,b);
            results(i,2)=math::cross(a,b).z;
            results(i,3)=math::norm(math::Vector3<R>{0,3,4});
            results(i,4)=(m*a).y;
            results(i,5)=(m*inverse)(1,1);
            const math::SymmetricTensor3<R> s{2,3,4,R{0.5},R{-0.25},R{0.75}};
            results(i,6)=math::double_contract(s,s);
        });
        Kokkos::deep_copy(exec,host,results);
    });
    for (std::size_t i=0;i<128;++i) {
        constexpr std::array<R,7> expected{1,8,-9,5,7,1,R{30.75}};
        for (int c=0;c<7;++c) near(host(i,c),expected[static_cast<std::size_t>(c)],tolerance<R>(),"Device-callable math operations");
    }
}
void kernel(const Exec& exec) { kernel_case<float>(exec); kernel_case<double>(exec); }
} // namespace

int main(int argc,char**argv)
{
    const std::array cases{Case{"vector",vector},Case{"matrix",matrix},Case{"tensor",tensor},Case{"inverse",inverse},
        Case{"inverse_failure",inverse_failure},Case{"precision",precision},Case{"scale",scale},Case{"kernel",kernel}};
    return krona::test::run(argc,argv,"math",cases);
}
