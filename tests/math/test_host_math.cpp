#include <krona/math/math.hpp>

// Deliberately no MPI/Kokkos initialization, KRONA config or execution-space header.
int main()
{
    namespace math=krona::math;
    constexpr math::Vector3<double> v{0,3,4};
    static_assert(v.norm2()==25.0);
    if (v.norm()!=5.0) return 1;
    math::Matrix3<double> inverse;
    if (!math::inverse_spd(math::Matrix3<double>::identity(),inverse)) return 2;
    const auto result=inverse*v;
    if (result.y!=3.0 || result.z!=4.0) return 3;
    if (math::double_contract(math::symmetric_outer(v),math::symmetric_outer(v))!=625.0) return 4;
    return 0;
} 
