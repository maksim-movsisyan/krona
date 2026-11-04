#pragma once
#include <krona/execution/types.hpp>
#include <initializer_list>

namespace krona::test::scalar_transient {
using Exec = execution::ExecutionSpace;
using Space = typename Exec::memory_space;
template<class T> using View = Kokkos::View<T*, Space>;

template<class T>
View<T> input(const Exec& exec, std::initializer_list<T> values)
{ 
    View<T> result(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "test/input"), values.size());
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, result);
    std::size_t i = 0;
    for (const auto value : values) host(i++) = value;
    Kokkos::deep_copy(exec, result, host);
    exec.fence("Test input ready");
    return result;
}
template<class V> auto host_copy(V v)
{ return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v); }
struct Entries { View<std::size_t> diagonal; };

} // namespace krona::test::scalar_transient
