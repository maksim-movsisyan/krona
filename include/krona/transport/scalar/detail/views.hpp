#pragma once

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>

#include <type_traits>

namespace krona::transport::scalar::detail {
template<class Exec, class View>
void validate_view(View view, std::size_t count)
{
    static_assert(Kokkos::is_view<View>::value && View::rank == 1);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);
    mpi::require(view.extent(0) >= count && (count == 0 || (view.data() != nullptr && view.stride(0) > 0)),
                 "Scalar transport: invalid component extent or storage");
}
template<class Exec, class View>
void validate_output(View view, std::size_t count)
{
    static_assert(!std::is_const_v<typename View::value_type>);
    static_assert(std::is_floating_point_v<typename View::non_const_value_type>);
    validate_view<Exec>(view, count);
}
} // namespace krona::transport::scalar::detail
