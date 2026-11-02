#pragma once 

#include <krona/execution/types.hpp>
#include <krona/mpi/error.hpp>

namespace krona::fields {

template<class T, class Space>
struct ComponentDescriptor {
    using value_type = T;
    using memory_space = Space;

    T* data = nullptr;
    std::size_t stride = 1;
};

template<class View>
auto descriptor_from_view(View view)
{
    static_assert(Kokkos::is_view<View>::value && View::rank == 1,
                  "Fields: a packed Kokkos view must have rank one");
    mpi::require(view.extent(0) == 0 || (view.data() != nullptr && view.stride(0) > 0),
                 "Fields: invalid packed component storage");
    return ComponentDescriptor<typename View::value_type, typename View::memory_space>{view.data(), view.stride(0)};
}

}