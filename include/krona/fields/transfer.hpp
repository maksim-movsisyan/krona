#pragma once

#include <krona/fields/pack.hpp>
#include <krona/execution/utils.hpp>

namespace krona::fields {
namespace detail {

template<class Pack, class Dofs, bool ToDofs>
struct DofTransfer {
    Pack fields;
    Dofs dofs;

    template<std::size_t C>
    KOKKOS_INLINE_FUNCTION void copy(std::size_t cell) const
    {
        const auto row = cell*Pack::components + C;
        if constexpr (ToDofs) {
            dofs(row) = static_cast<typename Dofs::non_const_value_type>(fields.template get<C>(cell));
        } else {
            using Value = std::remove_reference_t<decltype(fields.template get<C>(cell))>;
            static_assert(!std::is_const_v<Value>, "Fields: cannot unpack into a const component");
            fields.template get<C>(cell) = static_cast<Value>(dofs(row));
        }
    }

    template<std::size_t... C>
    KOKKOS_INLINE_FUNCTION void copy_cell(std::size_t cell, std::index_sequence<C...>) const
    { (copy<C>(cell), ...); }

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const
    { copy_cell(cell, std::make_index_sequence<Pack::components>{}); }
};

template<bool ToDofs, class Exec, class Input, class Dofs>
void transfer_dofs(const Exec& exec, Input input, Dofs dofs)
{
    execution::require_kokkos();
    static_assert(Kokkos::is_view<Dofs>::value && Dofs::rank == 1,
                  "Fields: DOF storage must be a rank-one Kokkos view");
    static_assert(std::is_arithmetic_v<typename Dofs::non_const_value_type>);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename Dofs::memory_space>::accessible);
    if constexpr (ToDofs)
        static_assert(!std::is_const_v<typename Dofs::value_type>, "Fields: DOF destination must be writable");

    const auto packed = fields::pack(input);
    using Pack = decltype(packed);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename Pack::memory_space>::accessible);
    (void)descriptor_from_view(dofs);
    mpi::require(dofs.extent(0)%Pack::components == 0, "Fields: field pack/DOF size mismatch");
    const auto cells = dofs.extent(0)/Pack::components;
    mpi::require(packed.size() >= cells, "Fields: field is shorter than the requested cell prefix");

    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
    mpi::guarded([&] {
        Kokkos::parallel_for(ToDofs ? "fields/to_dofs" : "fields/from_dofs", Policy(exec, 0, cells),
            DofTransfer<Pack, Dofs, ToDofs>{packed, dofs});
    });
}

} // namespace detail

/// @brief Copies a field/pack to rank-one DOFs in cell-major component order.
/// No allocation or final fence. DOF extent selects the cell prefix; field tails are ignored.
/// Buffers must be accessible, non-overlapping and live until completion on exec.
template<class Exec, class Input, class Dofs>
void copy_to_dofs_async(const Exec& exec, Input input, Dofs dofs)
{ detail::transfer_dofs<true>(exec, input, dofs); }

/// @brief Copies rank-one DOFs to a field/pack, converting to each component's exact scalar type.
/// No allocation or final fence. Destinations must be writable and must not overlap each other
/// or DOF storage. Only the DOF-selected cell prefix is written; extended tails are untouched.
template<class Exec, class Dofs, class Output>
void copy_from_dofs_async(const Exec& exec, Dofs dofs, Output output)
{ detail::transfer_dofs<false>(exec, output, dofs); }

/// @brief Blocking counterpart; fences only exec. External owner access states are caller-managed.
template<class Exec, class Input, class Dofs>
void copy_to_dofs(const Exec& exec, Input input, Dofs dofs)
{
    copy_to_dofs_async(exec, input, dofs);
    mpi::guarded([&] { exec.fence("Fields: copy to DOFs complete"); });
}

/// @brief Blocking counterpart; fences only exec. External owner access states are caller-managed.
template<class Exec, class Dofs, class Output>
void copy_from_dofs(const Exec& exec, Dofs dofs, Output output)
{
    copy_from_dofs_async(exec, dofs, output);
    mpi::guarded([&] { exec.fence("Fields: copy from DOFs complete"); });
}

} // namespace krona::fields
