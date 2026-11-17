#pragma once

#include <krona/fields/pack.hpp>
#include <krona/execution/utils.hpp>

namespace krona::fields {
namespace detail {

/// @brief Copies all components of one cell between a pack and a flat DOF vector.
/// @tparam Pack Pack of borrowed field components.
/// @tparam Dofs Rank-one Kokkos view of the degrees of freedom.
/// @tparam ToDofs True to write the DOF vector, false to write the field components.
template<class Pack, class Dofs, bool ToDofs>
struct DofTransfer {
    Pack fields;
    Dofs dofs;

    template<std::size_t C>
    KOKKOS_INLINE_FUNCTION void copy(std::size_t cell) const {
        // The DOF vector stores all components of each cell together.
        const auto row = cell*Pack::components + C;
        // Use the component's exact type for conversion, including in mixed packs.
        if constexpr (ToDofs) {
            dofs(row) = static_cast<typename Dofs::non_const_value_type>(fields.template get<C>(cell));
        } else {
            using Value = std::remove_reference_t<decltype(fields.template get<C>(cell))>;
            static_assert(!std::is_const_v<Value>, "Fields: cannot unpack into a const component");
            fields.template get<C>(cell) = static_cast<Value>(dofs(row));
        }
    }

    // Expand the component copies at compile time so each one knows its scalar type.
    template<std::size_t... C>
    KOKKOS_INLINE_FUNCTION void copy_cell(std::size_t cell, std::index_sequence<C...>) const
    { (copy<C>(cell), ...); }

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t cell) const
    { copy_cell(cell, std::make_index_sequence<Pack::components>{}); }
};

template<bool ToDofs, class Exec, class Input, class Dofs>
void transfer_dofs(const Exec& exec, Input input, Dofs dofs) {
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

    // The vector length decides how many leading cells to copy. The field may have a tail.
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

/// @brief Queues a copy from field components to a flat vector, with all DOFs of cell 0 first.
/// @tparam Exec Kokkos execution space that can access both buffers.
/// @tparam Input Field view, rank-one Kokkos view or component pack.
/// @tparam Dofs Type of the writable rank-one Kokkos view that receives the values.
/// @param exec Instance used to queue the copy kernel.
/// @param input Source of the values, in field or pack component order.
/// @param dofs Destination. Its length must be a multiple of the component count
/// and sets how many leading cells are copied.
/// For two components and three cells, the output order is a(0), b(0), a(1), b(1), a(2), b(2).
/// Use disjoint source and destination storage and keep both alive until the copy finishes.
template<class Exec, class Input, class Dofs>
void copy_to_dofs_async(const Exec& exec, Input input, Dofs dofs)
{ detail::transfer_dofs<true>(exec, input, dofs); }

/// @brief Queues a copy from flat DOFs into the leading cells of a field or pack.
/// @tparam Exec Kokkos execution space that can access both buffers.
/// @tparam Dofs Type of the rank-one Kokkos view containing the DOFs.
/// @tparam Output Writable field view, rank-one Kokkos view or component pack.
/// @param exec Instance used to queue the copy kernel.
/// @param dofs Source in cell-major order, with a whole number of cells' worth of values.
/// @param output Destination components. Each receives values converted to its own scalar type.
/// Use disjoint destination components and DOF storage. Keep them alive until completion.
template<class Exec, class Dofs, class Output>
void copy_from_dofs_async(const Exec& exec, Dofs dofs, Output output)
{ detail::transfer_dofs<false>(exec, output, dofs); }

/// @brief Queues copy_to_dofs_async and waits on exec before returning.
/// Follow the DOF owner's own access rules when using a view from another module.
template<class Exec, class Input, class Dofs>
void copy_to_dofs(const Exec& exec, Input input, Dofs dofs) {
    copy_to_dofs_async(exec, input, dofs);
    mpi::guarded([&] { exec.fence("Fields: copy to DOFs complete"); });
}

/// @brief Queues copy_from_dofs_async and waits on exec before returning.
/// Follow the DOF owner's own access rules when using a view from another module.
template<class Exec, class Dofs, class Output>
void copy_from_dofs(const Exec& exec, Dofs dofs, Output output) {
    copy_from_dofs_async(exec, dofs, output);
    mpi::guarded([&] { exec.fence("Fields: copy from DOFs complete"); });
}

} // namespace krona::fields
