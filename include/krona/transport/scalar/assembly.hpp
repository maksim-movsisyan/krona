#pragma once

#include <krona/linalg/csr_matrix.hpp>
#include <krona/linalg/face_assembly.hpp>
#include <krona/transport/scalar/system_view.hpp>

namespace krona::transport::scalar {

/// @brief Opens native read-write views for additive scalar assembly.
/// Native access may synchronize host changes. The caller initializes new targets and commits both wrappers.
/// @tparam Matrix KRONA CSR or BSR matrix storing the coefficients.
/// @tparam Exec Kokkos execution space used for the native views and assembly submissions.
/// @param matrix Matrix that shares its graph with assembly.
/// @param rhs Right-hand side on the matrix's point distribution.
/// @param assembly Entry map with the same local cell and face numbering as the spatial operator.
/// @param exec Execution instance on which the views will be used.
/// @param component Equation and variable selected within each BSR block. Use zero for CSR.
template<class Matrix, class Exec = typename Matrix::execution_space>
[[nodiscard]] auto make_system_view(Matrix& matrix, linalg::Vector<typename Matrix::scalar_type, Exec>& rhs,
    const linalg::FaceAssemblyPlan<Exec>& assembly, const Exec& exec = {}, linalg::LocalOrdinal component = 0) {
    mpi::require(matrix.pattern().native().getRawPtr() == assembly.pattern().native().getRawPtr(),
                 "Scalar transport: matrix must share the face assembly graph");
    mpi::require(rhs.distribution().native().getRawPtr() == matrix.distribution().native().getRawPtr(),
                 "Scalar transport: RHS must share the matrix distribution");
    const auto width = matrix.block_size();
    mpi::require(component >= 0 && component < width, "Scalar transport: invalid block component");

    // For BSR, take the same diagonal component from each block, including off-diagonal blocks.
    const auto values = [&] {
        if constexpr (Matrix::block_storage)
            return Kokkos::subview(matrix.edit_blocks(exec), Kokkos::ALL(), component, component);
        else return matrix.edit_values(exec);
    }();
    const auto native_rhs = rhs.edit(exec);
    const auto count = assembly.pattern().distribution().local_size();
    using Scalar = typename Matrix::scalar_type;
    using Rhs = Kokkos::View<Scalar*, Kokkos::LayoutStride, typename decltype(native_rhs)::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // The RHS stores cell components together. Select one component every width entries.
    auto* pointer = count == 0 ? native_rhs.data() : native_rhs.data()+static_cast<std::size_t>(component)*native_rhs.stride(0);
    const Rhs component_rhs(pointer,
        Kokkos::LayoutStride(count, static_cast<std::size_t>(width)*native_rhs.stride(0)));
    return SystemView<decltype(values), Rhs, typename linalg::FaceAssemblyPlan<Exec>::entry_map_type>{
        values, component_rhs, assembly.entry_map()};
}

} // namespace krona::transport::scalar
