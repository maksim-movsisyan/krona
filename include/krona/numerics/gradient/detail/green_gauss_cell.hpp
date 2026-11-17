#pragma once

#include <krona/numerics/gradient/detail/cell_accumulator.hpp>

namespace krona::numerics::gradient::detail {

/// @brief One iteration computes the gradient of an owned cell from its incident faces.
/// @tparam Space Kokkos memory space of the mesh arrays and component descriptors.
/// @tparam T Scalar type stored in the input and output fields.
/// @tparam N Number of input components processed together.
/// @tparam Real Scalar type used for contributions and accumulation.
template <class Space, class T, std::size_t N, class Real> 
struct GreenGaussCellKernel {
    StencilData<Space> stencil;
    Batch<T, N, Space> fields;

    KOKKOS_INLINE_FUNCTION void operator()(LocalIndex cell) const {
        const auto i = static_cast<std::size_t>(cell);
        const auto& mesh = stencil.mesh;
        const auto& conn = stencil.connectivity;
        const Real volume_inv = Real{1} / static_cast<Real>(mesh.geometry.cell_volume(i));
        CellAccumulator<T, N, Space, Real> sum(fields, i);

        // The cell's CSR row lists every face that contributes to its gradient.
        for (auto k = conn.cell_faces_offsets(i); k < conn.cell_faces_offsets(i + 1); ++k) {
            const auto face = conn.cell_faces(k);
            const auto f = static_cast<std::size_t>(face);
            const auto sample = stencil.sample(cell, face);
            const bool owner = mesh.connectivity.face_owner(f) == cell;

            // Express phi_face - phi_cell as alpha * (phi_sample - phi_cell). On a boundary,
            // the sample is already the face value. Inside, alpha is the opposite cell's weight.
            const Real alpha = face >= mesh.sizes.n_inner_faces
                                   ? Real{1}
                                   : (owner ? Real{1} - stencil.template owner_weight<Real>(f)
                                            : stencil.template owner_weight<Real>(f));

            // The stored normal points out of the owner. Reverse it when gathering for the neighbour.
            const Real factor =
                (owner ? alpha : -alpha) * static_cast<Real>(mesh.geometry.face_area(f)) * volume_inv;
            const Real x = factor * static_cast<Real>(mesh.geometry.face_normal_x(f));
            const Real y = factor * static_cast<Real>(mesh.geometry.face_normal_y(f));
            const Real z = factor * static_cast<Real>(mesh.geometry.face_normal_z(f));
            sum.add(fields, sample, x, y, z);
        }

        // This iteration owns the entire output for the cell, so one ordinary write per component is enough.
        sum.store(fields, i);
    }
};

} // namespace krona::numerics::gradient::detail
