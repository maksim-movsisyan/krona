#include "fixture.hpp"

#include <krona/boundary/physical/models.hpp>
#include <krona/transport/viscous/boundary_jacobian.hpp>
#include <krona/transport/viscous/detail/inner_face.hpp>
#include <array>

namespace viscous_test {

/// @brief Recovers affine gradients on a skew face and checks the traction and work signs.
void skew_face(const Exec&) {
    using namespace vt::detail;
    const Vector<R> normal{static_cast<R>(0.6), static_cast<R>(0.8), 0};
    const Vector<R> connection{2, 1, 0}, skew{0, 1, 0};
    const InnerGeometry<R> geometry{normal, connection, skew, connection.dot(normal), R{0.5}};
    const Gradients<R> g{{1, 2, 0}, {}, {}, {3, -2, 0}};
    const State<R> left{1, {3, 0, 0}, 300}, right{1, {7, 0, 0}, 304};
    FaceData<R> face;
    test::check(prepare_inner(geometry, left, right, g, g, properties, R{0}, face), "affine face preparation");
    const auto flux = outward_flux(face);
    test::check(flux.valid(), "affine face flux");
    const R tx = R{8} / R{3} * normal.x + R{4} * normal.y;
    const R ty = R{4} * normal.x - R{4} / R{3} * normal.y;
    const R q = -R{3} * (R{3} * normal.x - R{2} * normal.y);
    test::near(static_cast<double>(face.velocity.x), 7., tolerance(), "velocity at the actual centroid");
    test::near(static_cast<double>(flux.value.momentum.x), static_cast<double>(-tx), tolerance(), "normal stress");
    test::near(static_cast<double>(flux.value.momentum.y), static_cast<double>(-ty), tolerance(), "shear stress");
    test::near(static_cast<double>(flux.value.energy), static_cast<double>(q - R{7} * tx),
               8 * tolerance(), "Fourier heat and viscous work");
}

/// @brief Checks both compact primitive blocks against the same face discretization.
void inner_jacobian(const Exec&) {
    using namespace vt::detail;
    const Vector<R> n{static_cast<R>(0.6), static_cast<R>(0.8), 0};
    const InnerGeometry<R> geometry{n, {2, 1, 0}, {0, 1, 0}, 2, static_cast<R>(0.625)};
    const Gradients<R> gl{{1, 2, 0}, {0, 1, 0}, {}, {3, -2, 0}};
    const Gradients<R> gr{{2, 1, 0}, {0, 3, 1}, {}, {1, 1, 0}};
    const State<R> left{300, {3, 2, 1}, 300}, right{310, {7, -3, 2}, 304};
    const auto flux_at = [&](State<R> ql, State<R> qr) {
        FaceData<R> data;
        test::check(prepare_inner(geometry, ql, qr, gl, gr, properties, R{1}, data), "sampled inner face");
        const auto flux = outward_flux(data);
        test::check(flux.valid(), "sampled inner flux");
        return krona::transport::detail::flux_array(flux.value);
    };
    FaceData<R> face;
    test::check(prepare_inner(geometry, left, right, gl, gr, properties, R{1}, face), "base inner face");
    const auto value = surface(face);
    test::check(value.valid(), "base traction");

    for (const bool owner : {true, false}) {
        const R sign = owner ? R{-1} : R{1};
        const R weight = owner ? geometry.owner_weight : R{1} - geometry.owner_weight;
        const auto block = inner_primitive_block(face, value.value.traction, sign / geometry.height, weight);
        for (std::size_t column = 0; column < 5; ++column) {
            // Constant properties and the fixed input gradients implement the frozen linearization.
            constexpr R step = R{0.125};
            auto perturb = [&](R delta) {
                auto q = owner ? left : right;
                if (column == 0) q.pressure += delta;
                else if (column == 4) q.temperature += delta;
                else q.velocity = q.velocity + axis<R>(column - 1) * delta;
                return owner ? flux_at(q, right) : flux_at(left, q);
            };
            const auto plus = perturb(step);
            const auto minus = perturb(-step);
            for (std::size_t row = 0; row < 5; ++row)
                test::near(static_cast<double>((plus[row] - minus[row]) / (R{2} * step)),
                    static_cast<double>(block(row, column)), 16 * tolerance(), "inner flux derivative");
        }
    }
}

template<class Condition>
void compare_wall(const krona::boundary::FaceContext<R>& c, const Condition& bc, bool free_slip) {
    using namespace vt::detail;
    const ph::Compressible model{gas, ph::NoCharacteristics{}};
    const State<R> owner{320, {4, -2, 1}, 310};
    const Gradients<R> gradients{{1, 2, 3}, {2, 4, 8}, {1, 3, 5}, {-1, 2, -3}};
    BoundaryFace<R> face;
    test::check(prepare_boundary(c, bc, model, owner, gradients, properties, R{1}, face), "wall preparation");
    const auto surface_result = surface(face.data);
    test::check(surface_result.valid(), "wall surface law");
    if (free_slip) {
        const auto traction = surface_result.value.traction;
        const auto cross = traction.cross(face.data.normal);
        test::near(static_cast<double>(cross.norm2()), 0., tolerance(), "zero tangential traction");
        test::equal(surface_result.value.outward_heat_flux, R{0}, "zero slip-wall conduction");
    }
    if constexpr (std::is_same_v<Condition, ph::HeatFluxWall<R>>)
        test::equal(surface_result.value.outward_heat_flux, bc.flux, "prescribed outward heat");

    const vt::BoundaryLinearizationInput<R> input{c, owner, gradients, face};
    const auto exact = vt::WallLinearization{}(input, bc, model, properties);
    const auto difference = vt::FiniteDifferenceBoundary{}(input, bc, model, properties);
    test::check(exact.valid && difference.valid, "valid analytic and numerical wall blocks");
    const double tol = std::is_same_v<R, float> ? 3e-4 : 2e-9;
    for (std::size_t i = 0; i < exact.block.values.size(); ++i) {
        const auto reference = static_cast<double>(exact.block.values[i]);
        test::near(static_cast<double>(difference.block.values[i]), reference, tol * (1. + std::abs(reference)),
                   "analytic frozen derivative matches boundary flux differences");
    }
}

/// @brief Covers moving walls, each thermal condition and an inclined symmetry plane.
void walls(const Exec&) {
    krona::boundary::FaceContext<R> c;
    c.distance = R{0.5};
    c.area = 2;
    c.nx = static_cast<R>(0.6);
    c.ny = static_cast<R>(0.8);
    c.tx = static_cast<R>(-0.4);
    c.ty = static_cast<R>(0.3);
    const vt::detail::Vector<R> moving{4, -3, 0};
    compare_wall(c, ph::IsothermalWall<R>{300, moving}, false);
    compare_wall(c, ph::AdiabaticWall<R>{moving}, false);
    compare_wall(c, ph::HeatFluxWall<R>{6, moving}, false);
    compare_wall(c, ph::SlipWall<R>{moving}, true);
    compare_wall(c, ph::Symmetry{}, true);
}

} // namespace viscous_test

int main(int argc, char** argv) {
    using namespace viscous_test;
    const std::array cases{test::Case{"skew_face", skew_face}, test::Case{"walls", walls},
                           test::Case{"inner_jacobian", inner_jacobian},
                           test::Case{"rows", rows}, test::Case{"distributed", distributed}};
    return test::run(argc, argv, "transport.viscous", cases);
}
