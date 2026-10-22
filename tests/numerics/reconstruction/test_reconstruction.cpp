#include "common/test_support.hpp"

#include <krona/numerics/reconstruction/mesh_stencil.hpp>
#include <krona/fields/field_storage.hpp>
#include <krona/mesh/compute/mesh_storage.hpp>
#include <krona/mesh/compute/aux_storage.hpp>
#include <krona/halo/exchange.hpp>

#include <Kokkos_NumericTraits.hpp>

#include <array>
#include <stdexcept>

namespace {
using namespace krona;
namespace rec = numerics::reconstruction;
using Exec = execution::ExecutionSpace;
using Space = typename Exec::memory_space;
using krona::test::check;
using IndexPolicy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;

template<class View>
auto host_copy(View view, const Exec& exec)
{
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    Kokkos::deep_copy(exec, host, view); exec.fence("Reconstruction test snapshot"); return host;
}
template<class R> constexpr R tolerance() { return std::is_same_v<R, float> ? R{3e-5F} : R{3e-12}; }

template<class R, std::size_t N>
struct LocalValues {
    static constexpr std::size_t components = N;
    Kokkos::Array<Kokkos::Array<R, N>, 4> entries{};
    KOKKOS_INLINE_FUNCTION R operator()(std::size_t i, std::size_t c) const { return entries[i][c]; }
};
template<class R, std::size_t N>
struct LocalGradients {
    Kokkos::Array<Kokkos::Array<rec::Vector<R>, N>, 4> entries{};
    KOKKOS_INLINE_FUNCTION rec::Vector<R> operator()(std::size_t i, std::size_t c) const { return entries[i][c]; }
};

void first_order(const Exec& exec)
{
    Kokkos::View<double*, Space> result("first order", 6);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/first_order", IndexPolicy(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            LocalValues<double, 3> values{{{{2, 3, 4}, {7, 8, 9}, {}, {}}}};
            const auto inputs = rec::data(values); // No gradient/factor callable exists: unwanted access cannot compile.
            const double nan = Kokkos::quiet_NaN_v<double>;
            const auto face = rec::face_geometry(0, 1, rec::Vector<double>{nan, nan, nan}, rec::Vector<double>{nan, nan, nan});
            const auto states = rec::evaluate(rec::FirstOrder{}, face, inputs);
            for (std::size_t c = 0; c < 3; ++c) { result(c) = states.left[c]; result(3+c) = states.right[c]; }
        });
    });
    const auto h = host_copy(result, exec);
    for (std::size_t c = 0; c < 3; ++c) {
        test::near(h(c), static_cast<double>(2+c), 0.0, "first-order left value");
        test::near(h(3+c), static_cast<double>(7+c), 0.0, "first-order right value");
    }
}

template<class R, class Layout>
void affine_test(const Exec& exec)
{
    fields::FieldStorage<R, 5, Exec, Layout> values("state", 2, fields::FieldLocation::Custom, exec);
    fields::FieldStorage<R, 15, Exec, Layout> gradient("gradients", 2, fields::FieldLocation::Custom, exec);
    const auto q = values.view(); const auto g = gradient.view();
    const rec::Vector<R> p{static_cast<R>(-0.2), static_cast<R>(0.3), static_cast<R>(0.7)}, n{static_cast<R>(1.1), static_cast<R>(-0.4), R{0.5}}, f{static_cast<R>(0.6), static_cast<R>(0.2), static_cast<R>(0.9)};
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/affine_input", IndexPolicy(exec, 0, 2), KOKKOS_LAMBDA(std::size_t i) {
            const auto point = i == 0 ? p : n;
            for (int c = 0; c < 5; ++c) {
                const R a = static_cast<R>(c+1), b = static_cast<R>(2-c), z = R{0.25}*static_cast<R>(c+2);
                q(i, c) = R{7}+a*point.x+b*point.y+z*point.z;
                g(i, 3*c) = a; g(i, 3*c+1) = b; g(i, 3*c+2) = z;
            }
        });
    });
    // Mix rank-one scalar components with different AoS strides in one batch.
    Kokkos::View<R*, Space> separate("separate scalar", 2);
    Kokkos::deep_copy(exec, separate, values.view().component(4));
    const auto inputs = rec::data(rec::pack(q.component(0), q.component(1), q.component(2), q.component(3), separate),
                                  rec::gradient_pack(g));
    Kokkos::View<R**, Space> output("affine states", 4, 5);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/affine", IndexPolicy(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            // Descriptor factories must also be callable from device kernels.
            const auto local_gradients = rec::gradients(inputs.gradients.x, inputs.gradients.y, inputs.gradients.z);
            const auto local_inputs = rec::data(inputs.values, local_gradients);
            const auto face = rec::face_geometry(0, 1, f-p, f-n, static_cast<R>(0.4));
            const auto a = rec::evaluate<R>(rec::SecondOrderUpwind{}, face, local_inputs);
            const auto b = rec::evaluate<R>(rec::Tvd{rec::tvd::VanLeer{}}, face, local_inputs);
            for (std::size_t c = 0; c < 5; ++c) {
                output(0, c) = a.left[c]; output(1, c) = a.right[c];
                output(2, c) = b.left[c]; output(3, c) = b.right[c];
            }
        });
    });
    const auto h = host_copy(output, exec);
    for (std::size_t c = 0; c < 5; ++c) {
        const R expected = R{7}+static_cast<R>(c+1)*f.x+static_cast<R>(2-static_cast<int>(c))*f.y+R{0.25}*static_cast<R>(c+2)*f.z;
        for (std::size_t row = 0; row < 4; ++row) test::near(h(row, c), expected, tolerance<R>()*R{8}, "skew affine batch reconstruction");
    }
}
void affine(const Exec& e) { affine_test<double, Kokkos::LayoutLeft>(e); affine_test<float, Kokkos::LayoutRight>(e); }

template<class L>
void tvd_policy(const Exec& exec, L limiter)
{
    Kokkos::View<double*, Space> out("limiter curve", 10);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/tvd", IndexPolicy(exec, 0, 10), KOKKOS_LAMBDA(std::size_t i) {
            const Kokkos::Array<double, 10> r{-3, 0, 0.1, 0.5, 1, 2, 10, 1e150, 1e300, Kokkos::infinity_v<double>};
            out(i) = limiter(r[i]);
        });
    });
    const auto h = host_copy(out, exec);
    test::near(h(0), 0.0, 0.0, "negative-r limiter"); test::near(h(1), 0.0, 0.0, "zero-r limiter");
    test::near(h(4), 1.0, 1e-15, "linear consistency psi(1)");
    const std::array<double, 10> r{-3, 0, 0.1, 0.5, 1, 2, 10, 1e150, 1e300, Kokkos::infinity_v<double>};
    for (std::size_t i = 2; i < r.size(); ++i)
        check(std::isfinite(h(i)) && h(i) >= 0 && h(i) <= std::min(2.0, 2*r[i])+1e-15, "TVD limiter outside Sweby region");
}
void tvd(const Exec& e)
{
    tvd_policy(e, rec::tvd::Minmod{}); tvd_policy(e, rec::tvd::Superbee{});
    tvd_policy(e, rec::tvd::MonotonizedCentral{}); tvd_policy(e, rec::tvd::VanLeer{});
    tvd_policy(e, rec::tvd::VanAlbada{}); tvd_policy(e, rec::tvd::Koren{});
    test::near(rec::tvd::VanLeer{}(0.5), 2.0/3, 1e-15, "van Leer curve");
    test::near(rec::tvd::VanAlbada{}(0.5), 0.6, 1e-15, "van Albada curve");
    test::near(rec::tvd::Superbee{}(0.5), 1.0, 0.0, "Superbee curve");
}

void darwish(const Exec& exec)
{
    Kokkos::View<double*, Space> result("Darwish", 9);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/darwish", IndexPolicy(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            LocalValues<double, 1> q; q.entries[0][0] = 1; q.entries[1][0] = 4;
            LocalGradients<double, 1> g; g.entries[0][0] = {2, 0, 0};
            const auto d = rec::data(q, g);
            const rec::CellSide<double> side{0, 1, {0.5, 0, 0}, {1, 0, 0}, 0.5};
            result(0) = rec::evaluate_side(rec::Tvd{rec::tvd::Minmod{}}, side, d)[0];
            result(1) = rec::evaluate_side(rec::Tvd{rec::tvd::VanLeer{}}, side, d)[0];
            result(2) = rec::evaluate_side(rec::Tvd{rec::tvd::Superbee{}}, side, d)[0];
            result(3) = rec::evaluate_side(rec::SecondOrderUpwind{}, side, d)[0];
            q.entries[1][0] = q.entries[0][0];
            result(4) = rec::evaluate_side(rec::Tvd{}, side, rec::data(q, g))[0];
            result(5) = rec::DarwishMoukalled{}(2.0, 3.0);
            q.entries[0][0] = 1.0/12; q.entries[1][0] = 1+1.0/12; g.entries[0][0] = {};
            result(6) = rec::evaluate_side(rec::MusclKappa{}, side, rec::data(q, g))[0];
            result(7) = rec::evaluate_side(rec::MusclKappa{0.5}, side, rec::data(q, g))[0];
            q.entries[0][0] = 0; q.entries[1][0] = 1;
            result(8) = rec::evaluate_side(rec::MusclKappa{0.5}, side, rec::data(q, g))[0];
        });
    });
    const auto h = host_copy(result, exec);
    const std::array<double, 9> expected{1.5, 1.75, 2, 2, 1, 1.0/3, 0.25, 1.0/3, 0.25};
    for (std::size_t i = 0; i < expected.size(); ++i) test::near(h(i), expected[i], 1e-14, "Darwish virtual upstream stencil");
}

struct LineStencil {
    std::size_t cells = 1;
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return cells; }
    KOKKOS_INLINE_FUNCTION std::size_t required_input_size() const { return cells+2; }
    KOKKOS_INLINE_FUNCTION std::size_t size(std::size_t) const { return 2; }
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell, std::size_t k) const {
        const auto sample = k == 0 ? cell+1 : cell+2;
        const double sign = k == 0 ? -1.0 : 1.0;
        return rec::StencilEntry<double>{sample, {cell, sample, {0.5*sign, 0, 0}, {sign, 0, 0}, 0.5}};
    }
};
struct SkewStencil : LineStencil {
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t cell, std::size_t k) const {
        auto entry = LineStencil::operator()(cell, k); entry.side.offset.y = 1; return entry;
    }
};

void bounds(const Exec& exec)
{
    Kokkos::View<double*, Space> q("q", 3), gx("gx", 1), gy("gy", 1), gz("gz", 1), factor("factor", 1);
    auto h = Kokkos::create_mirror(q); h(0) = 0.5; h(1) = 0; h(2) = 1;
    Kokkos::deep_copy(exec, q, h); Kokkos::deep_copy(exec, gx, 4.0); Kokkos::deep_copy(exec, gy, 0.0); Kokkos::deep_copy(exec, gz, 0.0);
    const auto inputs = rec::data(rec::pack(q), rec::gradients(rec::pack(gx), rec::pack(gy), rec::pack(gz)));
    rec::compute_limiters_async(exec, LineStencil{}, inputs, rec::pack(factor));
    // Immediately consume on the same instance: there is deliberately no intermediate fence.
    const auto limited = rec::data(inputs.values, inputs.gradients, rec::pack(factor));
    Kokkos::View<double*, Space> out("limited faces", 2);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/bounds", IndexPolicy(exec, 0, 2), KOKKOS_LAMBDA(std::size_t i) {
            out(i) = rec::evaluate_side(rec::LimitedLinear{}, LineStencil{}(0, i).side, limited)[0];
        });
    });
    const auto hf = host_copy(factor, exec), ho = host_copy(out, exec);
    test::near(hf(0), 0.25, 0.0, "cell limiter minimum over all faces");
    test::near(ho(0), 0.0, 0.0, "limited lower face"); test::near(ho(1), 1.0, 0.0, "limited upper face");
    // At a strict extremum all neighbours equal the owner: any nonzero slope is suppressed.
    Kokkos::deep_copy(exec, q, 0.5);
    rec::compute_limiters_async(exec, LineStencil{}, inputs, rec::pack(factor));
    test::near(host_copy(factor, exec)(0), 0.0, 0.0, "extremum limiter");
    h(0) = 1e12; h(1) = 1e12-0.125; h(2) = 1e12+0.125;
    Kokkos::deep_copy(exec, q, h); Kokkos::deep_copy(exec, gx, 0.6);
    rec::compute_limiters_async(exec, LineStencil{}, inputs, rec::pack(factor));
    test::near(host_copy(factor, exec)(0), 5.0/12, 1e-15, "large constant offset polluted limiter increments");
    h(0) = 0.5; h(1) = 0; h(2) = 1;
    Kokkos::deep_copy(exec, q, h); Kokkos::deep_copy(exec, gx, 0.5); Kokkos::deep_copy(exec, gy, 4.0);
    rec::compute_limiters_async(exec, SkewStencil{}, inputs, rec::pack(factor), rec::BarthJespersen{}, rec::Tvd{});
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/bounded_skew_tvd", IndexPolicy(exec, 0, 2), KOKKOS_LAMBDA(std::size_t i) {
            out(i) = rec::evaluate_side(rec::Limited<rec::Tvd<>>{}, SkewStencil{}(0, i).side, limited)[0];
        });
    });
    const auto skew = host_copy(out, exec);
    test::near(host_copy(factor, exec)(0), 2.0/17, 1e-15, "complete skew TVD increment was not limited");
    for (std::size_t i = 0; i < 2; ++i) check(skew(i) >= 0 && skew(i) <= 1, "multidimensional TVD correction escaped local bounds");
}

void venkat(const Exec& exec)
{
    Kokkos::View<double*, Space> out("Venkatakrishnan", 5);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/venkat", IndexPolicy(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            const rec::LimiterContext<double> a{0, 0, 0, -1, 1, 1};
            out(0) = rec::Venkatakrishnan{}(a);
            out(1) = rec::Venkatakrishnan{}(rec::LimiterContext<double>{0, 0, 0, -1, 1, 2});
            out(2) = rec::Venkatakrishnan{}(rec::LimiterContext<double>{0, 0, 0, -1e200, 1e200, 2e200});
            out(3) = rec::Venkatakrishnan{rec::ConstantRegularization<double>{1}}(rec::LimiterContext<double>{0, 0, 0, 0, 0, 1});
            out(4) = rec::Venkatakrishnan{}(rec::LimiterContext<double>{0, 0, 0, 0, 0, 0});
        });
    });
    const auto h = host_copy(out, exec);
    const std::array<double, 5> expected{0.75, 5.0/11, 5.0/11, 1.0/3, 1};
    for (std::size_t i = 0; i < expected.size(); ++i) test::near(h(i), expected[i], 1e-15, "Venkatakrishnan regularization/scaling");
    check(h(3) > 0, "regularized Venkatakrishnan must not be presented as strictly bounded");
}

void coupling(const Exec& exec)
{
    fields::FieldStorage<double, 3, Exec, Kokkos::LayoutRight> q("q", 3, fields::FieldLocation::Custom, exec),
        g("gx", 1, fields::FieldLocation::Custom, exec), zero("zero", 1, fields::FieldLocation::Custom, exec),
        factors("factors", 1, fields::FieldLocation::Custom, exec), shared("shared", 1, fields::FieldLocation::Custom, exec);
    const auto values = q.view(); const auto gx = g.view();
    zero.fill(exec, 0.0);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/coupled_inputs", IndexPolicy(exec, 0, 3), KOKKOS_LAMBDA(std::size_t c) {
            values(0, static_cast<int>(c)) = 0.5; values(1, static_cast<int>(c)) = 0; values(2, static_cast<int>(c)) = 1;
            gx(0, static_cast<int>(c)) = c == 0 ? 4.0 : c == 1 ? 2.0 : 0.0;
        });
    });
    const auto inputs = rec::data(rec::field_pack(values), rec::gradients(rec::field_pack(gx), rec::field_pack(zero.view()), rec::field_pack(zero.view())));
    Kokkos::View<double*, Space> one("one shared factor", 1);
    rec::compute_limiters_async(exec, LineStencil{}, inputs, rec::field_pack(factors.view()));
    rec::compute_limiters_async(exec, LineStencil{}, inputs, rec::field_pack(shared.view()), rec::BarthJespersen{}, rec::SecondOrderUpwind{}, rec::SharedMinimum{});
    rec::compute_limiters_async(exec, LineStencil{}, inputs, rec::pack(one), rec::BarthJespersen{}, rec::SecondOrderUpwind{}, rec::SharedMinimum{});
    const auto h = host_copy(factors.view().kokkos_view(), exec), hs = host_copy(shared.view().kokkos_view(), exec);
    const std::array<double, 3> expected{0.25, 0.5, 1};
    for (std::size_t c = 0; c < 3; ++c) {
        test::near(h(0, c), expected[c], 0.0, "componentwise factors"); test::near(hs(0, c), 0.25, 0.0, "shared factor");
    }
    test::near(host_copy(one, exec)(0), 0.25, 0.0, "single shared factor storage");
    const auto broadcast = rec::data(inputs.values, inputs.gradients, rec::pack(one));
    Kokkos::View<double*, Space> result("broadcast result", 2);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/shared_factor", IndexPolicy(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            const auto b = rec::evaluate_side(rec::LimitedLinear{}, LineStencil{}(0, 1).side, broadcast);
            result(0) = b[0]; result(1) = b[1];
        });
    });
    const auto b = host_copy(result, exec);
    test::near(b(0), 1.0, 0.0, "shared factor broadcast component zero");
    test::near(b(1), 0.75, 0.0, "shared factor broadcast component one");
}

void ranges(const Exec& exec)
{
    fields::FieldStorage<float, 3, Exec, Kokkos::LayoutRight> value("q", 7, fields::FieldLocation::Custom, exec),
        gradient("g", 5, fields::FieldLocation::Custom, exec), output("theta", 8, fields::FieldLocation::Custom, exec);
    value.fill(exec, 2.0F); gradient.fill(exec, 0.0F); output.fill(exec, -7.0F);
    const auto input = rec::data(rec::field_pack(value.view()), rec::gradients(rec::field_pack(gradient.view()), rec::field_pack(gradient.view()), rec::field_pack(gradient.view())));
    rec::compute_limiters_async<float>(exec, LineStencil{5}, input, rec::field_pack(output.view()), rec::BarthJespersen{}, rec::SecondOrderUpwind{}, rec::Componentwise{}, {1, 4});
    const auto h = host_copy(output.view().kokkos_view(), exec);
    for (std::size_t i = 0; i < 8; ++i) for (std::size_t c = 0; c < 3; ++c)
        test::near(h(i, c), i >= 1 && i < 4 ? 1.0F : -7.0F, 0.0F, "owned range or strided output tail changed");
    Kokkos::View<double*, Space> none("empty", 0);
    const auto empty = rec::data(rec::pack(none), rec::gradients(rec::pack(none), rec::pack(none), rec::pack(none)));
    struct Empty {
        KOKKOS_INLINE_FUNCTION std::size_t size() const { return 0; }
        KOKKOS_INLINE_FUNCTION std::size_t required_input_size() const { return 0; }
        KOKKOS_INLINE_FUNCTION std::size_t size(std::size_t) const { return 0; }
        KOKKOS_INLINE_FUNCTION rec::StencilEntry<double> operator()(std::size_t, std::size_t) const { return {}; }
    };
    rec::compute_limiters_async(exec, Empty{}, empty, rec::pack(none));
    exec.fence("empty/range test complete");
}

void quick(const Exec& exec)
{
    const auto averaged = rec::cell_average_quick_stencil(Kokkos::Array<std::size_t, 3>{0, 1, 2},
        Kokkos::Array<double, 3>{-1, 0, 1}, Kokkos::Array<double, 3>{1.0/12, 1.0/12, 1.0/12}, 0.5);
    const auto point = rec::point_quick_stencil(Kokkos::Array<std::size_t, 3>{0, 1, 2}, Kokkos::Array<double, 3>{-1, 0, 1}, 0.5);
    check(averaged.valid && point.valid, "uniform quadratic stencils rejected");
    test::near(averaged.weights[0], -1.0/6, 1e-15, "cell-average upstream weight");
    test::near(averaged.weights[1], 5.0/6, 1e-15, "cell-average owner weight");
    test::near(averaged.weights[2], 1.0/3, 1e-15, "cell-average downstream weight");
    test::near(point.weights[0], -1.0/8, 1e-15, "point QUICK weight");
    Kokkos::View<double*, Space> out("quadratic stencils", 7);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/quick_interpretation", IndexPolicy(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            LocalValues<double, 1> q;
            for (std::size_t j = 0; j < 4; ++j) { const double x = static_cast<double>(j)-1; q.entries[j][0] = x*x+1.0/12; }
            const auto right = rec::cell_average_quick_stencil(Kokkos::Array<std::size_t, 3>{3, 2, 1},
                Kokkos::Array<double, 3>{2, 1, 0}, Kokkos::Array<double, 3>{1.0/12, 1.0/12, 1.0/12}, 0.5);
            const auto pair = rec::evaluate(rec::Quick{}, rec::FaceStencil{averaged, right}, rec::data(q));
            out(0) = pair.left[0]; out(1) = pair.right[0];
            out(2) = rec::evaluate_side(rec::Quick{}, point, rec::data(q))[0];
            for (std::size_t j = 0; j < 4; ++j) q.entries[j][0] -= 1.0/12;
            out(3) = rec::evaluate_side(rec::Quick{}, point, rec::data(q))[0];
        });
        // Nonuniform adjacent cells; transformed coordinates test dimensional scaling and translation.
        const std::array<double, 3> scales{1e-8, 1, 1e8};
        for (std::size_t k = 0; k < scales.size(); ++k) {
            const double scale = scales[k];
            Kokkos::parallel_for("test/nonuniform_quick", IndexPolicy(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
                const Kokkos::Array<double, 3> x{-1.7, -0.75, 1.0}, width{0.6, 1.3, 2.2};
                Kokkos::Array<double, 3> centres{}, moments{};
                LocalValues<double, 1> q;
                for (std::size_t j = 0; j < 3; ++j) {
                    centres[j] = scale*(13+x[j]); moments[j] = scale*scale*width[j]*width[j]/12;
                    q.entries[j][0] = 2+3*x[j]+4*(x[j]*x[j]+width[j]*width[j]/12);
                }
                const auto stencil = rec::cell_average_quick_stencil(Kokkos::Array<std::size_t, 3>{0, 1, 2}, centres, moments, scale*12.9);
                out(4+k) = rec::evaluate_side(rec::Quick{}, stencil, rec::data(q))[0];
            });
        }
    });
    const auto h = host_copy(out, exec);
    test::near(h(0), 0.25, 1e-15, "cell-average quadratic reconstruction");
    test::near(h(1), 0.25, 1e-15, "reverse upwind quadratic reconstruction");
    test::near(h(2)-h(0), 1.0/12, 1e-15, "point weights must not be silently used for cell averages");
    test::near(h(3), 0.25, 1e-15, "explicit point-valued QUICK");
    for (std::size_t i = 4; i < 7; ++i) test::near(h(i), 1.74, 5e-13, "nonuniform scaled quadratic fit");
}

struct Tensor {
    rec::SymmetricTensor<double> value;
    KOKKOS_INLINE_FUNCTION auto operator()(std::size_t, std::size_t = 0) const { return value; }
};
struct Factor {
    static constexpr std::size_t components = 1;
    double value;
    KOKKOS_INLINE_FUNCTION double operator()(std::size_t, std::size_t) const { return value; }
};
KOKKOS_INLINE_FUNCTION double polynomial(rec::Vector<double> x)
{
    return 1+2*x.x-3*x.y+0.5*x.z + x.x*x.x+2*x.y*x.y-0.5*x.z*x.z+
        0.5*x.x*x.y-0.7*x.x*x.z+0.3*x.y*x.z;
}
void quadratic(const Exec& exec)
{
    Kokkos::View<double*, Space> out("3D quadratic", 4);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/volume_moments", IndexPolicy(exec, 0, 1), KOKKOS_LAMBDA(std::size_t) {
            const rec::Vector<double> centre{0.4, -0.2, 0.3}, target{0.8, 0.1, -0.1};
            const Kokkos::Array<rec::Vector<double>, 3> edge{{{0.7, 0.2, 0}, {-0.1, 0.6, 0.1}, {0.2, 0, 0.5}}};
            rec::SymmetricTensor<double> moment;
            for (std::size_t i = 0; i < 3; ++i) {
                const auto e = edge[i];
                moment.xx += e.x*e.x/3; moment.yy += e.y*e.y/3; moment.zz += e.z*e.z/3;
                moment.xy += e.x*e.y/3; moment.xz += e.x*e.z/3; moment.yz += e.y*e.z/3;
            }
            Kokkos::Array<rec::Vector<double>, 8> positions{};
            double average = 0;
            for (std::size_t i = 0; i < 8; ++i) {
                const double a = (i & 1 ? 1.0 : -1.0)/Kokkos::sqrt(3.0);
                const double b = (i & 2 ? 1.0 : -1.0)/Kokkos::sqrt(3.0);
                const double c = (i & 4 ? 1.0 : -1.0)/Kokkos::sqrt(3.0);
                positions[i] = {centre.x+a*edge[0].x+b*edge[1].x+c*edge[2].x,
                    centre.y+a*edge[0].y+b*edge[1].y+c*edge[2].y,
                    centre.z+a*edge[0].z+b*edge[1].z+c*edge[2].z};
                average += polynomial(positions[i])/8;
            }
            LocalValues<double, 1> q; q.entries[0][0] = average;
            LocalGradients<double, 1> g;
            g.entries[0][0] = {2+2*centre.x+0.5*centre.y-0.7*centre.z,
                -3+0.5*centre.x+4*centre.y+0.3*centre.z, 0.5-0.7*centre.x+0.3*centre.y-centre.z};
            const Tensor hessian{{2, 4, -1, 0.5, -0.7, 0.3}}, moments{moment};
            const auto input = rec::data(q, g, rec::UnitFactors{}, hessian, moments);
            out(0) = rec::evaluate_side(rec::Quadratic{}, rec::CellSide<double>{0, 0, target-centre}, input)[0]-polynomial(target);
            double limited_average = 0, unlimited_average = 0;
            for (std::size_t i = 0; i < 8; ++i) {
                const rec::CellSide<double> side{0, 0, positions[i]-centre};
                unlimited_average += rec::evaluate_side(rec::Quadratic{}, side, input)[0]/8;
                limited_average += rec::evaluate_side(rec::LimitedQuadratic{}, side, rec::data(q, g, Factor{0.3}, hessian, moments))[0]/8;
            }
            out(1) = unlimited_average-average; out(2) = limited_average-average;
            out(3) = rec::evaluate_side(rec::LimitedQuadratic{}, rec::CellSide<double>{0, 0, target-centre}, rec::data(q, g, Factor{0}, hessian, moments))[0]-average;
        });
    });
    const auto h = host_copy(out, exec);
    for (std::size_t i = 0; i < 4; ++i) test::near(h(i), 0.0, 2e-14, "3D quadratic exactness or preservation of the cell average");
}

std::array<double, 3> reconstruction_errors(std::size_t count, const Exec& exec)
{
    Kokkos::View<double*[3], Space> error("reconstruction error", count);
    const double step = 2*3.14159265358979323846/static_cast<double>(count);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/convergence", IndexPolicy(exec, 0, count), KOKKOS_LAMBDA(std::size_t i) {
            const double centre = (static_cast<double>(i)+0.5)*step;
            const double target = centre+step/2;
            LocalValues<double, 1> q; LocalGradients<double, 1> g;
            for (std::size_t j = 0; j < 3; ++j) {
                const double x = centre+(static_cast<double>(j)-1)*step;
                q.entries[j][0] = (Kokkos::cos(x-step/2)-Kokkos::cos(x+step/2))/step+
                    0.25*(Kokkos::sin(2*x+step)-Kokkos::sin(2*x-step))/(2*step);
            }
            g.entries[1][0] = {Kokkos::cos(centre)-0.5*Kokkos::sin(2*centre), 0, 0};
            const auto d = rec::data(q, g);
            const rec::CellSide<double> side{1, 2, {step/2, 0, 0}};
            const auto stencil = rec::cell_average_quick_stencil(Kokkos::Array<std::size_t, 3>{0, 1, 2},
                Kokkos::Array<double, 3>{-step, 0, step}, Kokkos::Array<double, 3>{step*step/12, step*step/12, step*step/12}, step/2);
            const double exact = Kokkos::sin(target)+0.25*Kokkos::cos(2*target);
            error(i, 0) = rec::evaluate_side(rec::FirstOrder{}, side, d)[0]-exact;
            error(i, 1) = rec::evaluate_side(rec::SecondOrderUpwind{}, side, d)[0]-exact;
            error(i, 2) = rec::evaluate_side(rec::Quick{}, stencil, d)[0]-exact;
        });
    });
    const auto h = host_copy(error, exec);
    std::array<double, 3> norm{};
    for (std::size_t i = 0; i < count; ++i) for (std::size_t c = 0; c < 3; ++c) norm[c] += h(i, c)*h(i, c);
    for (auto& value : norm) value = std::sqrt(value/static_cast<double>(count));
    return norm;
}
void convergence(const Exec& exec)
{
    auto previous = reconstruction_errors(32, exec);
    for (const auto count : {std::size_t{64}, std::size_t{128}}) {
        const auto current = reconstruction_errors(count, exec);
        for (std::size_t c = 0; c < 3; ++c) {
            const double expected = c == 0 ? 2 : c == 1 ? 4 : 8;
            test::near(previous[c]/current[c], expected, 0.2, "measured face reconstruction order");
        }
        previous = current;
    }
}

struct Fixture { mesh::MeshPart mesh; mesh::MeshAuxConnectivity incidence; };
Fixture make_mesh(mpi::CommView comm, bool distributed)
{
    Fixture f;
    auto& m = f.mesh;
    m.rank = comm.rank(); m.nprocs = comm.size(); m.faces_grouped = true;
    m.n_own = distributed ? 1 : 3; m.n_cells = distributed ? 2 : 3;
    if (distributed) {
        m.cell_centroid_x = {static_cast<GeometryReal>(comm.rank())+GeometryReal{0.5},
                             static_cast<GeometryReal>(1-comm.rank())+GeometryReal{0.5}};
        m.face_owner = {0, 0}; m.face_neigh = {1, -1};
        m.face_centroid_x = {1, static_cast<GeometryReal>(2*comm.rank())};
        m.face_normal_x = {comm.rank() == 0 ? GeometryReal{1} : GeometryReal{-1}, comm.rank() == 0 ? GeometryReal{-1} : GeometryReal{1}};
        m.n_internal_faces = 0; m.n_inner_faces = 1;
        m.nb_ranks = {1-comm.rank()}; m.send_offsets = {0, 1}; m.recv_offsets = {0, 1};
        m.send_owned_local = {0}; m.recv_ghost_local = {1};
        f.incidence.cell_faces = {0, 1}; f.incidence.cell_faces_offsets = {0, 2};
    } else {
        m.cell_centroid_x = {0.5, 1.5, 2.5};
        m.face_owner = {0, 1, 0, 2}; m.face_neigh = {1, 2, -1, -1};
        m.face_centroid_x = {1, 2, 0, 3}; m.face_normal_x = {1, 1, -1, 1};
        m.n_internal_faces = m.n_inner_faces = 2;
        f.incidence.cell_faces = {0, 2, 0, 1, 1, 3}; f.incidence.cell_faces_offsets = {0, 2, 4, 6};
    }
    m.n_faces = static_cast<LocalIndex>(m.face_owner.size());
    m.cell_centroid_y.assign(static_cast<std::size_t>(m.n_cells), 0); m.cell_centroid_z = m.cell_centroid_y;
    m.cell_volume.assign(static_cast<std::size_t>(m.n_cells), 1);
    m.face_centroid_y.assign(static_cast<std::size_t>(m.n_faces), 0); m.face_centroid_z = m.face_centroid_y;
    m.face_normal_y = m.face_centroid_y; m.face_normal_z = m.face_centroid_y;
    m.face_area.assign(static_cast<std::size_t>(m.n_faces), 1);
    f.incidence.n_cells_own = m.n_own; f.incidence.n_faces = m.n_faces;
    f.incidence.active_mask = mesh::AuxConnType::CellFaces;
    return f;
}

void mesh_case(const Exec& exec, bool distributed)
{
    const auto comm = mpi::CommView::world();
    if (distributed) check(comm.size() == 2, "distributed reconstruction requires exactly two ranks");
    const auto f = make_mesh(comm, distributed);
    using mesh::compute::MeshData;
    const mesh::compute::MeshStorage<Exec> mesh(f.mesh,
        {.data = MeshData::FaceCells | MeshData::CellGeometry | MeshData::FaceGeometry | MeshData::Halo}, exec);
    const mesh::compute::AuxConnectivityStorage<Exec> incidence(f.mesh, f.incidence, exec);
    const rec::MeshStencil<Space> stencil(mesh.view(), incidence.view());
    const auto total = stencil.required_input_size();
    fields::FieldStorage<double, 2, Exec, Kokkos::LayoutRight> q("state", total, fields::FieldLocation::Custom, exec),
        theta("theta", static_cast<std::size_t>(f.mesh.n_cells), fields::FieldLocation::Custom, exec);
    fields::FieldStorage<double, 6, Exec, Kokkos::LayoutRight> gradient("gradient", static_cast<std::size_t>(f.mesh.n_cells), fields::FieldLocation::Custom, exec);
    q.fill(exec, -99); theta.fill(exec, -99); gradient.fill(exec, -99);
    const auto values = q.view(); const auto g = gradient.view();
    const auto mv = mesh.view();
    const auto owned = static_cast<std::size_t>(f.mesh.n_own);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/owned_and_boundary", IndexPolicy(exec, 0, total), KOKKOS_LAMBDA(std::size_t i) {
            double x = 0;
            if (i < owned) x = static_cast<double>(mv.geometry.cell_centroid_x(i));
            else if (i >= static_cast<std::size_t>(mv.sizes.n_cells))
                x = static_cast<double>(mv.geometry.face_centroid_x(i-static_cast<std::size_t>(mv.sizes.n_cells)+static_cast<std::size_t>(mv.sizes.n_inner_faces)));
            else return; // Deliberately leave ghost values stale until the exchange.
            values(i, 0) = 2+3*x; values(i, 1) = 5-2*x;
            if (i < owned) { g(i, 0) = 3; g(i, 1) = g(i, 2) = 0; g(i, 3) = -2; g(i, 4) = g(i, 5) = 0; }
        });
    });
    halo::Exchange<double, Exec> value_exchange(mesh.view(), mesh.halo_plan(), comm, 21, exec);
    value_exchange.add_field(q.view()); value_exchange.prepare(); value_exchange.exchange();
    const auto raw = rec::data(rec::field_pack(q.view()), rec::gradient_pack(gradient.view()));
    rec::compute_limiters_async(exec, stencil, raw, rec::field_pack(theta.view()));
    halo::Exchange<double, Exec> derived_exchange(mesh.view(), mesh.halo_plan(), comm, 22, exec);
    derived_exchange.add_field(gradient.view()); derived_exchange.add_field(theta.view());
    derived_exchange.prepare(); derived_exchange.exchange();
    const auto input = rec::data(raw.values, raw.gradients, rec::field_pack(theta.view()));
    Kokkos::View<double***, Space> out("face states", static_cast<std::size_t>(f.mesh.n_faces), 2, 2);
    test::complete(exec, [&] {
        Kokkos::parallel_for("test/mesh_faces", IndexPolicy(exec, 0, static_cast<std::size_t>(f.mesh.n_faces)), KOKKOS_LAMBDA(std::size_t face) {
            if (face < static_cast<std::size_t>(mv.sizes.n_inner_faces)) {
                const auto a = rec::evaluate(rec::LimitedLinear{}, stencil.face<rec::LimitedLinear>(face), input);
                const auto b = rec::evaluate(rec::Tvd{}, stencil.face<rec::Tvd<>>(face), input);
                for (std::size_t c = 0; c < 2; ++c) { out(face, 0, c) = a.left[c]; out(face, 1, c) = a.right[c]; }
                // Both independent paths must reproduce the affine face state.
                for (std::size_t c = 0; c < 2; ++c) out(face, 0, c) += b.left[c]-b.right[c];
            } else {
                const auto a = rec::evaluate_side(rec::LimitedLinear{}, stencil.boundary_side(face), input);
                for (std::size_t c = 0; c < 2; ++c) out(face, 0, c) = out(face, 1, c) = a[c];
            }
        });
    });
    const auto h = host_copy(out, exec);
    const auto factors = host_copy(theta.view().kokkos_view(), exec);
    for (std::size_t face = 0; face < static_cast<std::size_t>(f.mesh.n_faces); ++face) {
        const auto x = static_cast<double>(f.mesh.face_centroid_x[face]);
        for (std::size_t side = 0; side < 2; ++side) {
            test::near(h(face, side, 0), 2+3*x, 1e-13, "mesh/MPI reconstructed scalar zero");
            test::near(h(face, side, 1), 5-2*x, 1e-13, "mesh/MPI reconstructed scalar one");
        }
    }
    for (std::size_t cell = 0; cell < static_cast<std::size_t>(f.mesh.n_cells); ++cell)
        for (std::size_t c = 0; c < 2; ++c) test::near(factors(cell, c), 1.0, 0.0, "owned/ghost affine limiter factors");
}
void mesh_test(const Exec& e) { mesh_case(e, false); }
void distributed(const Exec& e) { mesh_case(e, true); }

template<class F> void rejects(F action)
{
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid host descriptor was accepted");
}
void misuse(const Exec& exec)
{
    Kokkos::View<double*, Space> values("values", 3), gradient("gradient", 1), output("output", 1);
    const auto input = rec::data(rec::pack(values), rec::gradients(rec::pack(gradient), rec::pack(gradient), rec::pack(gradient)));
    rejects([&] { rec::compute_limiters_async(exec, LineStencil{}, input, rec::pack(output), rec::BarthJespersen{}, rec::SecondOrderUpwind{}, rec::Componentwise{}, {0, 2}); });
    rejects([&] { rec::compute_limiters_async(exec, LineStencil{2}, input, rec::pack(output)); });
    Kokkos::View<double*, Space> none("none", 0);
    rejects([&] { rec::compute_limiters_async(exec, LineStencil{}, rec::data(rec::pack(values), rec::gradients(rec::pack(none), rec::pack(none), rec::pack(none))), rec::pack(output)); });
    const auto bad = rec::cell_average_quick_stencil(Kokkos::Array<std::size_t, 3>{0, 1, 2},
        Kokkos::Array<double, 3>{0, 0, 0}, Kokkos::Array<double, 3>{0, 0, 0}, 0.5);
    check(!bad.valid && std::isnan(bad.weights[0]), "degenerate QUICK geometry was silently accepted");
    const auto negative = rec::cell_average_quick_stencil(Kokkos::Array<std::size_t, 3>{0, 1, 2},
        Kokkos::Array<double, 3>{-1, 0, 1}, Kokkos::Array<double, 3>{0, -1, 0}, 0.5);
    check(!negative.valid, "negative cell moment was accepted");
}

} // namespace

int main(int argc, char** argv)
{
    const std::array cases{
        test::Case{"first_order", first_order}, test::Case{"affine", affine}, test::Case{"tvd", tvd},
        test::Case{"darwish", darwish}, test::Case{"bounds", bounds}, test::Case{"venkat", venkat},
        test::Case{"coupling", coupling}, test::Case{"ranges", ranges}, test::Case{"quick", quick},
        test::Case{"quadratic", quadratic}, test::Case{"convergence", convergence},
        test::Case{"mesh", mesh_test}, test::Case{"misuse", misuse}};
    const std::array mpi_cases{test::Case{"distributed", distributed}};
    if (argc == 2 && std::string_view(argv[1]) == "distributed")
        return test::run(argc, argv, "numerics.reconstruction", mpi_cases);
    return test::run(argc, argv, "numerics.reconstruction", cases);
}