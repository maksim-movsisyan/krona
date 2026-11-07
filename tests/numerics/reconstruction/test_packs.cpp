#include "common/test_support.hpp"

#include <krona/fields/field_storage.hpp>
#include <krona/numerics/reconstruction/reconstruction.hpp>

#include <array>

namespace {
using namespace krona;
namespace rec = numerics::reconstruction;
using Exec = execution::ExecutionSpace;
using Space = typename Exec::memory_space;

/// @brief Returns an independent host snapshot after completing work on exec.
template<class View>
auto snapshot(View view, const Exec& exec)
{
    auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, view);
    Kokkos::deep_copy(exec, host, view);
    exec.fence("Reconstruction pack test snapshot");
    return host;
}

/// @brief Checks mixed-type gradient triples, axis ordering, borrowed storage and common extent.
void gradient_packs(const Exec& exec)
{
    fields::FieldStorage<float, 3, Exec, Kokkos::LayoutRight> first("first gradient", 2, fields::FieldLocation::Custom, exec);
    fields::FieldStorage<double, 6, Exec, Kokkos::LayoutLeft> rest("other gradients", 3, fields::FieldLocation::Custom, exec);
    const auto a = first.view(); const auto b = rest.view();
    Kokkos::parallel_for("test/gradient_pack_input", Kokkos::RangePolicy<Exec>(exec, 0, 2), KOKKOS_LAMBDA(int i) {
        for (int axis = 0; axis < 3; ++axis) {
            a(i, axis) = static_cast<float>(100*i+axis);
            b(i, axis) = 100*i+10+axis;
            b(i, 3+axis) = 100*i+20+axis;
        }
    });

    // Nine scalar components become three gradients without copying either field.
    const auto packed = rec::gradient_pack(fields::pack(a, b));
    static_assert(decltype(packed)::components == 3);
    static_assert(std::is_same_v<typename decltype(packed)::value_type, double>);
    test::equal(packed.size(), std::size_t{2}, "gradient pack minimum extent");
    test::check(packed.x.template component<0>().data() == a.component(0).data(), "gradient adapter copied its first component");
    test::check(packed.z.template component<2>().data() == b.component(5).data(), "gradient adapter copied its last component");

    // Read the adapter in a kernel and compare each component/axis with its encoded value.
    Kokkos::View<double***, Space> out("gradient vectors", 2, 3, 3);
    Kokkos::parallel_for("test/gradient_pack", Kokkos::RangePolicy<Exec>(exec, 0, 2), KOKKOS_LAMBDA(int i) {
        for (std::size_t c = 0; c < 3; ++c) {
            const auto g = packed(static_cast<std::size_t>(i), c);
            out(i, c, 0) = g.x; out(i, c, 1) = g.y; out(i, c, 2) = g.z;
        }
    });
    const auto host = snapshot(out, exec);
    for (int i = 0; i < 2; ++i) for (int c = 0; c < 3; ++c) for (int axis = 0; axis < 3; ++axis)
        test::near(host(i, c, axis), static_cast<double>(100*i+10*c+axis), 0.0, "mixed gradient axis order");
}

/// @brief Supplies one owner and two neighbouring samples at opposite face offsets.
struct Samples {
    /// @brief Returns the number of cells whose factors are computed.
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return 1; }

    /// @brief Includes the owner and both sample values in the input extent.
    KOKKOS_INLINE_FUNCTION std::size_t required_input_size() const { return 3; }

    /// @brief Returns the two face samples used by this owner.
    KOKKOS_INLINE_FUNCTION std::size_t size(std::size_t) const { return 2; }

    /// @brief Returns a sample index and its owner-side reconstruction geometry.
    KOKKOS_INLINE_FUNCTION rec::StencilEntry<double> operator()(std::size_t cell, std::size_t k) const
    {
        const auto sample = k+1;
        const double sign = k == 0 ? -1.0 : 1.0;
        return {sample, {cell, sample, {0.5*sign, 0, 0}, {sign, 0, 0}, 0.5}};
    }
};

/// @brief Writes float/double limiter factors and checks shared coupling and untouched tails.
void mixed_factors(const Exec& exec)
{
    fields::FieldStorage<float, 1, Exec> qa("qa", 3, fields::FieldLocation::Custom, exec), fa("fa", 2, fields::FieldLocation::Custom, exec);
    fields::FieldStorage<double, 1, Exec> qb("qb", 3, fields::FieldLocation::Custom, exec), fb("fb", 2, fields::FieldLocation::Custom, exec);
    fields::FieldStorage<float, 3, Exec> ga("ga", 1, fields::FieldLocation::Custom, exec);
    fields::FieldStorage<double, 3, Exec> gb("gb", 1, fields::FieldLocation::Custom, exec);
    const auto a = qa.view(); const auto b = qb.view(); const auto x = ga.view(); const auto y = gb.view();

    // Different slopes require factors of one quarter and one half for the same bounds.
    fa.fill_async(exec, -7); fb.fill_async(exec, -7);
    Kokkos::parallel_for("test/mixed_factor_input", Kokkos::RangePolicy<Exec>(exec, 0, 1), KOKKOS_LAMBDA(int) {
        a(0, 0) = 0.5F; a(1, 0) = 0; a(2, 0) = 1;
        b(0, 0) = 0.5; b(1, 0) = 0; b(2, 0) = 1;
        x(0, 0) = 4; x(0, 1) = x(0, 2) = 0;
        y(0, 0) = 2; y(0, 1) = y(0, 2) = 0;
    });
    const auto raw = rec::data(fields::pack(a.as_const(), b.as_const()), rec::gradient_pack(fields::pack(x.as_const(), y.as_const())));
    const auto output = fields::pack(fa.view(), fb.view());
    static_assert(!decltype(output)::homogeneous);
    rec::compute_limiters_async<double>(exec, Samples{}, raw, output);
    test::near(snapshot(fa.view().component(0), exec)(0), 0.25F, 0.0F, "float factor destination");
    test::near(snapshot(fb.view().component(0), exec)(0), 0.5, 0.0, "double factor destination");

    // Typed component writes must remain usable by a later mixed-type reconstruction.
    const auto limited = rec::data(raw.values, raw.gradients, output);
    Kokkos::View<double*, Space> traces("limited mixed traces", 2);
    Kokkos::parallel_for("test/mixed_factor_read", Kokkos::RangePolicy<Exec>(exec, 0, 1), KOKKOS_LAMBDA(int) {
        const auto trace = rec::evaluate_side<double>(rec::LimitedLinear{}, Samples{}(0, 1).side, limited);
        traces(0) = trace[0]; traces(1) = trace[1];
    });
    const auto host = snapshot(traces, exec);
    test::near(host(0), 1.0, 0.0, "limited float component");
    test::near(host(1), 1.0, 0.0, "limited double component");

    // SharedMinimum applies the strictest factor to both components and preserves tails.
    rec::compute_limiters_async<double>(exec, Samples{}, raw, output, rec::BarthJespersen{}, rec::SecondOrderUpwind{}, rec::SharedMinimum{});
    const auto ha = snapshot(fa.view().component(0), exec);
    const auto hb = snapshot(fb.view().component(0), exec);
    test::near(ha(0), 0.25F, 0.0F, "shared float destination"); test::near(hb(0), 0.25, 0.0, "shared double destination");
    test::near(ha(1), -7.0F, 0.0F, "float tail changed"); test::near(hb(1), -7.0, 0.0, "double tail changed");
}

/// @brief Supplies a constant tensor and a configurable extent for requirement validation.
struct Tensors {
    std::size_t extent;

    /// @brief Returns the extent advertised to host-side validation.
    KOKKOS_INLINE_FUNCTION std::size_t size() const { return extent; }

    /// @brief Returns a tensor whose xx entry gives an analytically known increment.
    KOKKOS_INLINE_FUNCTION rec::SymmetricTensor<double> operator()(std::size_t, std::size_t = 0) const
    { return {2, 0, 0, 0, 0, 0}; }
};

/// @brief Requires cell moments while deliberately requiring no gradient or Hessian.
struct MomentOnly {
    static constexpr unsigned requirements = rec::Moments;

    /// @brief Uses only the owner's xx moment as the proposed increment.
    template<class R, class S, class D>
    KOKKOS_INLINE_FUNCTION R increment(const S& s, const D& d, std::size_t) const { return static_cast<R>(d.moments(s.cell).xx); }
};

/// @brief Requires a Hessian while deliberately requiring no gradient or moment.
struct HessianOnly {
    static constexpr unsigned requirements = rec::Hessians;

    /// @brief Uses only the selected scalar Hessian's xx component.
    template<class R, class S, class D>
    KOKKOS_INLINE_FUNCTION R increment(const S& s, const D& d, std::size_t c) const { return static_cast<R>(d.hessians(s.cell, c).xx); }
};

/// @brief Checks independent tensor requirements or deliberately shortens one required input.
template<bool RejectMoment = false, bool RejectHessian = false>
void requirements_test(const Exec& exec)
{
    Kokkos::View<double*, Space> q("values", 3), f("factor", 1);
    Kokkos::parallel_for("test/requirement_input", Kokkos::RangePolicy<Exec>(exec, 0, 1), KOKKOS_LAMBDA(int) {
        q(0) = 0.5; q(1) = 0; q(2) = 1;
    });

    // Each custom scheme gets only its declared tensor; absent unrelated data stays valid.
    const auto moments = rec::data(fields::pack(q), rec::NoGradients{}, rec::UnitFactors{}, rec::NoHessians{}, Tensors{RejectMoment ? 0U : 1U});
    rec::compute_limiters_async(exec, Samples{}, moments, fields::pack(f), rec::BarthJespersen{}, MomentOnly{});
    const auto hessians = rec::data(fields::pack(q), rec::NoGradients{}, rec::UnitFactors{}, Tensors{RejectHessian ? 0U : 1U});
    rec::compute_limiters_async(exec, Samples{}, hessians, fields::pack(f), rec::BarthJespersen{}, HessianOnly{});
    test::near(snapshot(f, exec)(0), 0.25, 0.0, "independent tensor requirements");
}

/// @brief Checks valid moment-only and Hessian-only schemes without gradient storage.
void requirements(const Exec& exec) { requirements_test(exec); }

/// @brief Rejects a required moment descriptor with zero extent.
void reject_moment(const Exec& exec) { requirements_test<true>(exec); }

/// @brief Rejects a required Hessian descriptor with zero extent.
void reject_hessian(const Exec& exec) { requirements_test<false, true>(exec); }
} // namespace

/// @brief Runs adapter cases or an individually selected tensor-extent rejection.
int main(int argc, char** argv)
{
    const std::array cases{test::Case{"gradient_packs", gradient_packs}, test::Case{"mixed_factors", mixed_factors}, test::Case{"requirements", requirements}};
    const std::array rejected{test::Case{"reject_moment", reject_moment}, test::Case{"reject_hessian", reject_hessian}};
    if (argc == 2 && std::string_view(argv[1]).starts_with("reject_"))
        return test::run(argc, argv, "numerics.reconstruction", rejected);
    return test::run(argc, argv, "numerics.reconstruction", cases);
}
