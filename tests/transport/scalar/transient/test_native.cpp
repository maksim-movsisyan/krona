#include "common/test_support.hpp"
#include "test_helpers.hpp"
#include <krona/transport/scalar/transient/operator.hpp>
#include <krona/linalg/bsr_matrix.hpp>
#include <krona/linalg/csr_matrix.hpp>
#include <array>
#include <vector>

namespace {
using namespace krona;
using namespace krona::test::scalar_transient;
namespace tt = transport::scalar::transient;
namespace ts = transport::scalar;
namespace nt = numerics::time;

void native(const Exec& exec)
{
    Tpetra::ScopeGuard tpetra(nullptr, nullptr);
    const auto comm = mpi::CommView::world();
    const std::vector<linalg::GlobalOrdinal> gids{2*comm.rank(), 2*comm.rank()+1};
    const linalg::Distribution<Exec> distribution(gids, comm);
    const linalg::CsrInput graph{{0, 1, 2}, gids};
    const linalg::CsrPattern<Exec> pattern(distribution, graph, exec);
    const Entries entries{input<std::size_t>(exec, {0, 1})};
    const auto old = input<double>(exec, {2, 4});
    const auto history = tt::time_history(tt::TimeLevel{tt::UniformMass{3.0}, old});
    tt::TransientOperator<Exec, double> op(2, exec);
    ts::AssemblyStatus<Exec> status(exec);
    using NativeScalar = SolverReal;
    linalg::CsrMatrix<NativeScalar, Exec> csr(pattern);
    linalg::Vector<NativeScalar, Exec> csr_rhs(distribution);
    const auto cv = csr.edit_values(exec);
    const auto cr = Kokkos::subview(csr_rhs.edit(exec), Kokkos::ALL(), 0);
    Kokkos::deep_copy(exec, cv, 0.0); Kokkos::deep_copy(exec, cr, 0.0);
    status.begin_async();
    op.add_system_async(ts::SystemView{cv, cr, entries}, status, nt::BackwardEuler{}.coefficients(0.5), tt::UniformMass{2.0}, history);
    static_cast<void>(status.check(comm)); csr.commit(); csr_rhs.commit();
    const auto ca = host_copy(cv);
    const auto cb = host_copy(cr);
    for (std::size_t i = 0; i < 2; ++i) {
        test::equal(ca(i), 4.0, "Native CSR temporal diagonal");
        test::equal(cb(i), 12.0*static_cast<double>(i+1), "Native CSR temporal RHS");
    }
    linalg::BsrMatrix<NativeScalar, Exec> bsr(pattern, 3);
    linalg::Vector<NativeScalar, Exec> bsr_rhs(bsr.distribution());
    const auto blocks = bsr.edit_blocks(exec);
    const auto br = bsr_rhs.edit(exec);
    Kokkos::deep_copy(exec, blocks, 7.0); Kokkos::deep_copy(exec, br, 11.0);
    const auto bs = Kokkos::subview(blocks, Kokkos::ALL(), 1, 1);
    using Rhs = Kokkos::View<NativeScalar*, Kokkos::LayoutStride, typename decltype(br)::memory_space,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const Rhs rb(br.data()+br.stride(0), Kokkos::LayoutStride(2, 3*br.stride(0)));
    status.begin_async();
    op.add_system_async(ts::SystemView{bs, rb, entries}, status, nt::BackwardEuler{}.coefficients(0.5), tt::UniformMass{2.0}, history);
    static_cast<void>(status.check(comm)); bsr.commit(); bsr_rhs.commit();
    const auto ba = host_copy(blocks);
    const auto bb = host_copy(br);
    for (std::size_t i = 0; i < 2; ++i) {
        test::equal(ba(i, 1, 1), 11.0, "Native BSR selected diagonal");
        test::equal(ba(i, 0, 1), 7.0, "Native BSR cross component preserved");
        test::equal(bb(3*i+1, 0), 11.0+12.0*static_cast<double>(i+1), "Native BSR AoS RHS");
        test::equal(bb(3*i, 0), 11.0, "Native BSR other RHS preserved");
    }
}

void run_native(const Exec& exec)
{
    mpi::guarded([&] { test::complete(exec, [&] { native(exec); }); });
}
} // namespace
int main(int argc, char** argv)
{
    const std::array cases{test::Case{"native", run_native}};
    return krona::test::run(argc, argv, "transport.scalar.transient", cases);
}
