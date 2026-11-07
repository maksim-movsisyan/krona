#pragma once

#include <krona/linalg/types.hpp>
#include <utility>

namespace krona::linalg::detail {

/// @brief Wraps another scalar type; work vectors are cached, and apply is not reentrant.
template<class Scalar, class InnerScalar, class Exec>
class PrecisionOperator final : public Types<Scalar, Exec>::operator_type {
    using Outer = Types<Scalar, Exec>;
    using Inner = Types<InnerScalar, Exec>;
    using MV = typename Outer::multivector_type;
    using InnerMV = typename Inner::multivector_type;
    using Map = typename Outer::map_type;
public:
    explicit PrecisionOperator(Teuchos::RCP<const typename Inner::operator_type> inner)
        : inner_(std::move(inner))
    { mpi::require(!inner_.is_null(), "Linalg: null mixed-precision operator"); }

    Teuchos::RCP<const Map> getDomainMap() const override { return inner_->getDomainMap(); }
    Teuchos::RCP<const Map> getRangeMap() const override { return inner_->getRangeMap(); }
    bool hasTransposeApply() const override { return inner_->hasTransposeApply(); }

    void apply(const MV& x, MV& y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               Scalar alpha = Scalar{1}, Scalar beta = Scalar{0}) const override
    {
        mpi::require(x.getNumVectors() == y.getNumVectors(), "Linalg: mixed-precision multivector width mismatch");
        mpi::require(mode == Teuchos::NO_TRANS || hasTransposeApply(), "Linalg: preconditioner does not support transpose apply");
        if (alpha == Scalar{0}) {
            if (beta == Scalar{0}) y.putScalar(Scalar{0});
            else if (beta != Scalar{1}) y.scale(beta);
            return;
        }
        const auto domain = mode == Teuchos::NO_TRANS ? getDomainMap() : getRangeMap();
        const auto range = mode == Teuchos::NO_TRANS ? getRangeMap() : getDomainMap();
        if (x_.is_null() || width_ != x.getNumVectors() || transposed_ != (mode != Teuchos::NO_TRANS)) {
            width_ = x.getNumVectors();
            transposed_ = mode != Teuchos::NO_TRANS;
            x_ = Teuchos::rcp(new InnerMV(domain, width_, false));
            y_ = Teuchos::rcp(new InnerMV(range, width_, false));
            converted_ = Teuchos::null;
        }
        // Native conversion also handles nonconstant-stride multivector subviews.
        Tpetra::deep_copy(*x_, x);
        inner_->apply(*x_, *y_, mode, InnerScalar{1}, InnerScalar{0});
        if (beta == Scalar{0}) {
            Tpetra::deep_copy(y, *y_);
            if (alpha != Scalar{1}) y.scale(alpha);
        } else {
            // alpha/beta arithmetic stays in solver precision, and uses the actual y.
            if (converted_.is_null()) converted_ = Teuchos::rcp(new MV(range, width_, false));
            Tpetra::deep_copy(*converted_, *y_);
            y.update(alpha, *converted_, beta);
        }
    }

private:
    Teuchos::RCP<const typename Inner::operator_type> inner_;
    mutable Teuchos::RCP<InnerMV> x_, y_;
    mutable Teuchos::RCP<MV> converted_;
    mutable std::size_t width_ = 0;
    mutable bool transposed_ = false;
};

} // namespace krona::linalg::detail