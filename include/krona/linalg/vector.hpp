#pragma once

#include <krona/linalg/distribution.hpp>

namespace krona::linalg {

namespace detail {
template<class Source, class Destination, bool ToVector>
struct ComponentCopy {
    Source source;
    Destination destination;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const
    {
        if constexpr (ToVector) destination(i, 0) = static_cast<typename Destination::non_const_value_type>(source(i));
        else destination(i) = static_cast<typename Destination::non_const_value_type>(source(i, 0));
    }
};
} // namespace detail

/// @brief Owned unknowns only; Tpetra imports remote values for its matrix operations.
template<class Scalar = SolverReal, class Exec = execution::ExecutionSpace>
class Vector {
public:
    using native_type = typename Types<Scalar, Exec>::vector_type;
    explicit Vector(const Distribution<Exec>& distribution) : distribution_(distribution)
    {
        detail::guarded([&] {
            vector_ = Teuchos::rcp(new native_type(distribution_.native(), true));
            Kokkos::fence("Linalg: vector initialization complete");
        });
    }
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;

    [[nodiscard]] std::size_t local_size() const noexcept { return distribution_.local_size(); }
    [[nodiscard]] const Distribution<Exec>& distribution() const noexcept { return distribution_; }

    /// @brief Begins an asynchronous write batch; commit before entering Trilinos.
    [[nodiscard]] auto edit(const Exec& exec = {})
    {
        require_committed();
        pending_exec_ = exec; pending_ = true;
        return detail::guarded([&] { return vector_->getLocalViewDevice(Tpetra::Access::ReadWrite); });
    }
    [[nodiscard]] auto read() const
    {
        require_committed();
        return detail::guarded([&] { return std::as_const(*vector_).getLocalViewDevice(Tpetra::Access::ReadOnly); });
    }
    /// @brief Completes the custom execution-space write batch.
    void commit()
    {
        if (pending_) { pending_exec_.fence("Linalg: vector writes committed"); pending_ = false; }
    }
    void require_committed() const
    { detail::require(!pending_, "Linalg: vector has pending writes; call commit()"); }

    /// @brief Blocking native operation; accepts a nonzero initial guess.
    void fill(Scalar value)
    {
        require_committed();
        detail::guarded([&] { vector_->putScalar(value); Kokkos::fence("Linalg: vector fill complete"); });
    }

    /// @brief Copies an owned prefix asynchronously; component strides are respected.
    template<class Component>
    void copy_from_component(Component component, const Exec& exec = {})
    {
        static_assert(Component::rank == 1);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Component::memory_space>::accessible);
        detail::require(component.extent(0) >= local_size(), "Linalg: component is shorter than the owned vector");
        auto target = edit(exec);
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        Kokkos::parallel_for("linalg/component_to_vector", Policy(exec, 0, local_size()),
            detail::ComponentCopy<Component, decltype(target), true>{component, target});
    }

    /// @brief Copies to an owned prefix asynchronously; ghost/boundary tails are untouched.
    template<class Component>
    void copy_to_component(Component component, const Exec& exec = {}) const
    {
        static_assert(Component::rank == 1 && !std::is_const_v<typename Component::value_type>);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Component::memory_space>::accessible);
        detail::require(component.extent(0) >= local_size(), "Linalg: component is shorter than the owned vector");
        auto source = read();
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        Kokkos::parallel_for("linalg/vector_to_component", Policy(exec, 0, local_size()),
            detail::ComponentCopy<decltype(source), Component, false>{source, component});
    }

    /// @brief Advanced interoperability; native access bypasses wrapper state checks.
    [[nodiscard]] const Teuchos::RCP<native_type>& native() const noexcept { return vector_; }

private:
    Distribution<Exec> distribution_;
    Teuchos::RCP<native_type> vector_;
    Exec pending_exec_{};
    bool pending_ = false;
};

} // namespace krona::linalg
