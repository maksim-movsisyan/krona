#pragma once

#include <krona/linalg/distribution.hpp>

#include <utility>

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
    using scalar_type = Scalar;
    using execution_space = Exec;
    using memory_space = typename Types<Scalar, Exec>::memory_space;
    using native_type = typename Types<Scalar, Exec>::vector_type;
    
    explicit Vector(const Distribution<Exec>& distribution) : distribution_(distribution)
    {
        mpi::guarded([&] {
            vector_ = Teuchos::rcp(new native_type(distribution_.native(), true));
        });
    }
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;
    ~Vector() { commit(); }

    [[nodiscard]] std::size_t local_size() const noexcept { return distribution_.local_size(); }
    [[nodiscard]] const Distribution<Exec>& distribution() const noexcept { return distribution_; }

    /// @brief Joins an access batch on exec; commit before native operations or another instance.
    [[nodiscard]] auto edit(const Exec& exec = {})
    {
        begin_access(exec);
        return mpi::guarded([&] { return vector_->getLocalViewDevice(Tpetra::Access::ReadWrite); });
    }
    /// @brief Joins an asynchronous read/write batch on the same execution instance.
    [[nodiscard]] auto read(const Exec& exec = {}) const
    {
        begin_access(exec);
        return mpi::guarded([&] { return std::as_const(*vector_).getLocalViewDevice(Tpetra::Access::ReadOnly); });
    }
    /// @brief Full overwrite; old host data need not be copied to device.
    [[nodiscard]] auto overwrite(const Exec& exec = {})
    {
        begin_access(exec);
        return mpi::guarded([&] { return vector_->getLocalViewDevice(Tpetra::Access::OverwriteAll); });
    }
    /// @brief Completes the custom execution-space read/write batch.
    void commit() const
    {
        if (pending_) { pending_exec_.fence("Linalg: vector accesses committed"); pending_ = false; }
    }
    void require_committed() const
    { mpi::require(!pending_, "Linalg: vector has pending reads or writes; call commit()"); }


    void fill_async(Scalar value, const Exec& exec = {})
    { Kokkos::deep_copy(exec, overwrite(exec), value); }

    /// @brief Blocking fill; waits only for the selected execution instance.
    void fill(Scalar value, const Exec& exec = {})
    { fill_async(value, exec); commit(); }

    /// @brief Copies an owned prefix asynchronously; component strides are respected.
    template<class Component>
    void copy_from_component(Component component, const Exec& exec = {})
    {
        static_assert(Component::rank == 1);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Component::memory_space>::accessible);
        mpi::require(component.extent(0) >= local_size(), "Linalg: component is shorter than the owned vector");
        auto target = overwrite(exec);
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
        mpi::require(component.extent(0) >= local_size(), "Linalg: component is shorter than the owned vector");
        auto source = read(exec);
        using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
        Kokkos::parallel_for("linalg/vector_to_component", Policy(exec, 0, local_size()),
            detail::ComponentCopy<decltype(source), Component, false>{source, component});
    }

    /// @brief Advanced interoperability; native access bypasses wrapper state checks.
    [[nodiscard]] const Teuchos::RCP<native_type>& native() const noexcept { return vector_; }
    

private:
    Distribution<Exec> distribution_;
    Teuchos::RCP<native_type> vector_;
    mutable Exec pending_exec_{};
    mutable bool pending_ = false;

    void begin_access(const Exec& exec) const
    {
        mpi::require(!pending_ || pending_exec_ == exec,
                     "Linalg: commit before switching execution instances");
        pending_exec_ = exec; pending_ = true;
    }
};

} // namespace krona::linalg
