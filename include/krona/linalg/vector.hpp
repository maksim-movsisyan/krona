#pragma once

#include <krona/linalg/distribution.hpp>
#include <krona/fields/transfer.hpp>

#include <utility>

namespace krona::linalg {

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
        if (pending_) { mpi::guarded([&] { pending_exec_.fence("Linalg: vector accesses committed"); }); pending_ = false; }
    }
    void require_committed() const
    { mpi::require(!pending_, "Linalg: vector has pending reads or writes; call commit()"); }


    void fill_async(Scalar value, const Exec& exec = {})
    { mpi::guarded([&] { Kokkos::deep_copy(exec, overwrite(exec), value); }); }

    /// @brief Blocking fill; waits only for the selected execution instance.
    void fill(Scalar value, const Exec& exec = {})
    { fill_async(value, exec); commit(); }

    /// @brief Copies an owned prefix asynchronously; component strides are respected.
    template<class Component>
    void copy_from_component_async(Component component, const Exec& exec = {})
    {
        static_assert(Kokkos::is_view<Component>::value && Component::rank == 1,
                      "Linalg: a component must be a rank-one Kokkos view");
        const auto target = overwrite(exec);
        fields::copy_to_dofs_async(exec, component, Kokkos::subview(target, Kokkos::ALL(), 0));
    }

    /// @brief Copies to an owned prefix asynchronously; ghost/boundary tails are untouched.
    template<class Component>
    void copy_to_component_async(Component component, const Exec& exec = {}) const
    {
        static_assert(Kokkos::is_view<Component>::value && Component::rank == 1,
                      "Linalg: a component must be a rank-one Kokkos view");
        const auto source = read(exec);
        fields::copy_from_dofs_async(exec, Kokkos::subview(source, Kokkos::ALL(), 0), component);
    }

    /// @brief Blocking copy; commits this vector's access batch on exec before returning.
    template<class Component>
    void copy_from_component(Component component, const Exec& exec = {})
    { copy_from_component_async(component, exec); commit(); }

    template<class Component>
    void copy_to_component(Component component, const Exec& exec = {}) const
    { copy_to_component_async(component, exec); commit(); }

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