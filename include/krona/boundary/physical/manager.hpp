#pragma once

#include <krona/boundary/layout.hpp>
#include <krona/boundary/physical/config.hpp>
#include <krona/boundary/physical/detail/kernels.hpp>
#include <krona/boundary/physical/detail/validation.hpp>

#include <utility>
#include <variant>
#include <vector>

namespace krona::boundary::physical {

/// @brief Owns model/conditions and borrows a shared BoundaryLayout. No field binding.
template<class Model,class Exec=execution::ExecutionSpace,class Real=ComputeReal,class... CustomConditions>
class Manager {
public:
    using memory_space=typename Exec::memory_space;
    using model_type=Model;
    using state_type=Primitive<Real,Model::thermal>;
    using layout_type=BoundaryLayout<Exec,Real>;
private:
    using policy_type=Kokkos::RangePolicy<Exec,Kokkos::IndexType<LocalIndex>>;
    using condition_type=std::variant<std::monostate,Symmetry,SlipWall<Real>,AdiabaticWall<Real>,
        IsothermalWall<Real>,HeatFluxWall<Real>,PressureOutlet<Real>,TotalConditionsInlet<Real>,
        VelocityInlet<Real>,SupersonicInlet<Real>,SupersonicOutlet,std::remove_cvref_t<CustomConditions>...>;
    static_assert(std::is_copy_constructible_v<condition_type>,"Physical boundary conditions/providers must be copy constructible");
public:
    Manager(const layout_type& layout,Model model) : layout_(&layout),model_(std::move(model))
    {
        execution::require_kokkos();
        mpi::guarded([&] { model_.validate(); conditions_.resize(catalog().patches().size()); });
    }
    Manager(const layout_type& layout,Model model,const Configuration& config)
        : Manager(layout,std::move(model)) { configure(config); }
    Manager(layout_type&&,Model)=delete;
    Manager(layout_type&&,Model,const Configuration&)=delete;
    Manager(const Manager&)=delete; Manager& operator=(const Manager&)=delete;
    Manager(Manager&&)=delete; Manager& operator=(Manager&&)=delete;

    template<class C>
    void set(mesh::PatchId id,C condition)
    {
        static_assert(std::is_constructible_v<condition_type,C>,"Register the condition type in Manager");
        mpi::require(id>=0&&static_cast<std::size_t>(id)<conditions_.size(),"Physical boundary: unknown patch ID");
        mpi::guarded([&] {
            auto prepared=detail::prepare(std::move(condition),model_);
            validate_metadata(prepared,catalog().patches()[static_cast<std::size_t>(id)].range);
            conditions_[static_cast<std::size_t>(id)].template emplace<C>(std::move(prepared));
            finalized_=false;
        });
    }
    template<class C>
    void set_unlabelled(C condition)
    {
        static_assert(std::is_constructible_v<condition_type,C>);
        mpi::guarded([&] {
            auto prepared=detail::prepare(std::move(condition),model_);
            validate_metadata(prepared,catalog().unlabelled());
            unlabelled_.template emplace<C>(std::move(prepared)); finalized_=false;
        });
    }
    void configure(const Configuration& config)
    {
        mpi::guarded([&] {
            mpi::require(config.patches.size()==conditions_.size(),"Physical boundary: configuration must describe every mesh patch");
            std::vector<bool> seen(conditions_.size(),false);
            for (const auto& p:config.patches) {
                mpi::require(p.patch_id>=0&&static_cast<std::uint64_t>(p.patch_id)<conditions_.size(),
                    "Physical boundary: configuration patch ID is out of range");
                const auto id=static_cast<std::size_t>(p.patch_id);
                mpi::require(!seen[id],"Physical boundary: duplicate configuration patch ID");
                mpi::require(p.name==catalog().patches()[id].name,"Physical boundary: patch name does not match the mesh");
                seen[id]=true;
                std::visit([&](const auto& c) { set(static_cast<mesh::PatchId>(id),detail::condition_cast<Real>(c)); },p.condition);
            }
            unlabelled_.template emplace<std::monostate>();
            if (config.unlabelled) std::visit([&](const auto& c) { set_unlabelled(detail::condition_cast<Real>(c)); },*config.unlabelled);
            finalized_=false;
        });
    }
    /// @brief Host completeness/provider checks and cached requirements. No work submission or MPI.
    void finalize()
    {
        mpi::guarded([&] {
            require_complete(); needs_law_=false;
            const auto inspect=[&](PatchRange range,const condition_type& condition) {
                std::visit([&](const auto& c) {
                    using C=std::remove_cvref_t<decltype(c)>;
                    if constexpr (!std::is_same_v<C,std::monostate>) {
                        validate_metadata(c,range); needs_law_=needs_law_||requires_flux_law<C>;
                    }
                },condition);
            };
            for (std::size_t p=0;p<conditions_.size();++p) inspect(catalog().patches()[p].range,conditions_[p]);
            inspect(catalog().unlabelled(),unlabelled_); finalized_=true;
        });
    }
    [[nodiscard]] bool finalized() const noexcept { return finalized_; }

    template<class Visitor>
    void for_each_patch(Visitor&& visitor) const
    {
        require_finalized();
        mpi::guarded([&] {
            for (std::size_t p=0;p<conditions_.size();++p) visit(catalog().patches()[p].range,conditions_[p],visitor);
            visit(catalog().unlabelled(),unlabelled_,visitor);
        });
    }
    template<class Sink>
    void for_each_face_async(Sink sink,Real time=0) const
    {
        mpi::require(std::isfinite(time),"Physical boundary: time must be finite");
        for_each_patch([&](PatchRange range,const auto& condition) {
            using C=std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("boundary/physical/for_each_face",policy_type(execution_space(),range.begin,range.end),
                boundary::detail::ForEachFace<memory_space,Real,Sink,C>{geometry_view(),sink,condition,time});
        });
    }

    /// @brief Reader(context,condition) supplies the interior; sink(context,evaluation) consumes both traces.
    template<class Sink,class Reader,class Law=scalar::NoFluxLaw,class Correction=Corrections<>>
    void evaluate_async(Sink sink,Reader reader,Status<Exec>& status,Law law={},Correction correction={},Real time=0) const
    {
        validate_inputs<Law>(correction);
        mpi::guarded([&] {
            if constexpr (requires { reader.template validate<Exec>(layout_->owned_cells()); })
                reader.template validate<Exec>(layout_->owned_cells());
        });
        submit(detail::CheckedSink<Sink>{sink},reader,status,law,correction,time);
    }
    /// @brief Writes only primitive physical-face slots, never stored exterior/conservative states.
    template<class Fields,class Law=scalar::NoFluxLaw,class Correction=Corrections<>>
    void apply_async(Fields fields,Status<Exec>& status,Law law={},Correction correction={},Real time=0) const
    {
        static_assert(Fields::thermal==Model::thermal,"Primitive binding must match the model's thermal capability");
        mpi::guarded([&] { fields.template validate<Exec,true>(required_field_size()); });
        validate_inputs<Law>(correction);
        submit(detail::Fill<Fields>{fields},OwnerReader{fields},status,law,correction,time);
    }

    [[nodiscard]] const Model& model() const noexcept { return model_; }
    [[nodiscard]] const Exec& execution_space() const noexcept { return layout_->execution_space_instance(); }
    [[nodiscard]] const PatchCatalog& catalog() const noexcept { return layout_->catalog(); }
    [[nodiscard]] GeometryView<memory_space> geometry_view() const noexcept { return layout_->geometry_view(); }
    [[nodiscard]] std::size_t required_field_size() const noexcept { return layout_->required_field_size(); }
    [[nodiscard]] const layout_type& layout() const noexcept { return *layout_; }
private:
    const layout_type* layout_;
    Model model_;
    std::vector<condition_type> conditions_;
    condition_type unlabelled_;
    bool finalized_=false,needs_law_=false;

    template<class C> static void validate_metadata(const C& c,PatchRange range)
    { profile::validate<Exec>(c,range); }
    void require_finalized() const
    { mpi::require(finalized_,"Physical boundary: call finalize() after assigning conditions"); }
    void require_complete() const
    {
        for (const auto& c:conditions_) mpi::require(!std::holds_alternative<std::monostate>(c),
            "Physical boundary: missing condition on a labelled patch");
        mpi::require(!catalog().needs_unlabelled()||!std::holds_alternative<std::monostate>(unlabelled_),
            "Physical boundary: unlabelled faces need an explicit condition");
    }
    template<class Visitor>
    static void visit(PatchRange range,const condition_type& condition,Visitor& visitor)
    {
        if (range.empty()) return;
        std::visit([&](const auto& c) {
            if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(c)>,std::monostate>) visitor(range,c);
        },condition);
    }
    template<class C> void validate_correction(const C& c) const
    {
        if constexpr (requires { c.gradients.template validate<Exec>(layout_->owned_cells()); })
            c.gradients.template validate<Exec>(layout_->owned_cells());
        else if constexpr (requires { c.template validate<Exec>(layout_->owned_cells()); })
            c.template validate<Exec>(layout_->owned_cells());
    }
    template<class X,class Y,class Z>
    void validate_correction(const VectorCorrection<X,Y,Z>& c) const
    { validate_correction(c.x); validate_correction(c.y); validate_correction(c.z); }
    template<class Law,class C>
    void validate_inputs(const C& correction) const
    {
        require_finalized();
        if constexpr (!scalar::FluxLaw<Law,Real>)
            mpi::require(!needs_law_,"Physical boundary: heat-flux wall requires an explicit flux law");
        mpi::guarded([&] {
            validate_correction(correction.pressure); validate_correction(correction.velocity);
            if constexpr (Model::thermal) validate_correction(correction.temperature);
        });
    }
    template<class Sink,class Reader,class Law,class Correction>
    void submit(Sink sink,Reader reader,Status<Exec>& status,Law law,Correction correction,Real time) const
    {
        using Flags=typename Status<Exec>::flags_type;
        for_each_face_async(detail::Evaluate<Model,Reader,Sink,Law,Correction,Flags>{
            model_,reader,sink,law,correction,status.device_flags(execution_space())},time);
    }
};

} // namespace krona::boundary::physical
