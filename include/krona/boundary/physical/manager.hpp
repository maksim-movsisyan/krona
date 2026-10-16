#pragma once

#include <krona/boundary/physical/config.hpp>
#include <krona/boundary/physical/detail/kernels.hpp>
#include <krona/boundary/physical/detail/validation.hpp>
#include <krona/boundary/detail/kernels.hpp>
#include <krona/boundary/patches.hpp>
#include <krona/execution/utils.hpp>
#include <limits>
#include <utility>
#include <variant>
#include <vector>

namespace krona::boundary::physical {

/// Typed host patch dispatch; model parameters are explicit, small copied device values.
template<class Model,class Exec=execution::ExecutionSpace,class Real=ComputeReal,class... CustomConditions>
class Manager {
public:
    using memory_space=typename Exec::memory_space;
    using model_type=Model;
    using state_type=Primitive<Real,Model::thermal>;
    using mesh_view_type=mesh::compute::MeshView<memory_space>;
private:
    using policy_type=Kokkos::RangePolicy<Exec,Kokkos::IndexType<LocalIndex>>;
    using condition_type=std::variant<std::monostate,Symmetry,SlipWall<Real>,AdiabaticWall<Real>,
        IsothermalWall<Real>,HeatFluxWall<Real>,PressureOutlet<Real>,TotalConditionsInlet<Real>,
        VelocityInlet<Real>,SupersonicInlet<Real>,SupersonicOutlet,CustomConditions...>;
public:
    Manager(const mesh::MeshPart& host,mesh_view_type mesh_view,Model model,Exec exec)
        : catalog_(host),exec_(exec),model_(std::move(model))
    {
        execution::require_kokkos(); model_.validate();
        const auto& s=mesh_view.sizes;
        mpi::require(s.faces_grouped && s.n_own==host.n_own && s.n_cells==host.n_cells &&
            s.n_faces==host.n_faces && s.n_inner_faces==host.n_inner_faces,
            "Physical boundary: host/compute mesh counts differ");
        const auto cells=static_cast<std::size_t>(s.n_cells);
        const auto faces=static_cast<std::size_t>(s.n_faces-s.n_inner_faces);
        mpi::require(cells<=std::numeric_limits<std::size_t>::max()-faces,"Physical boundary: extended size overflow");
        required_size_=cells+faces; owned_=static_cast<std::size_t>(s.n_own);
        mpi::guarded([&] {
            conditions_.resize(catalog_.patches().size());
            execution::complete(exec_,[&] { geometry_=boundary::detail::attach_geometry<Real>(exec_,mesh_view); });
        });
    }
    Manager(const mesh::MeshPart& host,mesh_view_type view,Model model,const Configuration& config,Exec exec)
        : Manager(host,view,std::move(model),exec) { configure(config); }
    Manager(const Manager&)=delete; Manager& operator=(const Manager&)=delete;
    Manager(Manager&&)=delete; Manager& operator=(Manager&&)=delete;

    template<class C>
    void set(mesh::PatchId id,C condition)
    {
        static_assert(std::is_constructible_v<condition_type,C>,"Register the custom condition in Manager");
        mpi::require(id>=0&&static_cast<std::size_t>(id)<conditions_.size(),"Physical boundary: unknown patch ID");
        condition=detail::prepare(condition,model_);
        mpi::guarded([&] { conditions_[static_cast<std::size_t>(id)]=std::move(condition); });
    }
    template<class C>
    void set_unlabelled(C condition)
    {
        static_assert(std::is_constructible_v<condition_type,C>);
        condition=detail::prepare(condition,model_);
        mpi::guarded([&] { unlabelled_=std::move(condition); });
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
                mpi::require(p.name==catalog_.patches()[id].name,"Physical boundary: patch name does not match the mesh");
                seen[id]=true;
                std::visit([&](const auto& c) { set(static_cast<mesh::PatchId>(id),detail::condition_cast<Real>(c)); },p.condition);
            }
            unlabelled_=std::monostate{};
            if (config.unlabelled) std::visit([&](const auto& c) { set_unlabelled(detail::condition_cast<Real>(c)); },*config.unlabelled);
            require_complete();
        });
    }

    template<class Visitor>
    void for_each_patch(Visitor&& visitor) const
    {
        require_complete();
        for (std::size_t p=0;p<conditions_.size();++p) visit(catalog_.patches()[p].range,conditions_[p],visitor);
        visit(catalog_.unlabelled(),unlabelled_,visitor);
    }
    template<class Sink>
    void for_each_face_async(Sink sink,Real time=0) const
    {
        mpi::require(std::isfinite(time),"Physical boundary: time must be finite");
        for_each_patch([&](PatchRange range,const auto& condition) {
            using C=std::remove_cvref_t<decltype(condition)>;
            Kokkos::parallel_for("boundary/physical/for_each_face",policy_type(exec_,range.begin,range.end),
                boundary::detail::ForEachFace<memory_space,Real,Sink,C>{geometry_,sink,condition,time});
        });
    }

    /// Reader(context,typed_condition) supplies the interior state; sink(context,evaluation) consumes both traces.
    template<class Sink,class Reader,class Law=scalar::NoFluxLaw,class Correction=Corrections<>>
    void evaluate_async(Sink sink,Reader reader,Law law={},Correction correction={},Real time=0) const
    {
        validate_inputs<Law>(correction);
        if constexpr (requires { reader.template validate<Exec>(owned_); }) reader.template validate<Exec>(owned_);
        for_each_face_async(detail::Evaluate<Model,Reader,Sink,Law,Correction>{model_,reader,sink,law,correction},time);
    }
    /// Writes primitive physical-face slots only. It does not store exterior/conservative states.
    template<class Fields,class Law=scalar::NoFluxLaw,class Correction=Corrections<>>
    void apply_async(Fields fields,Law law={},Correction correction={},Real time=0) const
    {
        static_assert(Fields::thermal==Model::thermal,"Primitive field binding must match the model's thermal capability");
        fields.template validate<Exec,true>(required_size_);
        evaluate_async(detail::Fill<Fields>{fields},OwnerReader{fields},law,correction,time);
    }

    [[nodiscard]] const Model& model() const noexcept { return model_; }
    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    [[nodiscard]] const PatchCatalog& catalog() const noexcept { return catalog_; }
    [[nodiscard]] GeometryView<memory_space> geometry_view() const noexcept { return geometry_; }
    [[nodiscard]] std::size_t required_field_size() const noexcept { return required_size_; }

private:
    PatchCatalog catalog_;
    Exec exec_;
    Model model_;
    GeometryView<memory_space> geometry_;
    std::vector<condition_type> conditions_;
    condition_type unlabelled_;
    std::size_t required_size_=0,owned_=0;

    void require_complete() const
    {
        for (const auto& c:conditions_) mpi::require(!std::holds_alternative<std::monostate>(c),
            "Physical boundary: missing condition on a labelled patch");
        mpi::require(!catalog_.needs_unlabelled()||!std::holds_alternative<std::monostate>(unlabelled_),
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
    template<class C> void validate_correction(const C&) const {}
    template<class X,class Y,class Z>
    void validate_correction(boundary::GradientCorrection<X,Y,Z> c) const
    {
        detail::validate_component<Exec,false>(c.x,owned_);
        detail::validate_component<Exec,false>(c.y,owned_);
        detail::validate_component<Exec,false>(c.z,owned_);
    }
    template<class X,class Y,class Z>
    void validate_correction(VectorCorrection<X,Y,Z> c) const
    { validate_correction(c.x); validate_correction(c.y); validate_correction(c.z); }
    template<class Law,class C>
    void validate_inputs(const C& correction) const
    {
        if constexpr (!scalar::FluxLaw<Law,Real>) {
            bool requires_law=std::holds_alternative<HeatFluxWall<Real>>(unlabelled_);
            for (const auto& c:conditions_) requires_law=requires_law||std::holds_alternative<HeatFluxWall<Real>>(c);
            mpi::require(!requires_law,"Physical boundary: heat-flux wall requires an explicit flux law");
        }
        validate_correction(correction.pressure); validate_correction(correction.velocity);
        if constexpr (Model::thermal) validate_correction(correction.temperature);
    }
};

} // namespace krona::boundary::physical
