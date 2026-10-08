#pragma once

#include <krona/fields/fields_view.hpp>
#include <krona/halo/detail/types.hpp>
#include <krona/mesh/compute/halo_plan.hpp>
#include <krona/mesh/compute/mesh_view.hpp>
#include <krona/mpi/context.hpp>

#include <algorithm>
#include <exception>
#include <limits>
#include <vector>

namespace krona::halo {

namespace detail {

// Named functors avoid NVCC's access and first-capture restrictions on extended lambdas.
template<bool Packing, bool UnitStride, class ComponentView, class MapView,
         class PeerView, class SegmentView, class BufferView>
struct TransferKernel {
    ComponentView fields;
    MapView map;
    PeerView peers;
    SegmentView segments;
    BufferView buffer;
    std::size_t width;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t c, std::size_t i) const
    {
        const auto field = fields(c);
        const auto segment = segments(peers(i));
        const std::size_t cell = static_cast<std::size_t>(map(i));
        const auto offset = segment.offset * width + c * segment.count + (i - segment.offset);
        const auto field_index = UnitStride ? cell : cell * field.stride;
        if constexpr (Packing) buffer(offset) = field.data[field_index];
        else field.data[field_index] = buffer(offset);
    }
};

} // namespace detail

/// @brief Staging is portable; DeviceDirect requires MPI support for the buffer memory space.
enum class Transfer { HostStaged, DeviceDirect };

/// @brief Aggregates arbitrary cell-field components into one SoA message per peer.
/// Register once, prepare once, then reuse without application-side allocations.
/// Mesh maps, fields, communicator and runtimes must outlive this object.
template<class T = StorageReal, class Exec = execution::ExecutionSpace>
class Exchange {
    static_assert(!std::is_const_v<T>);
    using Memory = typename Exec::memory_space;
    using Device = Kokkos::Device<Exec, Memory>;
    using Buffer = Kokkos::View<T*, Kokkos::LayoutRight, Device>;
    using HostBuffer = Kokkos::View<T*, Kokkos::LayoutRight, typename detail::StagingSpace<Exec>::type>;
    using Components = Kokkos::View<detail::Component<T>*, Device>;
    using Segments = Kokkos::View<detail::Segment*, Device>;
    using Peers = Kokkos::View<int*, Device>;
    using Map = mesh::compute::detail::MeshArray<LocalIndex, Memory, false>;
    using Policy = Kokkos::MDRangePolicy<Exec,
        Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
        Kokkos::IndexType<std::size_t>>;
    static constexpr bool host_memory = std::is_same_v<Memory, Kokkos::HostSpace>;

    enum class State { Idle, Communicating, Unpacking };
    struct Peer {
        int rank = 0;
        detail::Segment send, recv;
        int send_values = 0, recv_values = 0;
    };

public:
    Exchange(mesh::compute::MeshView<Memory> mesh, mesh::compute::HaloPlanView plan,
             MPI_Comm comm, int tag, Exec exec, Transfer transfer = Transfer::HostStaged)
        : exec_(exec), comm_(comm), tag_(tag), transfer_(transfer),
          n_own_(mesh.sizes.n_own), n_cells_(mesh.sizes.n_cells),
          send_map_(mesh.halo.send_owned_local), recv_map_(mesh.halo.recv_ghost_local)
    {
        detail::require(Kokkos::is_initialized() && !Kokkos::is_finalized(), "Halo requires active Kokkos");
        detail::require(mpi::is_active(), "Halo requires active MPI and a valid communicator");
        detail::require(mesh.has(mesh::compute::MeshData::Halo), "Halo maps were not selected");
        detail::require(n_own_ >= 0 && n_cells_ >= n_own_, "Halo: invalid cell counts");
        detail::require(transfer == Transfer::HostStaged || transfer == Transfer::DeviceDirect,
                        "Halo: invalid transfer mode");

        int inter = 0, size = 0, flag = 0;
        int* tag_limit = nullptr;
        mpi::check(MPI_Comm_test_inter(comm_, &inter), "MPI_Comm_test_inter");
        detail::require(inter == 0, "Halo supports intracommunicators only");
        mpi::check(MPI_Comm_size(comm_, &size), "MPI_Comm_size");
        mpi::check(MPI_Comm_get_attr(comm_, MPI_TAG_UB, &tag_limit, &flag), "MPI_Comm_get_attr");
        // In the MPI world model, predefined environmental attributes are guaranteed on WORLD.
        if (flag == 0)
            mpi::check(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_limit, &flag),
                       "MPI_Comm_get_attr(MPI_COMM_WORLD)");
        detail::require(flag != 0 && tag_limit != nullptr && tag >= 0 && tag <= *tag_limit,
                        "Halo: tag is outside the communicator's valid range");
        const auto count = plan.n_neighbors();
        detail::require(count <= static_cast<std::size_t>(std::numeric_limits<int>::max() / 2),
                        "Halo: too many MPI requests");
        validate_offsets(plan.send_offsets, count, send_map_.extent(0));
        validate_offsets(plan.recv_offsets, count, recv_map_.extent(0));
        peers_.reserve(count);
        std::vector<int> ranks;
        ranks.reserve(count);
        for (std::size_t p = 0; p < count; ++p) {
            const int rank = plan.nb_ranks[p];
            detail::require(rank >= 0 && rank < size, "Halo: peer rank is outside communicator");
            ranks.push_back(rank);
            peers_.push_back({rank, {plan.send_offsets[p], plan.send_count(p)},
                                   {plan.recv_offsets[p], plan.recv_count(p)}, 0, 0});
        }
        std::sort(ranks.begin(), ranks.end());
        detail::require(std::adjacent_find(ranks.begin(), ranks.end()) == ranks.end(), "Halo: duplicate peer rank");
        datatype_ = detail::datatype<T>();
    }

    Exchange(const Exchange&) = delete;
    Exchange& operator=(const Exchange&) = delete;
    Exchange(Exchange&&) = delete;
    Exchange& operator=(Exchange&&) = delete;

    /// @brief Drains outstanding MPI and unpack work before releasing communication buffers.
    ~Exchange() { if (state_ != State::Idle) wait(); }

    /// @brief Registers all components in field order; no names or physics enter kernels.
    template<int N, class Space, class Layout>
    void add_field(fields::FieldView<T, N, Space, Layout> field)
    {
        static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible,
                      "Execution space cannot access this field");
        detail::require(!prepared_, "Halo: registration is closed after prepare");
        detail::require(field.size() >= static_cast<std::size_t>(n_cells_),
                        "Halo fields must contain all owned and ghost cells");
        for (int c = 0; c < N; ++c) add_component(field.component(c));
    }

    /// @brief Registers one mutable rank-one Kokkos View, including a strided component subview.
    template<class View>
    void add_component(View component)
    {
        static_assert(Kokkos::is_view<View>::value && View::rank == 1);
        static_assert(std::is_same_v<typename View::value_type, T>);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);
        detail::require(!prepared_, "Halo: registration is closed after prepare");
        detail::require(component.extent(0) >= static_cast<std::size_t>(n_cells_),
                        "Halo components must contain all owned and ghost cells");
        const detail::Component<T> descriptor{component.data(), component.stride(0)};
        if (component.extent(0) != 0) {
            detail::require(descriptor.data != nullptr && descriptor.stride != 0,
                            "Halo: component requires valid storage and a positive stride");
            for (const auto& previous : registered_)
                detail::require(previous.data != descriptor.data, "Halo: duplicate field component");
        }
        registered_.push_back(descriptor);
        unit_stride_ = unit_stride_ && descriptor.stride == 1;
    }

    /// @brief Allocates reusable buffers and validates maps; local, not collective.
    void prepare()
    {
        detail::require(!prepared_, "Halo: prepare was already called");
        detail::require(!registered_.empty(), "Halo: register at least one field");
        const auto width = registered_.size();
        const auto send_values = detail::product(send_map_.extent(0), width);
        const auto recv_values = detail::product(recv_map_.extent(0), width);
        detail::product(send_values, sizeof(T));
        detail::product(recv_values, sizeof(T));
        for (auto& peer : peers_) {
            const auto send = detail::product(peer.send.count, width);
            const auto recv = detail::product(peer.recv.count, width);
            const std::size_t limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
            detail::require(send <= limit && recv <= limit,
                            "Halo: peer payload exceeds MPI-3 int count; split the group");
            peer.send_values = static_cast<int>(send);
            peer.recv_values = static_cast<int>(recv);
        }
        // Validate once on host. CPU mirrors alias the maps; GPU mirrors are temporary copies.
        auto send_host = [&] {
            if constexpr (host_memory) return send_map_;
            else return Kokkos::create_mirror(Kokkos::WithoutInitializing, send_map_);
        }();
        auto recv_host = [&] {
            if constexpr (host_memory) return recv_map_;
            else return Kokkos::create_mirror(Kokkos::WithoutInitializing, recv_map_);
        }();
        detail::complete(exec_, [&] {
            if constexpr (!host_memory) {
                Kokkos::deep_copy(exec_, send_host, send_map_);
                Kokkos::deep_copy(exec_, recv_host, recv_map_);
            }
        });
        for (std::size_t i = 0; i < send_host.extent(0); ++i)
            detail::require(send_host(i) >= 0 && send_host(i) < n_own_,
                            "Halo: send map must reference owned cells");
        std::vector<LocalIndex> destinations;
        destinations.reserve(recv_host.extent(0));
        for (std::size_t i = 0; i < recv_host.extent(0); ++i) {
            detail::require(recv_host(i) >= n_own_ && recv_host(i) < n_cells_,
                            "Halo: receive map must reference ghost cells");
            destinations.push_back(recv_host(i));
        }
        std::sort(destinations.begin(), destinations.end());
        detail::require(std::adjacent_find(destinations.begin(), destinations.end()) == destinations.end(),
                        "Halo: receive map contains duplicate ghost cells");

        send_ = Buffer(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "halo/send"), send_values);
        recv_ = Buffer(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "halo/recv"), recv_values);
        if constexpr (!host_memory) {
            if (transfer_ == Transfer::HostStaged) {
                send_host_ = HostBuffer(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                       "halo/send_host"), send_values);
                recv_host_ = HostBuffer(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                       "halo/recv_host"), recv_values);
            }
        }
        components_ = Components(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing,
                                                   "halo/components"), width);
        auto components_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, components_);
        for (std::size_t c = 0; c < width; ++c) components_host(c) = registered_[c];
        detail::complete(exec_, [&] {
            Kokkos::deep_copy(exec_, components_, components_host);
            make_segments(send_segments_, send_peers_, true);
            make_segments(recv_segments_, recv_peers_, false);
            requests_.assign(peers_.size() * 2, MPI_REQUEST_NULL);
        });
        prepared_ = true;
    }

    /// @brief Posts receives, packs, then posts sends. Waits only for send-buffer readiness.
    /// Producers on another execution-space instance must be synchronized by the caller.
    void start()
    {
        detail::require(prepared_, "Halo: prepare must precede start");
        detail::require(state_ != State::Communicating, "Halo: exchange is already active");
        if (state_ == State::Unpacking) wait(); // Previous unpack still owns the receive buffer.
        state_ = State::Communicating;
        try {
            for (std::size_t p = 0; p < peers_.size(); ++p) {
                const auto& peer = peers_[p];
                if (peer.recv_values == 0) continue;
                mpi::check(MPI_Irecv(recv_pointer() + peer.recv.offset * registered_.size(),
                    peer.recv_values, datatype_, peer.rank, tag_, comm_, &requests_[p]), "MPI_Irecv");
            }
            if (send_.extent(0) != 0) {
                if (unit_stride_) pack<true>(); else pack<false>();
                if constexpr (!host_memory)
                    if (transfer_ == Transfer::HostStaged) Kokkos::deep_copy(exec_, send_host_, send_);
                // Standard MPI has no execution-space/stream dependency mechanism.
                exec_.fence("KRONA halo: send buffer ready for MPI");
            }
            for (std::size_t p = 0; p < peers_.size(); ++p) {
                const auto& peer = peers_[p];
                if (peer.send_values == 0) continue;
                mpi::check(MPI_Isend(send_pointer() + peer.send.offset * registered_.size(),
                    peer.send_values, datatype_, peer.rank, tag_, comm_,
                    &requests_[peers_.size() + p]), "MPI_Isend");
            }
        } catch (const std::exception& error) { mpi::fatal(error.what()); }
          catch (...) { mpi::fatal("Halo: failed to start exchange"); }
    }

    /// @brief Waits for MPI, then queues receive upload and ghost scatter on exec.
    /// Subsequent kernels on this same instance observe updated ghosts in order.
    void finish_async()
    {
        detail::require(state_ == State::Communicating, "Halo: no active MPI exchange");
        try {
            if (!requests_.empty())
                mpi::check(MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(),
                                       MPI_STATUSES_IGNORE), "MPI_Waitall");
            if (recv_.extent(0) != 0) {
                state_ = State::Unpacking;
                if constexpr (!host_memory)
                    if (transfer_ == Transfer::HostStaged) Kokkos::deep_copy(exec_, recv_, recv_host_);
                if (unit_stride_) unpack<true>(); else unpack<false>();
            } else state_ = State::Idle;
        } catch (const std::exception& error) { mpi::fatal(error.what()); }
          catch (...) { mpi::fatal("Halo: failed to finish exchange"); }
    }

    /// @brief Completes outstanding communication and makes ghost values ready for external users.
    void wait()
    {
        if (state_ == State::Communicating) finish_async();
        if (state_ == State::Unpacking) {
            try { exec_.fence("KRONA halo: ghost values ready"); }
            catch (const std::exception& error) { mpi::fatal(error.what()); }
            catch (...) { mpi::fatal("Halo: failed to synchronize unpack"); }
            state_ = State::Idle;
        }
    }

    /// @brief Blocking convenience path using the same reusable communication buffers.
    void exchange() { start(); wait(); }

    [[nodiscard]] std::size_t component_count() const noexcept { return registered_.size(); }
    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    [[nodiscard]] bool pending() const noexcept { return state_ != State::Idle; }

private:
    Exec exec_;
    MPI_Comm comm_;
    int tag_;
    Transfer transfer_;
    MPI_Datatype datatype_ = MPI_DATATYPE_NULL;
    LocalIndex n_own_, n_cells_;
    Map send_map_, recv_map_;
    std::vector<Peer> peers_;
    std::vector<detail::Component<T>> registered_;
    std::vector<MPI_Request> requests_;
    Components components_;
    Segments send_segments_, recv_segments_;
    Peers send_peers_, recv_peers_;
    Buffer send_, recv_;
    HostBuffer send_host_, recv_host_;
    bool prepared_ = false, unit_stride_ = true;
    State state_ = State::Idle;

    static void validate_offsets(std::span<const std::size_t> offsets,
                                 std::size_t peers, std::size_t entries)
    {
        if (peers == 0 && offsets.empty()) {
            detail::require(entries == 0, "Halo: map entries without peers");
            return;
        }
        detail::require(offsets.size() == peers + 1 && offsets.front() == 0 &&
                        offsets.back() == entries && std::is_sorted(offsets.begin(), offsets.end()),
                        "Halo: invalid peer offsets");
    }

    void make_segments(Segments& segments, Peers& lookup, bool sending)
    {
        segments = Segments(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing,
                                               "halo/segments"), peers_.size());
        lookup = Peers(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "halo/slot_peers"),
                       sending ? send_map_.extent(0) : recv_map_.extent(0));
        auto segment_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, segments);
        auto lookup_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, lookup);
        for (std::size_t p = 0; p < peers_.size(); ++p) {
            const auto segment = sending ? peers_[p].send : peers_[p].recv;
            segment_host(p) = segment;
            for (std::size_t i = segment.offset; i < segment.offset + segment.count; ++i)
                lookup_host(i) = static_cast<int>(p);
        }
        detail::complete(exec_, [&] {
            Kokkos::deep_copy(exec_, segments, segment_host);
            Kokkos::deep_copy(exec_, lookup, lookup_host);
        });
    }

    T* send_pointer() const noexcept
    {
        if constexpr (host_memory) return send_.data();
        else return transfer_ == Transfer::DeviceDirect ? send_.data() : send_host_.data();
    }
    T* recv_pointer() const noexcept
    {
        if constexpr (host_memory) return recv_.data();
        else return transfer_ == Transfer::DeviceDirect ? recv_.data() : recv_host_.data();
    }

    template<bool UnitStride>
    void pack()
    {
        const auto width = registered_.size();
        using Kernel = detail::TransferKernel<true, UnitStride, Components, Map, Peers, Segments, Buffer>;
        // Neighboring lanes process neighboring map slots within one component.
        Kokkos::parallel_for("halo/pack", Policy(exec_, {0, 0}, {width, send_map_.extent(0)}, {1, 128}),
            Kernel{components_, send_map_, send_peers_, send_segments_, send_, width});
    }

    template<bool UnitStride>
    void unpack()
    {
        const auto width = registered_.size();
        using Kernel = detail::TransferKernel<false, UnitStride, Components, Map, Peers, Segments, Buffer>;
        Kokkos::parallel_for("halo/unpack", Policy(exec_, {0, 0}, {width, recv_map_.extent(0)}, {1, 128}),
            Kernel{components_, recv_map_, recv_peers_, recv_segments_, recv_, width});
    }
};

} // namespace krona::halo