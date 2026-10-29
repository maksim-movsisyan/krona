#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mpi/communicator.hpp>
#include <krona/mpi/context.hpp>
#include <krona/fields/fields_view.hpp>
#include <krona/halo/detail/types.hpp>
#include <krona/mesh/host/halo_plan.hpp>
#include <krona/mesh/device/mesh_view.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace krona::halo {

namespace detail {

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
    using value_type = T;
    using memory_space = typename Exec::memory_space;
    using device_type = Kokkos::Device<Exec, memory_space>;
    using buffer_type = Kokkos::View<T*, Kokkos::LayoutRight, device_type>;
    using host_buffer_type = Kokkos::View<T*, Kokkos::LayoutRight, typename detail::StagingSpace<Exec>::type>;
    using components_type = Kokkos::View<detail::Component<T>*, device_type>;
    using segments_type = Kokkos::View<detail::Segment*, device_type>;
    using peers_type = Kokkos::View<int*, device_type>;
    using map_type = mesh::device::detail::MeshArray<LocalIndex, memory_space, false>;
    using policy_type = Kokkos::MDRangePolicy<Exec,
                        Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
                        Kokkos::IndexType<std::size_t>>;
    static constexpr bool host_memory = std::is_same_v<memory_space, Kokkos::HostSpace>;

    enum class State { Idle, Communicating, Unpacking };
    struct Peer {
        int rank = 0;
        detail::Segment send, recv;
        int send_values = 0, recv_values = 0;
    };

public:
    Exchange(mesh::device::MeshView<memory_space> mesh, mesh::host::HaloPlanView plan,
             mpi::CommView comm, int tag, Exec exec, Transfer transfer = Transfer::HostStaged)
        : exec_(exec), comm_(comm), tag_(tag), transfer_(transfer),
          n_own_(mesh.sizes.n_own), n_cells_(mesh.sizes.n_cells),
          send_map_(mesh.halo.send_owned_local), recv_map_(mesh.halo.recv_ghost_local)
    {
        execution::require_kokkos();
        mpi::require(mpi::is_active(), "Halo requires active MPI and a valid communicator");
        mpi::require(mesh.has(mesh::device::MeshData::Halo), "Halo maps were not selected");
        mpi::require(n_own_ >= 0 && n_cells_ >= n_own_, "Halo: invalid cell counts");
        mpi::require(transfer == Transfer::HostStaged || transfer == Transfer::DeviceDirect, "Halo: invalid transfer mode");

        int size = comm_.size(), flag = 0;
        int* tag_limit = nullptr;
        mpi::check(MPI_Comm_get_attr(comm_.native(), MPI_TAG_UB, &tag_limit, &flag), "MPI_Comm_get_attr");
        // In the MPI world model, predefined environmental attributes are guaranteed on WORLD.
        if (flag == 0)
            mpi::check(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_limit, &flag),
                       "MPI_Comm_get_attr(MPI_COMM_WORLD)");
        mpi::require(flag != 0 && tag_limit != nullptr && tag >= 0 && tag <= *tag_limit,
                     "Halo: tag is outside the communicator's valid range");

        const std::size_t count = plan.n_neighbors();
        mpi::require(count <= static_cast<std::size_t>(std::numeric_limits<int>::max() / 2),
                     "Halo: too many MPI requests");
        mpi::require_offsets(plan.send_offsets, count, send_map_.extent(0));
        mpi::require_offsets(plan.recv_offsets, count, recv_map_.extent(0));
        peers_.reserve(count);
        std::vector<int> ranks;
        ranks.reserve(count);
        for (std::size_t p = 0; p < count; ++p) {
            const int rank = plan.nb_ranks[p];
            mpi::require(rank >= 0 && rank < size, "Halo: peer rank is outside communicator");
            ranks.push_back(rank);
            peers_.push_back({rank, {plan.send_offsets[p], plan.send_count(p)},
                                   {plan.recv_offsets[p], plan.recv_count(p)}, 0, 0});
        }
        std::sort(ranks.begin(), ranks.end());
        mpi::require(std::adjacent_find(ranks.begin(), ranks.end()) == ranks.end(), "Halo: duplicate peer rank");
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
        mpi::require(!prepared_, "Halo: registration is closed after prepare");
        mpi::require(field.size() >= static_cast<std::size_t>(n_cells_),
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
        mpi::require(!prepared_, "Halo: registration is closed after prepare");
        mpi::require(component.extent(0) >= static_cast<std::size_t>(n_cells_),
                     "Halo components must contain all owned and ghost cells");
        const detail::Component<T> descriptor{component.data(), component.stride(0)};
        if (component.extent(0) != 0) {
            mpi::require(descriptor.data != nullptr && descriptor.stride != 0,
                         "Halo: component requires valid storage and a positive stride");
            for (const auto& previous : registered_)
                mpi::require(previous.data != descriptor.data, "Halo: duplicate field component");
        }
        registered_.push_back(descriptor);
        unit_stride_ = unit_stride_ && descriptor.stride == 1;
    }

    /// @brief Allocates reusable buffers and validates maps; local, not collective.
    void prepare()
    {
        mpi::require(!prepared_, "Halo: prepare was already called");
        mpi::require(!registered_.empty(), "Halo: register at least one field");
        const auto width = registered_.size();
        const auto send_values = mpi::checked_product(send_map_.extent(0), width);
        const auto recv_values = mpi::checked_product(recv_map_.extent(0), width);
        static_cast<void>(mpi::checked_product(send_values, sizeof(T)));
        static_cast<void>(mpi::checked_product(recv_values, sizeof(T)));
        for (auto& peer : peers_) {
            const auto send = mpi::checked_product(peer.send.count, width);
            const auto recv = mpi::checked_product(peer.recv.count, width);
            const std::size_t limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
            mpi::require(send <= limit && recv <= limit,
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
        execution::complete(exec_, [&] {
            if constexpr (!host_memory) {
                Kokkos::deep_copy(exec_, send_host, send_map_);
                Kokkos::deep_copy(exec_, recv_host, recv_map_);
            }
        });
        for (std::size_t i = 0; i < send_host.extent(0); ++i)
            mpi::require(send_host(i) >= 0 && send_host(i) < n_own_,
                         "Halo: send map must reference owned cells");
        std::vector<LocalIndex> destinations;
        destinations.reserve(recv_host.extent(0));
        for (std::size_t i = 0; i < recv_host.extent(0); ++i) {
            mpi::require(recv_host(i) >= n_own_ && recv_host(i) < n_cells_,
                         "Halo: receive map must reference ghost cells");
            destinations.push_back(recv_host(i));
        }
        std::sort(destinations.begin(), destinations.end());
        mpi::require(std::adjacent_find(destinations.begin(), destinations.end()) == destinations.end(),
                     "Halo: receive map contains duplicate ghost cells");

        send_ = buffer_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "halo/send"), send_values);
        recv_ = buffer_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "halo/recv"), recv_values);
        if constexpr (!host_memory) {
            if (transfer_ == Transfer::HostStaged) {
                send_host_ = host_buffer_type(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              "halo/send_host"), send_values);
                recv_host_ = host_buffer_type(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              "halo/recv_host"), recv_values);
            }
        }
        components_ = components_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing,
                                      "halo/components"), width);
        auto components_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, components_);
        for (std::size_t c = 0; c < width; ++c) components_host(c) = registered_[c];
        execution::complete(exec_, [&] {
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
        mpi::require(prepared_, "Halo: prepare must precede start");
        mpi::require(state_ != State::Communicating, "Halo: exchange is already active");
        if (state_ == State::Unpacking) wait(); // Previous unpack still owns the receive buffer.
        state_ = State::Communicating;
        mpi::guarded([&](){
            for (std::size_t p = 0; p < peers_.size(); ++p) {
                const auto& peer = peers_[p];
                if (peer.recv_values == 0) continue;
                mpi::check(MPI_Irecv(recv_pointer() + peer.recv.offset * registered_.size(),
                    peer.recv_values, datatype_, peer.rank, tag_,
                    comm_.native(), &requests_[p]), "MPI_Irecv");
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
                     peer.send_values, datatype_, peer.rank, tag_,
                     comm_.native(), &requests_[peers_.size() + p]), "MPI_Isend");
            }
        });
    }

    /// @brief Waits for MPI, then queues receive upload and ghost scatter on exec.
    /// Subsequent kernels on this same instance observe updated ghosts in order.
    void finish_async()
    {
        mpi::require(state_ == State::Communicating, "Halo: no active MPI exchange");
        mpi::guarded([&](){
            if (!requests_.empty())
                mpi::check(MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(),
                           MPI_STATUSES_IGNORE), "MPI_Waitall");
            if (recv_.extent(0) != 0) {
                state_ = State::Unpacking;
                if constexpr (!host_memory)
                    if (transfer_ == Transfer::HostStaged) Kokkos::deep_copy(exec_, recv_, recv_host_);
                if (unit_stride_) unpack<true>(); else unpack<false>();
            } else state_ = State::Idle;
        });
    }

    /// @brief Completes outstanding communication and makes ghost values ready for external users.
    void wait()
    {
        if (state_ == State::Communicating) finish_async();
        if (state_ == State::Unpacking) {
            mpi::guarded([&](){exec_.fence("KRONA halo: ghost values ready");});
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
    mpi::CommView comm_;
    int tag_;
    Transfer transfer_;
    MPI_Datatype datatype_ = MPI_DATATYPE_NULL;
    LocalIndex n_own_, n_cells_;
    map_type send_map_, recv_map_;
    std::vector<Peer> peers_;
    std::vector<detail::Component<T>> registered_;
    std::vector<MPI_Request> requests_;
    components_type components_;
    segments_type send_segments_, recv_segments_;
    peers_type send_peers_, recv_peers_;
    buffer_type send_, recv_;
    host_buffer_type send_host_, recv_host_;
    bool prepared_ = false, unit_stride_ = true;
    State state_ = State::Idle;

    void make_segments(segments_type& segments, peers_type& lookup, bool sending)
    {
        segments = segments_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing,
                                 "halo/segments"), peers_.size());
        lookup = peers_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "halo/slot_peers"),
                            sending ? send_map_.extent(0) : recv_map_.extent(0));
        auto segment_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, segments);
        auto lookup_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, lookup);
        for (std::size_t p = 0; p < peers_.size(); ++p) {
            const auto segment = sending ? peers_[p].send : peers_[p].recv;
            segment_host(p) = segment;
            for (std::size_t i = segment.offset; i < segment.offset + segment.count; ++i)
                lookup_host(i) = static_cast<int>(p);
        }
        execution::complete(exec_, [&] {
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
        using Kernel = detail::TransferKernel<true, UnitStride, components_type, map_type, peers_type, segments_type, buffer_type>;
        // Neighboring lanes process neighboring map slots within one component.
        Kokkos::parallel_for("halo/pack", policy_type(exec_, {0, 0}, {width, send_map_.extent(0)}, {1, 128}),
            Kernel{components_, send_map_, send_peers_, send_segments_, send_, width});
    }

    template<bool UnitStride>
    void unpack()
    {
        const auto width = registered_.size();
        using Kernel = detail::TransferKernel<false, UnitStride, components_type, map_type, peers_type, segments_type, buffer_type>;
        Kokkos::parallel_for("halo/unpack", policy_type(exec_, {0, 0}, {width, recv_map_.extent(0)}, {1, 128}),
            Kernel{components_, recv_map_, recv_peers_, recv_segments_, recv_, width});
    }
};

} // namespace krona::halo