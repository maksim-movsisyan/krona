#pragma once

#include <krona/execution/utils.hpp>
#include <krona/mpi/error.hpp>
#include <krona/mpi/communicator.hpp>
#include <krona/mpi/context.hpp>
#include <krona/mpi/datatype.hpp>
#include <krona/fields/fields_view.hpp>
#include <krona/fields/component.hpp>
#include <krona/halo/detail/types.hpp>
#include <krona/mesh/host/halo_plan.hpp>
#include <krona/mesh/device/mesh_view.hpp>

#include <algorithm>
#include <limits>
#include <vector>
#include <utility>
#include <type_traits>

namespace krona::halo {

namespace detail {

/// @brief Moves one component of one cell between the field and the buffer for its peer.
/// @tparam Packing True to gather owned values for sending, false to write received ghost values.
/// @tparam UnitStride True when every registered component stores consecutive cells contiguously.
/// @tparam ComponentView View of the registered component pointers and strides.
/// @tparam MapView View mapping send or receive slots to local cell IDs.
/// @tparam PeerView View mapping each slot to a peer index in the segment table.
/// @tparam SegmentView View of the start and cell count for each peer's part of the map.
/// @tparam BufferView View of the packed values for this transfer direction.
template<bool Packing, bool UnitStride, class ComponentView, class MapView,
         class PeerView, class SegmentView, class BufferView>
struct TransferKernel {
    ComponentView fields;
    MapView map;
    PeerView peers;
    SegmentView segments;
    BufferView buffer;
    std::size_t width;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t c, std::size_t i) const {
        // Find the component, peer and local cell for this map slot.
        const auto field = fields(c);
        const auto segment = segments(peers(i));
        const std::size_t cell = static_cast<std::size_t>(map(i));

        // Each peer gets its own message. Within it, all cells of one component are adjacent.
        // Skip the earlier peers, then the earlier components, then the earlier cells in this component.
        const auto offset = segment.offset * width + c * segment.count + (i - segment.offset);

        // The field may be strided, for example a single component of an AoS vector.
        const auto field_index = UnitStride ? cell : cell * field.stride;
        if constexpr (Packing) buffer(offset) = field.data[field_index];
        else field.data[field_index] = buffer(offset);
    }
};

} // namespace detail

/// @brief Selects the memory that MPI uses for packed messages.
enum class Transfer {
    HostStaged,  ///< Use host buffers for MPI, copying packed values to and from device memory as needed.
    DeviceDirect ///< Give MPI the packed buffers directly. MPI must support their memory space.
};

/// @brief Updates registered ghost values using one packed message per peer and direction.
/// Register components, call prepare(), then reuse the exchanger for each update.
/// Keep the mesh maps, fields, communicator and runtimes alive for the life of the exchanger.
/// @tparam T Mutable scalar type of every registered component, with a matching MPI datatype.
/// @tparam Exec Kokkos execution space for packing and unpacking. Its memory space holds the packed buffers.
template<class T = StorageReal, class Exec = execution::ExecutionSpace>
class Exchange {
    static_assert(!std::is_const_v<T>);
    using value_type = T;
    using memory_space = typename Exec::memory_space;
    using device_type = Kokkos::Device<Exec, memory_space>;
    using buffer_type = Kokkos::View<T*, Kokkos::LayoutRight, device_type>;
    using host_buffer_type = Kokkos::View<T*, Kokkos::LayoutRight, typename detail::StagingSpace<Exec>::type>;
    using component_type = fields::ComponentDescriptor<T, memory_space>;
    using components_type = Kokkos::View<component_type*, device_type>;
    using segments_type = Kokkos::View<detail::Segment*, device_type>;
    using peers_type = Kokkos::View<int*, device_type>;
    using map_type = mesh::device::detail::MeshArray<LocalIndex, memory_space, false>;
    using policy_type = Kokkos::MDRangePolicy<Exec,
                        Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
                        Kokkos::IndexType<std::size_t>>;
    static constexpr bool host_memory = std::is_same_v<memory_space, Kokkos::HostSpace>;

    // MPI completion and completion of the ghost writes are separate steps.
    enum class State { Idle, Communicating, Unpacking };
    struct Peer {
        int rank = 0;
        detail::Segment send, recv;
        int send_values = 0, recv_values = 0;
    };

public:
    /// @brief Borrows the cell maps and records the peer ranges for later exchanges.
    /// @param mesh View with MeshData::Halo selected. Its maps give local owned and ghost cell IDs.
    /// @param plan Host plan from the same mesh storage. Peer offsets are in cells and are copied here.
    /// @param comm Live MPI communicator whose rank numbering matches the plan.
    /// @param tag MPI tag shared by this exchange group across communicating ranks.
    /// @param exec Instance used for setup, pack and unpack kernels, and staging copies.
    /// @param transfer Memory to use for MPI buffers. HostStaged is the default.
    Exchange(mesh::device::MeshView<memory_space> mesh, mesh::host::HaloPlanView plan,
             mpi::CommView comm, int tag, Exec exec, Transfer transfer = Transfer::HostStaged)
        : exec_(exec), comm_(comm), tag_(tag), transfer_(transfer),
          n_own_(mesh.sizes.n_own), n_cells_(mesh.sizes.n_cells),
          send_map_(mesh.halo.send_owned_local), recv_map_(mesh.halo.recv_ghost_local) {
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

        // Copy each peer's slice of the maps. The view of the host plan is only needed here.
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
        datatype_ = mpi::datatype<T>();
    }

    Exchange(const Exchange&) = delete;
    Exchange& operator=(const Exchange&) = delete;
    Exchange(Exchange&&) = delete;
    Exchange& operator=(Exchange&&) = delete;

    /// @brief Drains outstanding MPI and unpack work before releasing communication buffers.
    ~Exchange() { if (state_ != State::Idle) wait(); }

    /// @brief Registers every component of a writable field in its original order.
    /// @tparam N Number of components per cell, for example 3 for a velocity field.
    /// @tparam Space Kokkos memory space holding the field values. Exec must be able to access it.
    /// @tparam Layout Kokkos layout of the field, which determines its component strides.
    /// @param field View indexed by local cell ID, covering at least the owned and ghost cells.
    template<int N, class Space, class Layout>
    void add_field(fields::FieldView<T, N, Space, Layout> field) {
        static_assert(Kokkos::SpaceAccessibility<Exec, Space>::accessible, "Execution space cannot access this field");

        mpi::require(!prepared_, "Halo: registration is closed after prepare");
        mpi::require(field.size() >= static_cast<std::size_t>(n_cells_),
                     "Halo fields must contain all owned and ghost cells");

        for (int c = 0; c < N; ++c) add_component(field.component(c));
    }

    /// @brief Registers every component of a pack in the order in which it appears in the pack.
    /// @tparam Pack A component pack with mutable values all of type T in memory accessible to Exec.
    /// @param pack Descriptors indexed by local cell ID, each covering at least the owned and ghost cells.
    /// Communicating peers must register the same components in the same order.
    template<class Pack>
    void add_pack(const Pack& pack) {
        static_assert(Pack::components > 0);
        static_assert(Pack::homogeneous, "Halo: one exchange requires a homogeneous component pack");
        static_assert(std::is_same_v<typename Pack::value_type, T>, "Halo: pack must be mutable and match the exchange scalar type");
        static_assert(Kokkos::SpaceAccessibility<Exec, typename Pack::memory_space>::accessible, "Halo: execution space cannot access this pack");

        mpi::require(!prepared_, "Halo: registration is closed after prepare");
        mpi::require(pack.size() >= static_cast<std::size_t>(n_cells_),
                     "Halo: pack must contain all owned and ghost cells");

        add_pack_impl(pack, std::make_index_sequence<Pack::components>{});
    }

    /// @brief Registers one writable scalar per cell, including a strided component of a vector.
    /// @tparam View Rank-one Kokkos view of mutable T values in memory accessible to Exec.
    /// @param component View covering all owned and ghost cells in local order. Keep its allocation alive.
    template<class View>
    void add_component(View component) {
        static_assert(Kokkos::is_view<View>::value && View::rank == 1);
        static_assert(std::is_same_v<typename View::value_type, T>);
        static_assert(Kokkos::SpaceAccessibility<Exec, typename View::memory_space>::accessible);

        mpi::require(!prepared_, "Halo: registration is closed after prepare");
        mpi::require(component.extent(0) >= static_cast<std::size_t>(n_cells_),
                     "Halo components must contain all owned and ghost cells");

        const component_type descriptor{component.data(), component.stride(0)};
        if (component.extent(0) != 0) {
            mpi::require(descriptor.data != nullptr && descriptor.stride != 0,
                         "Halo: component requires valid storage and a positive stride");
            for (const auto& previous : registered_)
                mpi::require(previous.data != descriptor.data, "Halo: duplicate field component");
        }
        registered_.push_back(descriptor);
        unit_stride_ = unit_stride_ && descriptor.stride == 1;
    }

    /// @brief Checks local maps and prepares reusable buffers for subsequent exchanges.
    /// Register at least one component before calling, including on ranks with no peers.
    /// This closes registration and waits for the setup copies on execution_space().
    void prepare() {
        mpi::require(!prepared_, "Halo: prepare was already called");
        mpi::require(!registered_.empty(), "Halo: register at least one field");
        const auto width = registered_.size();
        const auto send_values = mpi::checked_product(send_map_.extent(0), width);
        const auto recv_values = mpi::checked_product(recv_map_.extent(0), width);
        static_cast<void>(mpi::checked_product(send_values, sizeof(T)));
        static_cast<void>(mpi::checked_product(recv_values, sizeof(T)));

        // Convert cell counts to scalar counts for MPI_Isend and MPI_Irecv.
        for (auto& peer : peers_) {
            const auto send = mpi::checked_product(peer.send.count, width);
            const auto recv = mpi::checked_product(peer.recv.count, width);
            const std::size_t limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
            mpi::require(send <= limit && recv <= limit,
                         "Halo: peer payload exceeds MPI-3 int count; split the group");
            peer.send_values = static_cast<int>(send);
            peer.recv_values = static_cast<int>(recv);
        }

        // Inspect map indices on the host. HostSpace mirrors reuse the maps.
        // For device maps, wait for the copies before reading the temporary host mirrors.
        auto send_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, Kokkos::HostSpace{}, send_map_);
        auto recv_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, Kokkos::HostSpace{}, recv_map_);

        execution::complete(exec_, [&] {
            if constexpr (!host_memory) {
                Kokkos::deep_copy(exec_, send_host, send_map_);
                Kokkos::deep_copy(exec_, recv_host, recv_map_);
            }
        });
        for (std::size_t i = 0; i < send_host.extent(0); ++i)
            mpi::require(send_host(i) >= 0 && send_host(i) < n_own_, "Halo: send map must reference owned cells");

        // Two receive slots targeting the same ghost would race in the unpack kernel.
        std::vector<LocalIndex> destinations;
        destinations.reserve(recv_host.extent(0));
        for (std::size_t i = 0; i < recv_host.extent(0); ++i) {
            mpi::require(recv_host(i) >= n_own_ && recv_host(i) < n_cells_, "Halo: receive map must reference ghost cells");
            destinations.push_back(recv_host(i));
        }
        std::sort(destinations.begin(), destinations.end());
        mpi::require(std::adjacent_find(destinations.begin(), destinations.end()) == destinations.end(),
                     "Halo: receive map contains duplicate ghost cells");

        // Allocate the packed buffers in the execution space's memory for use by both kernels.
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
        // Copy the component descriptors and build the per-direction tables used by the kernels.
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

    /// @brief Posts receives, packs outgoing values, then posts sends.
    /// When sending, this waits for packing and any host staging before MPI reads the buffer.
    /// Finish producers on another execution-space instance before calling.
    /// If the previous round is still marked as unpacking, this waits before reusing its buffers.
    void start() {
        mpi::require(prepared_, "Halo: prepare must precede start");
        mpi::require(state_ != State::Communicating, "Halo: exchange is already active");
        if (state_ == State::Unpacking) wait(); // The previous unpack may still be reading the receive buffer.
        state_ = State::Communicating;
        mpi::guarded([&](){
            // Post receives first so peers can start sending while we pack our own values.
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
                // MPI reads the packed pointer on its own, so finish packing and staging before sending.
                exec_.fence("KRONA halo: send buffer ready for MPI");
            }

            // MPI addresses the start of each peer's packed message in scalar entries.
            for (std::size_t p = 0; p < peers_.size(); ++p) {
                const auto& peer = peers_[p];
                if (peer.send_values == 0) continue;
                mpi::check(MPI_Isend(send_pointer() + peer.send.offset * registered_.size(),
                     peer.send_values, datatype_, peer.rank, tag_,
                     comm_.native(), &requests_[peers_.size() + p]), "MPI_Isend");
            }
        });
    }

    /// @brief Waits for both MPI sends and receives, then queues writes to the ghost cells.
    /// Subsequent kernels on execution_space() see updated ghosts in order.
    /// Call wait() before reading the ghosts on the host or using them on another instance.
    void finish_async() {
        mpi::require(state_ == State::Communicating, "Halo: no active MPI exchange");
        mpi::guarded([&](){
            // MPI must release the receive buffer before staging or unpacking reads it.
            if (!requests_.empty())
                mpi::check(MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(),
                           MPI_STATUSES_IGNORE), "MPI_Waitall");

            if (recv_.extent(0) != 0) {
                state_ = State::Unpacking;
                // Queue any upload before the scatter so the kernel reads the received values.
                if constexpr (!host_memory)
                    if (transfer_ == Transfer::HostStaged) Kokkos::deep_copy(exec_, recv_, recv_host_);
                if (unit_stride_) unpack<true>(); else unpack<false>();
            } else state_ = State::Idle;
        });
    }

    /// @brief Completes this exchange's MPI requests and ghost writes.
    /// The execution-space fence runs when there is an unpack to finish.
    void wait() {
        if (state_ == State::Communicating) finish_async();
        if (state_ == State::Unpacking) {
            mpi::guarded([&](){exec_.fence("KRONA halo: ghost values ready");});
            state_ = State::Idle;
        }
    }

    /// @brief Runs a complete exchange and returns with the received ghost values ready.
    void exchange() { start(); wait(); }

    /// @brief Returns the number of registered scalar components. A vector with x, y and z adds three.
    [[nodiscard]] std::size_t component_count() const noexcept { return registered_.size(); }
    /// @brief Returns the instance on which to queue work that depends on this exchanger.
    [[nodiscard]] const Exec& execution_space() const noexcept { return exec_; }
    /// @brief Reports whether the exchanger still tracks a round as communicating or unpacking.
    /// finish_async() advances an active MPI exchange. wait() clears any pending state.
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
    std::vector<component_type> registered_;
    std::vector<MPI_Request> requests_;
    components_type components_;
    segments_type send_segments_, recv_segments_;
    peers_type send_peers_, recv_peers_;
    buffer_type send_, recv_;
    host_buffer_type send_host_, recv_host_;
    bool prepared_ = false, unit_stride_ = true;
    State state_ = State::Idle;

    /// @brief Builds the segment table and its slot-to-peer lookup for one direction.
    /// @param segments Output table with the start and length of each peer's map slice.
    /// @param lookup Output table giving the peer index for every slot in that map.
    /// @param sending True for the send map or false for the receive map.
    void make_segments(segments_type& segments, peers_type& lookup, bool sending) {
        segments = segments_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing,
                                 "halo/segments"), peers_.size());
        lookup = peers_type(Kokkos::view_alloc(exec_, Kokkos::WithoutInitializing, "halo/slot_peers"),
                            sending ? send_map_.extent(0) : recv_map_.extent(0));
        auto segment_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, segments);
        auto lookup_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, lookup);

        // Give the kernel a direct peer lookup for each slot, regardless of the peer's cell count.
        for (std::size_t p = 0; p < peers_.size(); ++p) {
            const auto segment = sending ? peers_[p].send : peers_[p].recv;
            segment_host(p) = segment;
            for (std::size_t i = segment.offset; i < segment.offset + segment.count; ++i)
                lookup_host(i) = static_cast<int>(p);
        }

        // Finish reading the host mirrors before they go out of scope.
        execution::complete(exec_, [&] {
            Kokkos::deep_copy(exec_, segments, segment_host);
            Kokkos::deep_copy(exec_, lookup, lookup_host);
        });
    }

    T* send_pointer() const noexcept {
        if constexpr (host_memory) return send_.data();
        else return transfer_ == Transfer::DeviceDirect ? send_.data() : send_host_.data();
    }
    T* recv_pointer() const noexcept {
        if constexpr (host_memory) return recv_.data();
        else return transfer_ == Transfer::DeviceDirect ? recv_.data() : recv_host_.data();
    }   

    template<class Pack, std::size_t... I>
    void add_pack_impl(const Pack& pack, std::index_sequence<I...>) {
        (add_component(pack.template component<I>()), ...);
    }

    template<bool UnitStride>
    void pack() {
        const auto width = registered_.size();
        using Kernel = detail::TransferKernel<true, UnitStride, components_type, map_type, peers_type, segments_type, buffer_type>;
        // Keep consecutive map slots in the inner dimension, with one component per tile.
        Kokkos::parallel_for("halo/pack", policy_type(exec_, {0, 0}, {width, send_map_.extent(0)}, {1, 128}),
            Kernel{components_, send_map_, send_peers_, send_segments_, send_, width});
    }

    template<bool UnitStride>
    void unpack() {
        const auto width = registered_.size();
        using Kernel = detail::TransferKernel<false, UnitStride, components_type, map_type, peers_type, segments_type, buffer_type>;
        Kokkos::parallel_for("halo/unpack", policy_type(exec_, {0, 0}, {width, recv_map_.extent(0)}, {1, 128}),
            Kernel{components_, recv_map_, recv_peers_, recv_segments_, recv_, width});
    }
};

} // namespace krona::halo