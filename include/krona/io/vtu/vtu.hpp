#pragma once

#include <krona/mesh/host/mesh_part.hpp>
#include <krona/mpi/communicator.hpp>

#include <array>
#include <filesystem>
#include <span>
#include <string_view>
#include <type_traits>

namespace krona::io::vtk {

enum class ScalarType { i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 };

/// @brief Resolves a supported numeric C++ type to its VTK scalar type.
template<class T> constexpr ScalarType scalar_type()
{
    using V = std::remove_cv_t<T>;
    // Exact numeric types keep the file's representation explicit.
    if constexpr (std::is_same_v<V, std::int8_t>) return ScalarType::i8;
    else if constexpr (std::is_same_v<V, std::uint8_t>) return ScalarType::u8;
    else if constexpr (std::is_same_v<V, std::int16_t>) return ScalarType::i16;
    else if constexpr (std::is_same_v<V, std::uint16_t>) return ScalarType::u16;
    else if constexpr (std::is_same_v<V, std::int32_t>) return ScalarType::i32;
    else if constexpr (std::is_same_v<V, std::uint32_t>) return ScalarType::u32;
    else if constexpr (std::is_same_v<V, std::int64_t>) return ScalarType::i64;
    else if constexpr (std::is_same_v<V, std::uint64_t>) return ScalarType::u64;
    else if constexpr (std::is_same_v<V, float>) return ScalarType::f32;
    else if constexpr (std::is_same_v<V, double>) return ScalarType::f64;
    else static_assert(!sizeof(T), "Unsupported VTK scalar type");
}

/// @brief Describes a borrowed numeric field or a generated index sequence.
/// stride=components is interleaved, stride=1 with soa=true is SoA, stride=0 repeats one tuple.
struct Field {
    std::string_view name;
    ScalarType type{};
    std::array<const void*, 3> data{};
    std::size_t tuples = 0, first_tuple = 0, stride = 1;
    int components = 1;
    bool soa = false;
    bool indices = false; // Generate first_tuple + i instead of reading data.

    /// @brief Creates a borrowed subrange without copying its numeric values.
    [[nodiscard]] Field slice(std::size_t first, std::size_t count) const;
};

/// @brief Borrows a scalar or interleaved vector/tensor array; components must divide its length.
template<class T, std::size_t Extent> [[nodiscard]] Field field(std::string_view name, std::span<T, Extent> values,
                                           int components = 1)
{
    // Shape validation is shared with the non-template implementation.
    Field result{name, scalar_type<T>(), {values.data(), nullptr, nullptr}, values.size(), 0,
                 static_cast<std::size_t>(components), components, false};
    if (components <= 0 || values.size() % static_cast<std::size_t>(components) != 0)
        result.components = 0;
    else result.tuples /= static_cast<std::size_t>(components);
    return result;
}

/// @brief Borrows a three-component vector in SoA layout without building an interleaved copy.
template<class T, std::size_t X, std::size_t Y, std::size_t Z>
[[nodiscard]] Field vector_field(std::string_view name, std::span<T, X> x,
                                 std::span<T, Y> y, std::span<T, Z> z)
{
    // A mismatched component length is rejected before creating the output file.
    return {name, scalar_type<T>(), {x.data(), y.data(), z.data()}, x.size(), 0, 1,
            x.size() == y.size() && x.size() == z.size() ? 3 : 0, true};
}

/// @brief Borrows one scalar and repeats it over every tuple without allocating an array.
template<class T> [[nodiscard]] Field constant_field(std::string_view name, T& value,
                                                    std::size_t tuples)
{
    // The lvalue reference prevents accidentally borrowing a temporary scalar.
    return {name, scalar_type<T>(), {&value, nullptr, nullptr}, tuples, 0, 0, 1, false};
}

/// @brief Generates consecutive LocalIndex values without allocating an ID array.
[[nodiscard]] Field index_field(std::string_view name, std::size_t tuples, std::size_t first = 0);

/// @brief Borrows a linear unstructured grid with CGNS node ordering and local CSR connectivity.
/// CSR offsets may start above zero when the view selects a contiguous subset of cells.
struct GridView {
    std::span<const GeometryReal> x, y, z;
    std::span<const std::size_t> offsets;
    std::span<const LocalIndex> connectivity;
    std::span<const mesh::CellType> types;
};

/// @brief Borrows optional point and cell fields for one grid piece.
struct Fields {
    std::span<const Field> point{}, cell{};
};

/// @brief Writes one local binary VTU piece; no MPI communication or VTK dependency is used.
void write_vtu_piece(GridView grid, const std::filesystem::path& path, Fields fields = {});

/// @brief Writes one piece per rank and a PVTU index collectively; field schemas must match.
void write_parallel_vtu(GridView grid, const std::filesystem::path& directory,
                        std::string_view stem, mpi::CommView comm, Fields fields = {});

/// @brief Borrows additional fields for MeshPart volume cells and boundary faces.
/// Point/cell/boundary arrays use all local nodes/cells/faces; already trimmed arrays also work.
struct MeshFields {
    std::span<const Field> point{}, cell{}, boundary{};
};

/// @brief Writes owned volume cells and physical boundary faces into two parallel VTK datasets.
/// Exports rank, GIDs, local IDs and metrics from a geometric mesh with grouped faces.
void write_vtu(const mesh::host::MeshPart& mesh, const std::filesystem::path& directory,
               std::string_view stem, mpi::CommView comm, MeshFields fields = {});

} // namespace krona::io::vtk