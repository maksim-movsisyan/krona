#pragma once

#include <cstdint>

namespace krona::mesh {

enum class AuxConnType : std::uint64_t {
    None                        = 0,
    CellFaces                   = std::uint64_t{1} << 0,
    CellCellsByFace             = std::uint64_t{1} << 1,
    NodeCells                   = std::uint64_t{1} << 2,
    NodeFaces                   = std::uint64_t{1} << 3,
    CellCellsByNode             = std::uint64_t{1} << 4,
    All                         = (std::uint64_t{1} << 5) - 1
};

/// @brief Combines requested connectivity flags.
[[nodiscard]] constexpr AuxConnType operator|(AuxConnType a, AuxConnType b) noexcept
{
    return static_cast<AuxConnType>(static_cast<std::uint64_t>(a) | static_cast<std::uint64_t>(b));
}

/// @brief Selects flags shared by two connectivity masks.
[[nodiscard]] constexpr AuxConnType operator&(AuxConnType a, AuxConnType b) noexcept
{
    return static_cast<AuxConnType>(static_cast<std::uint64_t>(a) & static_cast<std::uint64_t>(b));
}

/// @brief Returns whether every requested flag is present; None is always present.
[[nodiscard]] constexpr bool has_flag(AuxConnType mask, AuxConnType flag) noexcept
{
    return (mask & flag) == flag;
}


enum class AuxGeomType : std::uint64_t {
    None                        = std::uint64_t{0},
    FaceCellDistance            = std::uint64_t{1} << 0,
    FaceCellDistanceVector      = std::uint64_t{1} << 1,
    FaceCellDistanceInv         = std::uint64_t{1} << 2,
    FaceInterpolationWeights    = std::uint64_t{1} << 3,
    FaceNonOrthogonalityVector  = std::uint64_t{1} << 4,
    FaceNormalDistance          = std::uint64_t{1} << 5,
    FaceNormalDistanceInv       = std::uint64_t{1} << 6,
    FaceAreaOverNormalDistance  = std::uint64_t{1} << 7,
    WallDistance                = std::uint64_t{1} << 8,
    AllFaces                    = (std::uint64_t{1} << 8) - std::uint64_t{1},
    All                         = (std::uint64_t{1} << 9) - std::uint64_t{1},
};

/// @brief Combines requested auxiliary geometry flags.
[[nodiscard]] constexpr AuxGeomType operator|(AuxGeomType a, AuxGeomType b) noexcept
{
    return static_cast<AuxGeomType>(static_cast<std::uint64_t>(a) | static_cast<std::uint64_t>(b));
}
/// @brief Selects flags shared by two geometry masks.
[[nodiscard]] constexpr AuxGeomType operator&(AuxGeomType a, AuxGeomType b) noexcept
{
    return static_cast<AuxGeomType>(static_cast<std::uint64_t>(a) & static_cast<std::uint64_t>(b));
}
/// @brief Returns whether every requested flag is present; None is always present.
[[nodiscard]] constexpr bool has_flag(AuxGeomType mask, AuxGeomType flag) noexcept
{
    return (mask & flag) == flag;
}

} // namespace krona::mesh
