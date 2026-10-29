# KRONA Mesh Generator

Define a mesh in TOML using three parts: vertices describe the geometry, blocks specify the cell counts and spacing, and boundary patches name the outer faces. The result is a three-dimensional CGNS mesh with `HEXA_8` volume cells and `QUAD_4` boundary elements.

## Contents

- [1. Complete example: unit cube](#1-complete-example-unit-cube)
- [2. Configuration fields](#2-configuration-fields)
- [3. Vertex order, local axes, and faces](#3-vertex-order-local-axes-and-faces)
- [4. Cell spacing: grading](#4-cell-spacing-grading)
- [5. Connecting multiple blocks](#5-connecting-multiple-blocks)
- [6. Contracts and limitations](#6-contracts-and-limitations)
- [7. CGNS output](#7-cgns-output)
- [8. Fixing configuration errors](#8-fixing-configuration-errors)

## 1. Complete example: unit cube

This configuration describes `0 <= x, y, z <= 1`. It produces **10 × 8 × 6 = 480 cells** and **11 × 9 × 7 = 693 nodes**, with a separate patch for each side.

```text
   v7 ---------- v6
  /|            /|
 / |           / |
v4 ---------- v5 |
|  |          |  |
|  v3 --------|- v2
| /           | /
|/            |/
v0 ---------- v1

Local axes: xi = v0 -> v1, eta = v0 -> v3, zeta = v0 -> v4.
For this cube, they coincide with +x, +y, and +z.
```

```toml
[mesh]
name = "Unit cube"
scale = 1.0
vertices = [
    [0.0, 0.0, 0.0], # 0
    [1.0, 0.0, 0.0], # 1
    [1.0, 1.0, 0.0], # 2
    [0.0, 1.0, 0.0], # 3
    [0.0, 0.0, 1.0], # 4
    [1.0, 0.0, 1.0], # 5
    [1.0, 1.0, 1.0], # 6
    [0.0, 1.0, 1.0], # 7
]

[[blocks]]
name = "cube"
vertices = [0, 1, 2, 3, 4, 5, 6, 7]
cells = [10, 8, 6]
# Omitting grading_type and grading gives uniform spacing.

[[boundaries]]
name = "bottom"
faces = [[0, 3, 2, 1]]

[[boundaries]]
name = "front"
faces = [[0, 1, 5, 4]]

[[boundaries]]
name = "right"
faces = [[1, 2, 6, 5]]

[[boundaries]]
name = "back"
faces = [[2, 3, 7, 6]]

[[boundaries]]
name = "left"
faces = [[3, 0, 4, 7]]

[[boundaries]]
name = "top"
faces = [[4, 5, 6, 7]]
```

Change `mesh.vertices` to change the shape or dimensions. Change `cells` to change the resolution. Add `grading_type` and `grading` to a block to control its spacing.

## 2. Configuration fields

The only top-level sections are `[mesh]`, `[[blocks]]`, and `[[boundaries]]`. Unknown keys and incorrect value types are errors. Grading type names are case-sensitive and lowercase.

### `[mesh]`: shared geometry

| Field | Meaning | Required / default |
|---|---|---|
| `name` | Diagnostic name; does not set the output filename or CGNS zone name | Optional; `"Mesh"` |
| `scale` | Multiplier applied to every coordinate: `p = scale × p_input`. Must be finite and strictly positive | Optional; `1.0` |
| `vertices` | Nonempty array of points `[x, y, z]`. Coordinates must be numeric and finite after scaling | Required |

A vertex index is its position in `mesh.vertices`, **starting at 0**. Every block and boundary patch refers to this shared list. For example, `scale = 0.001` converts coordinates specified in millimetres to metres.

### `[[blocks]]`: one volume block

Repeat this section for each block. At least one block is required.

| Field | Meaning | Required / default |
|---|---|---|
| `name` | Diagnostic name | Optional; `"block_0"`, `"block_1"`, … |
| `vertices` | Exactly 8 distinct indices into `mesh.vertices`, in `HEXA_8` order | Required |
| `cells` | Exactly 3 positive integers `[N_xi, N_eta, N_zeta]`: cell counts along the local axes | Required |
| `grading_type` | Exactly 3 strings: one spacing type per local axis | Optional; `["uniform", "uniform", "uniform"]` |
| `grading` | Exactly 3 finite numbers: parameters for the corresponding spacing types | Optional; `[1.0, 1.0, 1.0]` |

`cells = [10, 8, 6]` means 480 cells. An isolated block has `(10 + 1) × (8 + 1) × (6 + 1)` nodes. Shared nodes between blocks are counted once.

Counts and indices must be TOML integers: `10` is valid; `10.0` is not. Every axis needs at least one cell. All blocks are three-dimensional.

### `[[boundaries]]`: one boundary patch

Repeat this section for each patch. One patch may contain several outer faces from different blocks.

| Field | Meaning | Required / default |
|---|---|---|
| `name` | Unique patch name, also used in CGNS | Optional; `"patch_0"`, `"patch_1"`, … |
| `faces` | Nonempty array of faces. Each face contains 4 distinct indices into `mesh.vertices`, ordered around its perimeter | Required for each patch |

Patch names must occupy **1–32 UTF-8 bytes**, contain neither `/` nor a null character, and differ from `.` and `..`. The names `Volume_HEXA`, `GridCoordinates`, `ZoneBC`, and `ZoneType` are reserved. All patch names, including automatically assigned names, must be unique.

Boundary patches are optional. **All unlisted outer macro faces are collected into one automatic patch**, normally named `patch_automatic`. Its faces are written as outward-oriented `QUAD_4` elements with a matching `BC_t` entry, just like an explicit patch.

- If all outer faces are assigned explicitly, no automatic patch is created and no warning is printed.
- If `boundaries` is omitted entirely, the automatic patch contains the whole outer boundary.
- Internal block interfaces are excluded.
- If `patch_automatic` is already an explicit patch name, the generator uses the first unused name from `patch_automatic_1`, `patch_automatic_2`, and so on. Explicit patches are kept separate.
- Rank 0 of the supplied communicator prints one warning with the number of added macro faces, boundary quads, and the chosen patch name. Other ranks print no such warning.

For the cube in Section 1, removing all `[[boundaries]]` sections produces one automatic patch containing all 6 outer macro faces and 376 boundary quads:

```text
[KRONA warning] 6 unassigned outer macro face(s) (376 boundary quads) were added to patch 'patch_automatic'.
```

Automatic assignment supplies a named boundary group. Assign its physical boundary conditions in the solver, or add explicit patches to distinguish its faces.

## 3. Vertex order, local axes, and faces

A position in `blocks.vertices` is a **local vertex number**. The value at that position is an **index into the shared vertex list**. For example, in `[1, 8, 9, 2, 5, 11, 10, 6]`, local vertex 0 is shared vertex 1.

The eight local positions follow this order:

| Local vertex | xi | eta | zeta |
|---|---|---|---|
| v0 | 0 | 0 | 0 |
| v1 | 1 | 0 | 0 |
| v2 | 1 | 1 | 0 |
| v3 | 0 | 1 | 0 |
| v4 | 0 | 0 | 1 |
| v5 | 1 | 0 | 1 |
| v6 | 1 | 1 | 1 |
| v7 | 0 | 1 | 1 |

The local directions are **xi: v0 → v1**, **eta: v0 → v3**, and **zeta: v0 → v4**. These coincide with `x`, `y`, and `z` only when the block is oriented that way. The entries in `cells`, `grading_type`, and `grading` always follow **xi, eta, zeta**.

The eight corners define a trilinear map from the parameter cube `[0, 1]^3` to the physical block. Edges are straight; faces are bilinear. Skewed blocks are allowed, provided the map preserves positive volume orientation.

### Selecting a face

The numbers below are **local**. Replace each number with the corresponding entry from `blocks.vertices` when writing `boundaries.faces`.

| Side | Location | Vertices in outward order | Number of boundary quads |
|---|---|---|---|
| Bottom | zeta = 0 | `[0, 3, 2, 1]` | `N_xi × N_eta` |
| Front | eta = 0 | `[0, 1, 5, 4]` | `N_xi × N_zeta` |
| Right | xi = 1 | `[1, 2, 6, 5]` | `N_eta × N_zeta` |
| Back | eta = 1 | `[2, 3, 7, 6]` | `N_xi × N_zeta` |
| Left | xi = 0 | `[3, 0, 4, 7]` | `N_eta × N_zeta` |
| Top | zeta = 1 | `[4, 5, 6, 7]` | `N_xi × N_eta` |

Vertices must follow the perimeter. A cyclic shift or reversed traversal is accepted: `[0, 3, 2, 1]`, `[3, 2, 1, 0]`, and `[0, 1, 2, 3]` identify the same face. A crossed order such as `[0, 2, 3, 1]` is invalid. The generator orients the output boundary quads outward regardless of the accepted input traversal.

## 4. Cell spacing: grading

Along an axis with N cells, the generator computes **N + 1 node coordinates**: `s[0] = 0`, …, `s[N] = 1`. The N intervals between consecutive nodes define the cells. “Start” and “end” refer to the local axis direction from Section 3.

| `grading_type` | Meaning of `grading` | Where cells are smaller | Effect of the parameter |
|---|---|---|---|
| `"uniform"` | Ignored; any finite number is accepted | Equal spacing | Use `1.0` for clarity |
| `"geometric"` | `r > 0`: last-cell width divided by first-cell width along this axis | Near the start for `r > 1`; near the end for `0 < r < 1` | `r = 1` gives uniform spacing. Moving r further from 1 increases the size contrast |
| `"tanh_start"` | `delta > 0`: dimensionless clustering strength | Near the start, `s = 0` | Larger delta gives stronger clustering |
| `"tanh_end"` | `delta > 0`: dimensionless clustering strength | Near the end, `s = 1` | Larger delta gives stronger clustering |
| `"tanh_both"` | `delta > 0`: dimensionless, symmetric clustering strength | Near both ends, `s = 0` and `s = 1` | Larger delta makes the end cells smaller relative to the central cells |

**For geometric grading, r is the ratio of the last and first cell widths.** For N > 1, consecutive cell widths have ratio `alpha = r^(1 / (N - 1))`.

For example, N = 4 and r = 8 give:

```text
Cell widths:      [1/15, 2/15, 4/15, 8/15]
Node coordinates: [0,    1/15, 3/15, 7/15, 1]

Last width / first width = 8.
Each width / previous width = 2.
```

The tanh parameter controls the distribution shape. The first-cell width and physical layer thickness also depend on the cell count and block geometry; delta does not prescribe either directly. As delta approaches zero, spacing approaches uniform. The implementation uses exactly uniform spacing for `0 < delta <= 1e-8`; delta = 0 is invalid.

### Example: clustering toward two walls

Add these fields to the cube's existing `[[blocks]]` section, immediately after `cells`:

```toml
# Fields of the existing cube block.
grading_type = ["uniform", "tanh_both", "uniform"]
grading = [1.0, 2.5, 1.0]
```

Spacing stays uniform along xi and zeta. Along eta, nodes cluster toward eta = 0 and eta = 1: the cube walls at `y = 0` and `y = 1`.

**Setting `grading` alone leaves all types as `uniform`, so the parameters have no effect.** Setting `grading_type` alone uses parameter `1.0` on every axis, including tanh axes.

With one cell along an axis, only the two endpoints exist. Grading cannot move them, but its parameter must still satisfy the validity rules.

## 5. Connecting multiple blocks

Adjacent blocks connect through **the same shared vertex indices**. Equal coordinates with different indices do not merge nodes.

The following views show the bottom and top layers of two adjacent blocks. In both views, +x points right and +y points up.

```text
Bottom: z = 0                      Top: z = 1

v3 ----------- v2 ----------- v9   v7 ----------- v6 ----------- v10
|              |              |    |              |              |
|     left     |    right     |    |     left     |    right     |
|              |              |    |              |              |
v0 ----------- v1 ----------- v8   v4 ----------- v5 ----------- v11

Shared face: [1, 2, 6, 5]. It is internal, not a boundary patch.
```

This complete example covers `0 <= x <= 2` and `0 <= y, z <= 1`. There are 8 cells along x in the left block and 10 in the right block. Both have a 12 × 6 subdivision of the shared face. The result has **1296 cells and 1729 unique nodes**.

```toml
[mesh]
name = "Two-block channel"
vertices = [
    [0.0, 0.0, 0.0], # 0
    [1.0, 0.0, 0.0], # 1
    [1.0, 1.0, 0.0], # 2
    [0.0, 1.0, 0.0], # 3
    [0.0, 0.0, 1.0], # 4
    [1.0, 0.0, 1.0], # 5
    [1.0, 1.0, 1.0], # 6
    [0.0, 1.0, 1.0], # 7
    [2.0, 0.0, 0.0], # 8
    [2.0, 1.0, 0.0], # 9
    [2.0, 1.0, 1.0], # 10
    [2.0, 0.0, 1.0], # 11
]

[[blocks]]
name = "left"
vertices = [0, 1, 2, 3, 4, 5, 6, 7]
cells = [8, 12, 6]

[[blocks]]
name = "right"
vertices = [1, 8, 9, 2, 5, 11, 10, 6]
cells = [10, 12, 6]

[[boundaries]]
name = "inlet"
faces = [[3, 0, 4, 7]]

[[boundaries]]
name = "outlet"
faces = [[8, 9, 10, 11]]

[[boundaries]]
name = "walls"
faces = [
    [0, 3, 2, 1],   # Left block: bottom.
    [1, 2, 9, 8],   # Right block: bottom.
    [4, 5, 6, 7],   # Left block: top.
    [5, 11, 10, 6], # Right block: top.
    [0, 1, 5, 4],   # Left block: front.
    [1, 8, 11, 5],  # Right block: front.
    [2, 3, 7, 6],   # Left block: back.
    [9, 2, 6, 10],  # Right block: back.
]
```

The shared face `[1, 2, 6, 5]` is omitted from `boundaries`. Cell counts **along the face** and node positions must agree. The cell count perpendicular to the face may differ.

### Rotated local axes

In the same example, replace the right block's `vertices` and `cells` fields with:

```toml
# Replacement fields of the existing right block.
vertices = [2, 9, 10, 6, 1, 8, 11, 5]
cells = [10, 6, 12]
```

Its local directions become **xi = +x, eta = +z, zeta = -y**. Counts 12 and 6 move with their directions. The physical geometry and boundary patches stay the same.

Grading along shared directions must also agree. To use geometric grading with ratio 3 along +y in the left block:

| Block | `grading_type` | `grading` |
|---|---|---|
| `left`: eta = +y | `["uniform", "geometric", "uniform"]` | `[1.0, 3.0, 1.0]` |
| Rotated `right`: zeta = -y | `["uniform", "uniform", "geometric"]` | `[1.0, 1.0, 0.3333333333333333]` |

When an axis direction is reversed, use these equivalent distributions:

| Original distribution | Same physical spacing along the reversed axis |
|---|---|
| `uniform` | `uniform` |
| `geometric`, r | `geometric`, `1/r` |
| `tanh_start`, delta | `tanh_end`, same delta |
| `tanh_end`, delta | `tanh_start`, same delta |
| `tanh_both`, delta | `tanh_both`, same delta |

For N cells, reversal means `s_reverse[i] = 1 - s_forward[N - i]`. The generator compares node positions; matching type names alone does not establish matching spacing.

## 6. Contracts and limitations

### Geometry and connections

| Requirement | Generator check |
|---|---|
| Every macro vertex belongs to at least one block | Unused vertices are rejected |
| Each block references 8 distinct, existing vertices | Duplicate and out-of-range indices are rejected |
| Each set of eight vertices describes one block | A second block with the same vertex set is rejected, regardless of ordering |
| The block map preserves positive orientation | The Jacobian is sampled at 27 points, as described below |
| Shared edges have matching cell counts and node positions | Reversed directions are accounted for; normalized positions use tolerance `64 × epsilon(double)` |
| Shared faces have the same cyclic perimeter and matching subdivisions | Rotations, swapped axes, and reversed axes are accounted for |
| A macro face belongs to at most two blocks | A third block sharing that face is rejected |
| Each explicitly listed boundary face is a complete outer face of one block | Nonexistent faces and internal block interfaces are rejected |
| An outer face is assigned explicitly at most once | Repetition within one patch or across patches is rejected |
| Every outer face has an output patch | Unlisted faces are collected in one automatic patch; rank 0 prints a warning |

**Blocks must form a conforming mesh:** neighbours meet on complete faces using common vertex indices and matching subdivisions. Partial face interfaces, hanging nodes, and coincident surfaces with different indices are not connected automatically. Block interiors must not overlap; the generator does not perform a general geometric intersection search.

### Jacobian check

For the trilinear map `p(xi, eta, zeta)`, let `p_xi`, `p_eta`, and `p_zeta` denote its three partial derivatives. The check uses their normalized scalar triple product:

```text
q = ((p_xi / |p_xi|) × (p_eta / |p_eta|)) · (p_zeta / |p_zeta|)
```

At every point in `{0, 0.5, 1}^3`, each derivative must have a finite, nonzero length, and q must be finite and **strictly greater than `64 × epsilon(double)`**, approximately `1.42e-14`. Normalization makes the tolerance independent of physical units.

This samples the macro block. **It does not prove positive Jacobians between sample points or validate every generated cell.** Strongly deformed blocks need a separate check of the resulting mesh quality.

### Numerical and size limits

- `scale`, scaled coordinates, and all grading parameters must be finite. `NaN` and `Inf` are invalid.
- Geometric and tanh parameters must be strictly positive. A uniform parameter is ignored but must still be finite.
- Node positions must strictly increase: `0 = s[0] < s[1] < ... < s[N] = 1`. Generation fails if rounding in `double` collapses adjacent nodes.
- Computed sizes, sums, and products must fit `size_t`. On each axis, `N + 1` must fit the maximum size of a `double` array.

Output indices use the limit `L = 2^63 - 2`:

| Quantity | Limit |
|---|---|
| Total unique nodes | `Nnodes + 1 <= L` |
| Total volume cells | `8 × Ncells <= L` |
| Boundary quads in each patch, including the automatic patch | `4 × Nquads_patch <= L` |
| All volume and written boundary elements | `Ncells + sum(Nquads_patch) <= L` |

Passing size checks does not guarantee sufficient memory. Configuration, allocation, MPI, and CGNS errors handled by the generator terminate the MPI job through `mpi::fatal`.

## 7. CGNS output

| Object | Contents |
|---|---|
| `Base_3D` / `Zone1` | One three-dimensional unstructured zone containing all blocks |
| `CoordinateX`, `CoordinateY`, `CoordinateZ` | Node coordinates stored as `double`, with `scale` already applied |
| `Volume_HEXA` | All `HEXA_8` volume cells |
| Element section named after each boundary patch | That patch's outward-oriented `QUAD_4` elements |
| `BC_t` named after each patch, including the automatic patch | Boundary element range; `GridLocation = FaceCenter`, type `BCTypeUserDefined` |

CGNS node IDs start at **1**. Volume element IDs are `1 ... Ncells`; boundary element IDs follow them. These IDs are assigned by the generator. TOML references still use the original macro vertex indices starting at **0**.

Names such as `inlet`, `outlet`, and `walls` identify face groups. Physical boundary conditions for those groups are assigned in the solver.

Writing replaces the selected output file. A failed write may leave an incomplete file.

## 8. Fixing configuration errors

| Message / cause | What to change |
|---|---|
| `unknown key` | Check the field name and the section it belongs to |
| `Shared edge ... cell count mismatch` | Match the cell counts along the shared edge; permute `cells` entries when rotating a block |
| `Shared edge has incompatible grading` | Match node positions: check the axis direction, reciprocal geometric ratio, or `tanh_start` / `tanh_end` reversal |
| `Grading collapses adjacent nodes` | Move geometric r toward 1 or reduce tanh delta. Reduce the cell count if necessary |
| `inverted or degenerate trilinear map` | Check the eight-corner order and geometry; reordering alone cannot fix every deformed block |
| `internal block face` | Remove the internal interface from boundary patches |
| `face ... absent` / `perimeter` | Use four shared indices of one existing outer block face, ordered around its perimeter |
| Warning: `unassigned outer macro face(s)` | Generation continues with an automatic patch. Add explicit patches if those faces need different boundary conditions |
| `Every macro vertex must be used` | Remove the unused vertex or include it in the intended block; update indices after removing a vertex |