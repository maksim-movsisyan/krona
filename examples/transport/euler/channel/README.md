# Steady supersonic channel

The channel is one metre long with a square cross-section. Its uniform steady
solution has pressure $100{,}000$ Pa, temperature $300$ K and velocity
$(650,0,0)$ m/s. The inlet fixes that state, the outlet is supersonic,
and the lateral patches are symmetry boundaries. The resulting density
is $100000/(287.05\cdot300)\approx1.16124$ kg/m³.

`--case channel` initializes a five-percent pressure bump in the middle of the
channel while leaving the reference temperature and velocity unchanged. The
steady solver advances the perturbed field by implicit pseudo-time iterations
towards the uniform solution. The bump is defined in
[initial_field.hpp](../../../../apps/euler_steady/initial_field.hpp).

## Generate, partition, and run

Run from `examples/transport/euler` with the KRONA executables on your `PATH`.

```sh
krona_mesh_generator channel/mesh.toml channel/mesh.cgns
mpirun -np 2 krona_mesh_partitioner channel/mesh.cgns channel/mesh.h5
mpirun -np 2 krona_euler_steady channel/mesh.h5 channel/euler.toml channel/bc.toml --case channel -vv
```

[mesh.toml](mesh.toml) defines a $32\times3\times3$ hexahedral mesh.
Use the same MPI size for partitioning and solving. The final fields are
in `channel/vtk/euler.pvtu`, including `density`, `pressure`,
`temperature` and `velocity`. The iteration log reports the norm of
the spatial residual. See [the steady Euler guide](../../../../docs/EULER-STEADY.md)
for the configuration and the pseudo-time discretization.

Set `reconstruction = "second_order"` in `channel/euler.toml` to run the
same case with linear reconstruction. Set `limiter = "barth_jespersen"` to
enable the componentwise limiter. Use `--vtu channel/vtk_second` to keep
both outputs. The steady reference is uniform for both schemes. A spatial
order study needs a smooth, spatially varying reference solution and
multiple mesh resolutions.