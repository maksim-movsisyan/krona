# Heated lid-driven cavity

Laminar ideal gas fills a 1 cm cube. The lid at `y = 0.01 m` moves in `+x`
at 90 m/s and is held at 320 K. The bottom wall is stationary at 300 K.
The four side walls are stationary and adiabatic. Based on the initial gas
state, the lid speed and cube side give `Re ≈ 389` and `Ma ≈ 0.26`.

The default mesh has `64 × 64 × 64 = 262,144` cells. The solver uses HLLC,
limited linear reconstruction, face-based least-squares gradients, molecular
viscosity and heat conduction. The case exercises convection, wall shear,
and heat transfer in the steady compressible solver.

## Generate and run

From this directory, with the KRONA executables on `PATH`:

```sh
krona_mesh_generator mesh.toml mesh.cgns
mpirun -np 1 krona_mesh_partitioner mesh.cgns mesh-1.h5 --reorder RCM
mpirun -np 1 krona_flow_steady mesh-1.h5 flow.toml bc.toml -v
```

For multiple ranks, rerun the partitioner and solver with the same rank count.
Output is disabled by default. Add `--vtu vtk` for a run whose solution you
want to inspect. For a short profiling capture, lower
`flow.pseudo_time.maximum_iterations` in a copy of `flow.toml`. The application
returns a failure status if it reaches that limit before convergence.

## Wall-normal spacing

The uniform mesh has a first-cell thickness of `156.25 µm` and a first
cell-centre distance of `78.125 µm`. To cluster toward both the lid and the
bottom, add these fields to `[[blocks]]`:

```toml
grading_type = ["uniform", "tanh_both", "uniform"]
grading = [1.0, 2.0, 1.0]
```

At 64 cells along `y`, this gives a first-cell thickness of about `24.3 µm`
and a first cell-centre distance of about `12.2 µm`. A strength of `1.482`
gives a first-cell thickness of about `50 µm`. For another thickness, solve
for the strength `b` by bisection using the generator's first-cell expression
`dy1 = L/2 * (1 - tanh(b*(1 - 2/N))/tanh(b))`.

For a grid-resolution study, compare velocity and temperature profiles in
the midplane and heat flux or wall shear over a central region of the lid.
Keep the sampling region fixed. The abrupt change in prescribed velocity
where the lid meets the side walls makes the corner shear peak sensitive to
refinement. Allow the spatial residual to converge on each grid before
comparing the results.
