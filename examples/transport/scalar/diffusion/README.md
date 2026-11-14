# Diffusion of a sine mode

The fluid is at rest, the source is zero, and $\rho$ and $\Gamma$ are
constant. The channel is $[0,L]\times[0,0.1]\times[0,0.1]$ with $L=1$.
The initial field varies only along $x$, and the side walls have zero
normal gradient. A solution depending only on $x$ satisfies the side-wall
conditions automatically.

## Analytical solution

Divide the transport equation by $\rho$ and set $\alpha=\Gamma/\rho$.
The problem along the channel is

$$
\partial_t\phi=\alpha\,\partial_{xx}\phi,\qquad
\phi(0,t)=\phi(L,t)=0,\qquad
\phi(x,0)=\sin\!\left(\frac{\pi x}{L}\right).
$$

Both end faces use zero Dirichlet data. The inlet reads this value from
[inlet.csv](inlet.csv), which is constant over the patch.

The initial sine is an eigenfunction of the second derivative. Its
second derivative is the same sine multiplied by $-(\pi/L)^2$. Seek a
solution of the form $\phi(x,t)=a(t)\sin(\pi x/L)$. Substitution into the
heat equation gives

$$
a'(t)=-\alpha\left(\frac{\pi}{L}\right)^2a(t),\qquad a(0)=1.
$$

Solving this ODE gives the exact solution for the channel:

$$
\boxed{\phi(x,t)=\sin\!\left(\frac{\pi x}{L}\right)
             \exp\!\left[-\alpha\left(\frac{\pi}{L}\right)^2 t\right].}
$$

The shape stays fixed and the amplitude decays exponentially. With $L=1$,
this is the `diffusion` branch in
[analytic_solution.hpp](../../../../apps/scalar_transport/analytic_solution.hpp).
The supplied configuration uses $\rho=1$, $\Gamma=0.05$, and 30 steps of
$\Delta t=0.01$, so the comparison is at $t=0.3$. BDF2 starts with one
Backward Euler step.

## Generate, partition, and run

Use the KRONA mesh generator, mesh partitioner, and scalar transport
executables on your `PATH`. Run from `examples/transport/scalar` so the
relative CSV path in [bc.toml](bc.toml) resolves on rank zero.

```sh
krona_mesh_generator diffusion/mesh.toml diffusion/mesh.cgns
mpirun -np 2 krona_mesh_partitioner diffusion/mesh.cgns diffusion/mesh.h5
mpirun -np 2 krona_scalar_transport diffusion/mesh.h5 diffusion/transport.toml diffusion/bc.toml \
    --case diffusion -v
```

[mesh.toml](mesh.toml) defines a uniform $40\times4\times4$ hexahedral mesh.
The first command regenerates the included `mesh.cgns`. Use the same MPI
size for partitioning and solving. The solver writes a snapshot every 10
accepted steps. Open `diffusion/vtk/step_30/scalar.pvtu` for
`phi`, the analytical value at each cell centroid `phi_exact`, and the
signed difference `phi_error = phi - phi_exact`. The solver also prints
the volume-weighted RMS and maximum absolute errors. The
[transport guide](../../../../docs/TRANSPORT-SCALAR.md) covers the build and
configuration options.
