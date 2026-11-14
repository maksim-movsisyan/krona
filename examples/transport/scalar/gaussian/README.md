# Advection and diffusion of a Gaussian pulse

The source is zero, the fluid moves at constant speed $u=0.8$ along $x$,
and the initial pulse is uniform across the channel cross section. Its
peak is one, its centre is $x_0=1.25$, and its initial standard deviation
is $\sigma_0=0.12$. With $\alpha=\Gamma/\rho$, the one-dimensional
constant-coefficient equation and initial data are

$$
\partial_t\phi+u\,\partial_x\phi=\alpha\,\partial_{xx}\phi,\qquad
\phi(x,0)=\exp\!\left[-\frac{(x-x_0)^2}{2\sigma_0^2}\right].
$$

## Derive the analytical reference

Use the moving coordinate $\xi=x-x_0-ut$ and write
$\phi(x,t)=g(\xi,t)$. By the chain rule,

$$
\partial_t\phi=\partial_t g-u\,\partial_\xi g,\qquad
\partial_x\phi=\partial_\xi g,\qquad
\partial_{xx}\phi=\partial_{\xi\xi}g.
$$

The terms proportional to $u$ cancel, leaving
$\partial_t g=\alpha\,\partial_{\xi\xi}g$. Try a Gaussian with
time-dependent amplitude $a(t)$ and variance $v(t)$:

$$
g(\xi,t)=a(t)\exp\!\left[-\frac{\xi^2}{2v(t)}\right],
\qquad a(0)=1,\quad v(0)=\sigma_0^2.
$$

Differentiating and dividing the heat equation by $g$ gives

$$
\frac{a'}{a}+\frac{\xi^2}{2v^2}v'
=\alpha\left(-\frac{1}{v}+\frac{\xi^2}{v^2}\right).
$$

Match the constant and $\xi^2$ terms. This gives $v'=2\alpha$ and
$a'/a=-\alpha/v$. Consequently $v=\sigma_0^2+2\alpha t$ and
$a\sqrt{v}=\sigma_0$. Returning to $x$ gives

$$
\boxed{\phi(x,t)=
\frac{\sigma_0}{\sqrt{\sigma_0^2+2\alpha t}}
\exp\!\left[-\frac{(x-x_0-ut)^2}{2(\sigma_0^2+2\alpha t)}\right].}
$$

Convection moves the centre to $x_0+ut$. Diffusion increases the variance,
and the amplitude decreases so the integral over the whole line stays
constant. The app evaluates this expression in
[analytic_solution.hpp](../../../../apps/scalar_transport/analytic_solution.hpp).

## Boundary conditions in the numerical run

The derived formula is exact on the infinite line. The numerical mesh
covers $0\le x\le5$, with `FIXED_VALUE = 0` at the inlet and zero normal
diffusive gradient at the outlet. The side walls have zero normal
gradient and exactly zero convective mass flux. The longitudinal
Gaussian satisfies the side-wall conditions. At the end faces the
Gaussian value or gradient is only approximately zero. The supplied
domain and run time keep those tails negligible for this comparison.

The configuration uses $\rho=1$, $\Gamma=0.002$, and 50 steps of
$\Delta t=0.01$. At $t=0.5$ the reference centre is $1.65$ and its standard
deviation is about $0.128$. The numerical run uses BDF2 after one
Backward Euler startup step and TVD reconstruction with Van Leer limiting.

## Generate, partition, and run

Use the KRONA mesh generator, mesh partitioner, and scalar transport
executables on your `PATH`. Run from `examples/transport/scalar`.

```sh
krona_mesh_generator gaussian/mesh.toml gaussian/mesh.cgns
mpirun -np 2 krona_mesh_partitioner gaussian/mesh.cgns gaussian/mesh.h5
mpirun -np 2 krona_scalar_transport gaussian/mesh.h5 gaussian/transport.toml gaussian/bc.toml \
    --case gaussian -v
```

[mesh.toml](mesh.toml) defines a uniform $200\times4\times4$ hexahedral mesh.
The first command regenerates the included `mesh.cgns`. Use the same MPI
size for partitioning and solving. The solver writes a snapshot every 10
accepted steps. Open `gaussian/vtk/step_50/scalar.pvtu` for
`phi`, the analytical reference at each cell centroid `phi_exact`, and
the signed difference `phi_error = phi - phi_exact`. The solver also
prints the volume-weighted RMS and maximum absolute errors. The
[transport guide](../../../../docs/TRANSPORT-SCALAR.md) covers the build
and configuration options.
