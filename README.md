# KRONA

KRONA (**K**okkos-based **R**eusable **O**perators for **N**umerical **A**pplications) is an experimental framework for distributed finite-volume simulations on unstructured meshes.

It uses **MPI** for distributed-memory parallelism, **Kokkos** for portable CPU/GPU execution, and **Trilinos** for sparse linear algebra.

Currently implemented:

- Distributed mesh partitioning and preprocessing.
- Mesh geometry, fields, halo exchange, and gradient reconstruction.
- Scalar and physical boundary conditions.
- Scalar CSR and block BSR matrices with configurable solvers and preconditioners.
- A steady scalar diffusion solver.

The project is under active development.