# ClusterExpansions.jl

[![CI](https://github.com/sanderdemeyer/ClusterExpansions/actions/workflows/CI.yml/badge.svg)](https://github.com/sanderdemeyer/ClusterExpansions/actions/workflows/CI.yml)
[![Phys. Rev. B](https://img.shields.io/badge/Phys.%20Rev.%20B-114%2C%20175113-blue.svg)](https://journals.aps.org/prb/abstract/10.1103/16gn-ysxt)
[![arXiv](https://img.shields.io/badge/arXiv-2602.22113-b31b1b.svg)](https://arxiv.org/abs/2602.22113)

A Julia package for constructing **projected entangled-pair operator (PEPO)** approximations of the imaginary-time evolution operator `exp(-βH)` using **cluster expansions**, and for using them to simulate **finite-temperature** (and ground-state) properties of two-dimensional quantum lattice models, both **bosonic** and **fermionic**.

Compared to the usual Suzuki–Trotter decomposition, the cluster expansion keeps the result extensive by organising contributions according to the size of connected clusters. As a result it is more accurate for a given step size `Δβ`, extends naturally to higher orders, and preserves the internal and lattice symmetries of the Hamiltonian, including full translation invariance.

The package is built on top of the [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) ecosystem ([PEPSKit.jl](https://github.com/QuantumKitHub/PEPSKit.jl), [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl), [TensorKitTensors.jl](https://github.com/QuantumKitHub/TensorKitTensors.jl)), so symmetric and fermionic tensors are supported out of the box.

## Features

- **Cluster-expansion PEPOs** for `exp(-βH)` of nearest-neighbour Hamiltonians on the square lattice and cubic lattice, up to a chosen cluster size `p` (including loop clusters).
- **Built-in models**: transverse-field Ising, Heisenberg (XXZ / XXX), J1–J2, spinless fermions (t–V), t–J, Hubbard and Bose–Hubbard.
- **Internal symmetries**: `Z2Irrep`, `U1Irrep`, `SU2Irrep` and `FermionParity`, 
- **Spatial symmetries**: `"C4v"` in the two-dimensional case and `"Oh"` in the three-dimensional case.
- **Imaginary-time evolution** by repeatedly multiplying and truncating PEPOs, with several schedules (uniform steps, squaring, time-dependent couplings) and truncation schemes:
  - `NoEnvTruncation`: local truncation without environment,
  - `ApproximateEnvTruncation`: truncation using an approximate CTMRG environment,
  - `VOPEPO_CTMRG` / `VOPEPO_VUMPS`: variational PEPO compression (only in separate branch).
- **Ground-state search** through long imaginary-time evolution (`UniformGroundStateTimeEvolution`, `SquaringGroundStateTimeEvolution`).
- **Observables in 2D** from CTMRG or VUMPS environments: local expectation values, correlation length and transfer-matrix spectrum.
- **Observables in 3D** from 3DCTMRG or variational boundary PEPS (vbPEPS) environments: local expectation values.

## Installation

The package is not (yet) registered. Install it directly from GitHub in the Julia REPL:

```julia
using Pkg
Pkg.add(url = "https://github.com/sanderdemeyer/ClusterExpansions")
```

The package is tested on Julia 1.12. The examples also use [Plots.jl](https://github.com/JuliaPlots/Plots.jl), which you need to install separately.

## Quick start

The example below computes the magnetisation and correlation length of the 2D transverse-field Ising model as a function of inverse temperature `β`.

```julia
using TensorKit, TensorKitTensors
using MPSKit, PEPSKit
using ClusterExpansions

# Model: H = -J Σ σᶻσᶻ - gJ Σ σˣ - zJ Σ σᶻ
J, g, z = 1.0, 2.5, 0.0
ce_alg = ising_operators(J, g, z; T = Float64)

# Imaginary-time schedule: start at β₀ and take steps of Δβ
β₀, Δβ, maxiter = 0.05, 0.05, 19
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

# Truncate the PEPO bond dimension back to Dcut after every step
Dcut = 4
trunc_alg = NoEnvTruncation(truncrank(Dcut))

# Observables, evaluated with environment bond dimension χenv
χenv = 12
vumps_alg = VUMPS(; maxiter = 100, verbosity = 0)
observables = PEPO_observables([SpinOperators.σᶻ(), SpinOperators.σˣ(), :spectrum], vumps_alg)
obs_function = (O, i) -> calculate_observables(O, χenv, observables)

βs, expvals, Os = time_evolve(ce_alg, time_alg, trunc_alg, obs_function)

mz = [e[1] for e in expvals]     # ⟨σᶻ⟩
mx = [e[2] for e in expvals]     # ⟨σˣ⟩
ξ  = [e[3][1] for e in expvals]  # correlation length
```

### Building a single PEPO

If you only need the PEPO approximation of `exp(-βH)` for a given `β`, e.g.

```julia
ce_alg = heisenberg_operators(1.0, 1.0, 1.0, 0.0; spin_symmetry = SU2Irrep, p = 3, T = ComplexF64)
O = evolution_operator(ce_alg, 0.1)  # local PEPO tensor approximating exp(-0.1 H)
```

### Custom Hamiltonians

Any nearest-neighbour Hamiltonian can be expanded by passing its two-site and one-site terms directly:

```julia
ce_alg = ClusterExpansion(twosite_op, onesite_op)
```

The expansion is solved in high precision (`BigFloat` by default) and can be converted to lower precision (double precision by default).

## Examples

The [`examples`](examples) folder contains complete scripts:

| Script | Description |
| --- | --- |
| [`ising_model.jl`](examples/ising_model.jl) | Finite-temperature transition of the 2D transverse-field Ising model; cluster expansions compared with QMC. |
| [`SF_model.jl`](examples/SF_model.jl) | Charge-density-wave transition of spinless fermions with nearest-neighbour repulsion. |
| [`heisenberg_model.jl`](examples/heisenberg_model.jl) | Energy of the Heisenberg model with and without `SU2` symmetry. |
| [`ising_model_3D.jl`](examples/ising_model_3D.jl) | 3D Ising model on the cubic lattice; cluster expansion vs. simple update. |

## Citation

If you use this package in your research, please cite:

```bibtex
@article{demeyer2026lowering,
  title     = {Lowering the temperature of two-dimensional fermionic tensor networks with cluster expansions},
  author    = {De Meyer, Sander and Ueda, Atsushi and He, Yuchi and Bultinck, Nick and Haegeman, Jutho},
  journal   = {Phys. Rev. B},
  volume    = {114},
  pages     = {175113},
  year      = {2026},
  publisher = {American Physical Society},
  doi       = {10.1103/16gn-ysxt},
  url       = {https://journals.aps.org/prb/abstract/10.1103/16gn-ysxt},
  eprint    = {2602.22113},
  archivePrefix = {arXiv}
}
```

The cluster-expansion construction itself was introduced in B. Vanhecke, L. Vanderstraeten and F. Verstraete, *Symmetric cluster expansions with tensor networks*, [Phys. Rev. A **103**, L020402 (2021)](https://arxiv.org/abs/1912.10512).

## Contributing

Bug reports, questions and pull requests are welcome. Please open an [issue](https://github.com/sanderdemeyer/ClusterExpansions/issues) to get in touch.
