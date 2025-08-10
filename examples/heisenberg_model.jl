using TensorKit
using ClusterExpansions
using PEPSKit
using Plots

# Set up the Heisenberg model
(Jx, Jy, Jz, z) = (1.0, 1.0, 1.0, 0.0)
spin_symmetry = U1Irrep
ce_alg = heisenberg_operators(Jx, Jy, Jz, z; spin_symmetry = spin_symmetry, p = 3)
Jx = 1.0
Jy = 1.0
Jz = 1.0
z = 0.0 # explicit symmetry-breaking fielda along the z-axis
model_param = (Jx, Jy, Jz, z)

χenv = 16 # Environment bond dimension used in the calculation of expectation values

# Parameters in the truncation scheme
Dcut = 4
trunc_alg = NoEnvTruncation(truncdim(Dcut))

# Set up time evolution algorithm
β₀ = 1e-1
Δβ = 1e-1
maxiter = 10
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

# Define observables
H = heisenberg_XYZ(ComplexF64, spin_symmetry, InfiniteSquare(); Jx, Jy, Jz)
ctm_alg = SimultaneousCTMRG(; maxiter = 250, verbosity = 0)
observables = PEPO_observables([H], ctm_alg)
observable = O -> ClusterExpansions.calculate_observables(O, χenv, observables)

# Perform the actual time evolution.
# `βs` is a list of times/βs at which the expectation values are computed
# 'expvals' are the expectation values. Every element of `expvals` is a list of the expectation values at the corresponding time
# `Os` is a list of PEPO density operators
βs, expvals, Os = time_evolve(ce_alg, time_alg, trunc_alg, observable)

# Extract the expectation values
energies = [e[1] for e in expvals]

# Plot them
plt = scatter(Float64.(βs), real.(energies), label = "Energies CE")
xlabel!("T")
ylabel!("Exchange")
title!("Heisenberg model")
display(plt)
