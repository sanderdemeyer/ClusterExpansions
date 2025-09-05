using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using PEPSKit
using Plots
using DelimitedFiles, JLD2


# Set up the Ising model
spin_symmetry = SU2Irrep
J1 = 1.0
J2 = 1.5
h = 0.0 # explicit symmetry-breaking fielda along the z-axis
p = 3
ce_alg = J1J2_operators(J1, J2, h; T = ComplexF64, symmetry = "C4", spin_symmetry, p, verbosity = 0);

T = ComplexF64
(Nr, Nc) = (1, 1) # Unit cell size
lattice = InfiniteSquare(Nr, Nc)
H = LocalOperator(
    fill(SpinOperators.spin_space(spin_symmetry), (Nr, Nc)),
    (neighbor => rmul!(SpinOperators.S_exchange(T, spin_symmetry), J1) for neighbor in PEPSKit.nearest_neighbours(lattice))...,
    (neighbor => rmul!(SpinOperators.S_exchange(T, spin_symmetry), J2) for neighbor in PEPSKit.next_nearest_neighbours(lattice))...,
)

χenv = 30 # Environment bond dimension used in the calculation of expectation values

# Parameters in the truncation scheme
Dcut = 6
trunc_alg = NoEnvTruncation(truncdim(Dcut); verbosity = 0)

# Set up time evolution algorithm
β₀ = 0.05
Δβ = 0.05
maxiter = 10
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

# Define observables
ctm_alg = SimultaneousCTMRG(; maxiter = 1000)
observables = PEPO_observables([H], [ctm_alg]; funcs = [x -> x])
observable = O -> ClusterExpansions.calculate_observables(O, χenv, observables)

# Perform the actual time evolution.
# `βs` is a list of times/βs at which the expectation values are computed
# 'expvals' are the expectation values. Every element of `expvals` is a list of the expectation values at the corresponding time
# `Os` is a list of PEPO density operators
βs, expvals, Os = time_evolve(ce_alg, time_alg, trunc_alg, observable);

# Extract the expectation values
Es = [e[1] for e in expvals]

# Plot them
plt = scatter(βs, real.(Es), label = "<X>")
xlabel!("T")
ylabel!("Energy")
title!("J1-J2 model")
display(plt)
