using TensorKit
using ClusterExpansions
using PEPSKit
using Plots
using TensorKitTensors

# Set up the Heisenberg model
(Jx, Jy, Jz, z) = (1.0, 1.0, 1.0, 0.0)

# Parameters in the truncation scheme
Dcut = 4
trunc_alg = NoEnvTruncation(truncrank(Dcut))
χenv = 16 # Environment bond dimension used in the calculation of expectation values

# Set up time evolution algorithm
β₀ = 1e-1
Δβ = 1e-1
maxiter = 10
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)
ctm_alg = SimultaneousCTMRG(; maxiter = 250, verbosity = 0)

plt = plot()
for spin_symmetry in [Trivial, SU2Irrep]
    ce_alg = heisenberg_operators(Jx, Jy, Jz, z; spin_symmetry, p = 3, T = ComplexF64)
    observables = PEPO_observables([SpinOperators.S_exchange(spin_symmetry)], ctm_alg)
    observable = (O,i) -> ClusterExpansions.calculate_observables(O, χenv, observables)
    
    # Perform the actual time evolution.
    # `βs` is a list of times/βs at which the expectation values are computed
    # 'expvals' are the expectation values. Every element of `expvals` is a list of the expectation values at the corresponding time
    # `Os` is a list of PEPO density operators
    βs, expvals, = time_evolve(ce_alg, time_alg, trunc_alg, observable)

    # Extract the expectation values
    energies = [e[1] for e in expvals]


    scatter!(plt, Float64.(βs), real.(energies), label = "Symmetry = $(spin_symmetry)")
end
xlabel!("T")
ylabel!("Energy")
title!("Heisenberg model")
display(plt)
