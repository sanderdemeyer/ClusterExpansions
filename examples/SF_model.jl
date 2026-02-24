using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using PEPSKit
using Test

V = -2.5
Dcut = 8
χenv = 20

max_beta = 1.0
β₀ = 0.01
Δβ = 0.01
maxiter = ceil((max_beta - β₀)/Δβ)

time_alg = UniformTimeEvolution(β₀, Δβ, maxiter)
trunc_alg = NoEnvTruncation(truncdim(Dcut))
ce_alg = spinless_fermion_operators(1.0, V, 0.0; symmetry = nothing, T = Float64)

# Define observables
vumps_alg = VUMPS(; maxiter = 100, verbosity = 0)
observables = PEPO_observables([FermionOperators.f_num(), :spectrum], vumps_alg)
obs_function = (O,i) -> ClusterExpansions.calculate_observables(O, χenv, observables)

βs, expvals, As = time_evolve(ce_alg, time_alg, trunc_alg, obs_function)

# Extract the expectation values
ns = [e[1] for e in expvals]
ndifs = [abs(e[1] - 0.5) for e in expvals]
ξs = [e[2][1] for e in expvals]
δs = [e[2][2] for e in expvals]

# QMC results up to one standard deviation
βc_QMC = 0.85
βc_QMC_plus = 1/(1.18 + 0.07)
βc_QMC_min = 1/(1.18 - 0.07)

plt = scatter()
scatter!(βs, ndifs, label = "Cluster Expansion")
vline!([βc_QMC_min], label = nothing, color = "black", linestyle = :dash)
vline!([βc_QMC_plus], label = "QMC", color = "black", linestyle = :dash)
xlabel!("β")
ylabel!("|0.5 - n|")
title!("Spinless Fermion model with V = $V")
display(plt)

# Exclude first data point - Too high temperature makes the calculation of the environment untrustworthy.
plt = scatter()
scatter!(βs[2:end], ξs[2:end], label = "Cluster Expansion")
vline!([βc_QMC_min], label = nothing, color = "black", linestyle = :dash)
vline!([βc_QMC_plus], label = "QMC", color = "black", linestyle = :dash)
xlabel!("β")
ylabel!("correlation length ξ")
title!("Spinless Fermion model with V = $V")
display(plt)
