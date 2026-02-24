using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using PEPSKit
using Plots

g = 2.5
Dcut = 6
χenv = 12

trunc_alg_CE = NoEnvTruncation(truncdim(Dcut))

β₀ = 0.05
Δβ = 0.05
max_beta = 1.0
maxiter = ceil(Int, (max_beta - β₀) / Δβ)
time_alg_CE = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

ce_alg = ising_operators(1.0, g, 0.0; T = Float64, symmetry = "C4")
t_alg = ising_operators_Trotter(1.0, g, 0.0; T = Float64)

# Define observables
vumps_alg = VUMPS(; maxiter = 100, verbosity = 0)
ctm_alg = SequentialCTMRG(; maxiter = 300)
observables = PEPO_observables([SpinOperators.σᶻ(), SpinOperators.σˣ(), :spectrum], vumps_alg)
observable = (O, i) -> ClusterExpansions.calculate_observables(O, χenv, observables)

βs, expvals_CE, As_CE = time_evolve(ce_alg, time_alg_CE, trunc_alg_CE, observable; normalizing = true);
βs, expvals_T, As_T = time_evolve(t_alg, time_alg_CE, trunc_alg_CE, observable; normalizing = true);

# Extract the expectation values
mzs_CE = [e[1] for e in expvals_CE]
mxs_CE = [e[2] for e in expvals_CE]
ξs_CE = [e[3][1] for e in expvals_CE]
δs_CE = [e[3][2] for e in expvals_CE]

mzs_T = [e[1] for e in expvals_T]
mxs_T = [e[2] for e in expvals_T]
ξs_T = [e[3][1] for e in expvals_T]
δs_T = [e[3][2] for e in expvals_T]

plt = scatter()
scatter!(βs, abs.(mzs_CE), label = "CE - D = $(Dcut)")
scatter!(βs, abs.(mzs_T), label = "Trotter - D = $(Dcut)")
vline!([1 /  1.2736], label = "QMC")
xlabel!("β")
ylabel!("magnetization")
title!("Ising model for g = $g")
display(plt)

plt = scatter()
scatter!(βs, ξs_CE, label = "CE - D = $(Dcut)")
scatter!(βs, ξs_T, label = "Trotter - D = $(Dcut)")
vline!([1 /  1.2736], label = "QMC")
xlabel!("β")
ylabel!("correlation length ξ")
title!("Ising model for g = $g")
display(plt)
