using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using PEPSKit
using Plots

V = -2.5
Dcut = 3
χenv = 12
dt = 1e-3

trunc_alg_SU = truncdim(Dcut) & truncerr(1e-10)
trunc_alg_CE = NoEnvTruncation(trunc_alg_SU)

β₀ = 0.05
Δβ = 0.05
max_beta = 1.2
maxiter = ceil(Int, (max_beta - β₀) / Δβ)
time_alg_CE = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 0)

maxiter = ceil(Int, max_beta / Δβ)
time_alg_SU = (dt = dt, Δt = Δβ, maxiter = maxiter)

βs_CE, ns_CE, ξs_CE, δs_CE, As_CE = data_generation_SF_CE(time_alg_CE, trunc_alg_CE, χenv; V)
βs_SU, ns_SU, ξs_SU, δs_SU, As_SU = data_generation_SF_SU(time_alg_SU, trunc_alg_SU, χenv; V)

plt = scatter()
scatter!(βs_CE, real.(ns_CE), label = "CE")
scatter!(βs_SU, real.(ns_SU), label = "SU")
xlabel!("β")
ylabel!("n")
title!("Spinless fermion model for V = $V")
display(plt)
