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
ce_alg = spinless_fermion_operators(1.0, V, 0.0; symmetry = nothing, T = BigFloat)

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

βc = βs[argmax(ξs)]
βc_QMC = 0.85

check_ndifs = [(β > βc) ? (ndif > 0.25) : (ndif < 0.25) for (β, ndif) in zip(βs, ndifs)]

@test sum(check_ndifs) > 0.98 * length(check_ndifs)
@test abs(βc - βc_QMC) < 0.06
