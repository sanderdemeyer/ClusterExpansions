using TensorKit
using PEPSKit
using ClusterExpansions
using Test
using Random

Random.seed!(439410384)

ce_alg_bosonic = ising_operators(1.0, 2.5, 0.0)
ce_alg_fermionic = spinless_fermion_operators(1.0, -2.5, 0.0)
models = [ce_alg_bosonic ce_alg_fermionic]

β = 0.01
Δβ = 0.01

ctm_alg = SimultaneousCTMRG(; maxiter = 1000, verbosity = 1)

χenv = 16
envspaces_fidel = [ℂ^χenv, Vect[fℤ₂](0 => div(χenv,2), 1 => div(χenv,2))]
envspaces_trunc = [ℂ^χenv, Vect[fℤ₂](0 => div(χenv,2), 1 => div(χenv,2))]
truncspaces = [ℂ^3, Vect[fℤ₂](0 => 2, 1 => 1)]

for (ce_alg, envspace_fidel, envspace_trunc, truncspace) in zip(models, envspaces_fidel, envspaces_trunc, truncspaces)
    trscheme = truncdim(3)
    trunc_alg_noenv = NoEnvTruncation(trscheme)
    trunc_alg_approxenv = ApproximateEnvTruncation(ctm_alg, envspace_trunc, trscheme)
    trunc_alg_vomps = VOPEPO_CTMRG(ctm_alg, envspace_trunc, truncspace, 6, 6)
    trunc_algs = [trunc_alg_noenv, trunc_alg_approxenv, trunc_alg_vomps]
    O1 = evolution_operator(ce_alg, β)
    O2 = evolution_operator(ce_alg, Δβ)
    O3 = evolution_operator(ce_alg, β + Δβ)
    
    ϵ_base = 1 - fidelity(O1, O3, ctm_alg, envspace_fidel)
    for trunc_alg in trunc_algs

        O_evolved, = approximate_state((O1, O2), trunc_alg);

        ϵ = 1 - fidelity(O_evolved, O3, ctm_alg, envspace_fidel)

        println("errors are $ϵ and $ϵ_base")
        # @test ϵ < 1e-3
        # @test ϵ_base > ϵ
    end
end