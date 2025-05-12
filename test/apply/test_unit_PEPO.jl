using Random
using TensorKit, PEPSKit, KrylovKit
using ClusterExpansions
import PEPSKit: rmul!, σᶻᶻ, σˣ
using Test

function make_peps(d, D)
    pspace = ℂ^d
    vspace = ℂ^D
    return randn(T, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
end

function make_pepo(d, D; perturbation = 1e-1)
    pspace = ℂ^d
    vspace = ℂ^D
    unit = permute(id(T, pspace ⊗ vspace ⊗ vspace), ((1,4),(5,6,2,3)))
    O = randn(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
    return unit + perturbation*norm(unit)/norm(O) * O
end

Random.seed!(84958430958)

T = Complex{Float64}
χenv = 45
χenv_fidel = 60
Dcut = 2
envspace = ℂ^χenv
envspace_fidel = ℂ^χenv_fidel
check_fidelity = true
ctm_alg = SimultaneousCTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10))
)

trunc_alg1 = ExactEnvTruncation(ctm_alg, envspace, truncdim(Dcut); check_fidelity)
trunc_alg2 = IntermediateEnvTruncation(ctm_alg, envspace, truncdim(Dcut); check_fidelity, maxiter = 30, verbosity = 0)
trunc_alg3 = ApproximateEnvTruncation(ctm_alg, envspace, truncdim(Dcut); check_fidelity, maxiter = 30, verbosity = 0)
trunc_algs = [trunc_alg1, trunc_alg2, trunc_alg3]

As = [make_peps(2,2), make_peps(2,3)]
Os = [make_pepo(2,2), make_pepo(2,3)]
states = [As[2], Os[2], (As[1],Os[1]), (Os[1],Os[1])]
state_types = ["A", "O", "AO", "OO"]

for trunc_alg in trunc_algs
    for (i,state) in enumerate(states)
        A_trunc, overlap = approximate_state(state, trunc_alg)
        println("For $(typeof(trunc_alg)) for $(state_types[i]), the fidelity = $overlap")
        # @test 0.85 < overlap < 1.0
        # error = 1 - fidelity(A_trunc, A_trunc_exact, trunc_alg.ctm_alg, trunc_alg.envspace)
        # println("error = $error")
    end
end

