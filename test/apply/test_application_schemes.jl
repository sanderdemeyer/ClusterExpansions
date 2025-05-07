using TensorKit, PEPSKit, KrylovKit
using ClusterExpansions
import PEPSKit: rmul!, σᶻᶻ, σˣ
using Test

T = Complex{Float64}

D = 2
DO = 3
χenv = 10
Dcut = 2

pspace = ℂ^D
vspace_peps = ℂ^D
vspace_O = ℂ^DO
envspace = ℂ^χenv

A = randn(T, pspace, vspace_peps ⊗ vspace_peps ⊗ vspace_peps' ⊗ vspace_peps')
O = randn(T, pspace ⊗ pspace', vspace_O ⊗ vspace_O ⊗ vspace_O' ⊗ vspace_O')

ctm_alg = SimultaneousCTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10))
)

check_fidelity = true

# trunc_alg1 = ExactEnvTruncation(ctm_alg, envspace, truncdim(Dcut); check_fidelity)

# trunc_alg3 = ApproximateEnvTruncation(ctm_alg, envspace, truncdim(Dcut); check_fidelity, maxiter = 30, verbosity = 2)
# A_trunc, overlap = approximate_state((O,O), trunc_alg1)

# println(done)
trunc_alg1 = ExactEnvTruncation(ctm_alg, envspace, truncdim(Dcut); check_fidelity)
trunc_alg2 = IntermediateEnvTruncation(ctm_alg, envspace, truncdim(Dcut); check_fidelity, maxiter = 30, verbosity = 0)
trunc_alg3 = ApproximateEnvTruncation(ctm_alg, envspace, truncdim(Dcut); check_fidelity, maxiter = 30, verbosity = 0)

trunc_algs = [trunc_alg1, trunc_alg2, trunc_alg3]
states = [A, O, (A,O), (O,O)]

for trunc_alg in trunc_algs
    for (i,state) in enumerate(states)
        A_trunc, overlap = approximate_state(state, trunc_alg)
        println("For trunc_alg = $trunc_alg and state type $i, the fidelity = $overlap")
        # @test 0.85 < overlap < 1.0
        # error = 1 - fidelity(A_trunc, A_trunc_exact, trunc_alg.ctm_alg, trunc_alg.envspace)
        # println("error = $error")
    end
end

println("Done!")
