using TensorKit, PEPSKit, KrylovKit
using ClusterExpansions
import PEPSKit: rmul!, σᶻᶻ, σˣ
using Test

T = Complex{Float64}

D = 2
DO = 2
χenv = 24
Dcut = 3

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

trunc_alg1 = ExactEnvTruncation(ctm_alg, envspace, truncdim(Dcut), true)
trunc_alg2 = IntermediateEnvTruncation(ctm_alg, envspace, truncdim(Dcut), true; maxiter = 30, verbosity = 2)
trunc_alg3 = ApproximateEnvTruncation(ctm_alg, envspace, truncdim(Dcut), true; maxiter = 30, verbosity = 2)

A_trunc_exact, fidelity_exact = approximate_state((A,O), trunc_alg1)

for trunc_alg in [trunc_alg2, trunc_alg3]
    A_trunc, fidelity_approx = approximate_state((A,O), trunc_alg)
    println("To test: $(fidelity_exact - fidelity_approx)")
    # @test 0 < fidelity_exact - fidelity_approx < 3e-2
    error = 1 - fidelity(A_trunc, A_trunc_exact, trunc_alg.ctm_alg, trunc_alg.envspace)
    println("error = $error")
end

println("Done!")
