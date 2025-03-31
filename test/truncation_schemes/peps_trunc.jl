using TensorKit
using KrylovKit
using MPSKitModels
using PEPSKit
using ClusterExpansions
using JLD2

T = ComplexF64

pspace = ℂ^2
Dspace = ℂ^3
space = ℂ^2
envspace = ℂ^4

ctm_alg = SimultaneousCTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10))
)

ψ = randn(T, pspace, Dspace ⊗ Dspace ⊗ Dspace' ⊗ Dspace')
O = randn(T, pspace ⊗ pspace', Dspace ⊗ Dspace ⊗ Dspace' ⊗ Dspace')

# ψnew, fidel = approximate_fullenv([ψ], space, ctm_alg, envspace; check_fidelity = true)
# Ws = find_isometry_approx(ψ, space, ctm_alg, envspace; method = "intermediate")
Ws = find_isometry_approx(O, space, ctm_alg, envspace; method = "intermediate")