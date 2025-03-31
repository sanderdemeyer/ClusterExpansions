using TensorKit
using KrylovKit
using MPSKitModels
using PEPSKit
using ClusterExpansions
using JLD2

symmetry = nothing
critical = false

setprecision(128)
T = Complex{BigFloat}

p = 4
β = 1e-2
J = T(1.0)
g = T(0.0)

pspace = ℂ^2
twosite_op = rmul!(σᶻᶻ(T), -J)
onesite_op = rmul!(σˣ(T), g * -J)

spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10

O, O_clust = clusterexpansion(T, p, β, twosite_op, onesite_op; spaces = spaces, symmetry = symmetry, verbosity = 2)
D = dim(domain(O_clust)[1])
println("D = $D")
O_clust = convert(TensorMap, O_clust)
space = ℂ^2
envspace = ℂ^4

ctm_alg = SimultaneousCTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10))
)

O_trunc, fidel = approximate_fullenv([O_clust], space, ctm_alg, envspace; check_fidelity = true);