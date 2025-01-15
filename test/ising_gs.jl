using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction
include("apply_PEPO.jl")

function imaginary_time_evolution(O, χenv; maxiter = 10)
    pspace = ℂ^2
    ψ = InfinitePEPS(TensorMap(randn, pspace, pspace ⊗ pspace ⊗ pspace' ⊗ pspace'))
    env0 = CTMRGEnv(ψ, ComplexSpace(χenv));
    E = 0
    for i = 1:maxiter
        println("summary: $(summary(ψ[1,1]))")
        ψ = InfinitePEPS(apply(ψ[1,1], O; tol = 1e-2, verbosity = 0))
        println("summary: $(summary(ψ[1,1]))")
        println("In step $i, the norm = $(norm(ψ))")
    
        envs = leading_boundary(env0, ψ, ctm_alg);
    
        E = expectation_value(ψ, H, envs)
        println("Energy is $(E)")
    end
    return ψ, E
end

p = 3
β = 1e-3
D = 2
χenv = 16
χenv_approx = 16

J = 1.0
g = 3.1
e = -1.6417 * 2
mˣ = 0.91
N1, N2 = (1,1)

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

pspace = ℂ^2
twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)

H = transverse_field_ising(InfiniteSquare(); g)

spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^10
O_clust = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces)
Magn = TensorMap([1.0 0.0; 0.0 -1.0], pspace, pspace)
PEPO = InfinitePEPO(O_clust)

ψ, E = imaginary_time_evolution(O_clust, χenv_approx; maxiter = 10)