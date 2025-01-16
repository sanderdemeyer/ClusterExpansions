using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction, LocalOperator, vertices
include("apply_PEPO.jl")

function imaginary_time_evolution(O, χenv; maxiter = 10)
    pspace = ℂ^2
    spaces = fill(pspace, (1,1))
    
    ψ = InfinitePEPS(TensorMap(randn, pspace, pspace ⊗ pspace ⊗ pspace' ⊗ pspace'))

    σx = TensorMap(scalartype(ψ)[0 1; 1 0], ℂ^2, ℂ^2)
    Magn = LocalOperator(spaces, (CartesianIndex(1, 1),) => σx)

    env0 = CTMRGEnv(ψ, ComplexSpace(χenv));
    E = 0
    for i = 1:maxiter
        println("In step $i, the norm = $(norm(ψ))")
    
        envs = leading_boundary(env0, ψ, ctm_alg);
    
        E = expectation_value(ψ, H, envs)
        magn = expectation_value(ψ, Magn, envs)
        println("Energy is $(E), Magnetization is $(magn)")

        ψ = InfinitePEPS(apply(ψ[1,1], O; tol = 1e-2, verbosity = 0))
    end
    return ψ, E, Magn
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
twosite_op = rmul!(σᶻᶻ(), -J)
onesite_op = rmul!(σˣ(), g * -J)

H = transverse_field_ising(InfiniteSquare(); g)

spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^10
O_clust = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces)

ψ, E, magn = imaginary_time_evolution(O_clust, χenv_approx; maxiter = 5)

@test E ≈ e atol = 1e-2
@test imag(magn) ≈ 0 atol = 1e-6
@test abs(magn) ≈ mˣ atol = 5e-2
