using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfiniteSquareNetwork, InfinitePartitionFunction, LocalOperator, vertices
# include("apply_PEPO.jl")

function imaginary_time_evolution(ψ, O, χenv; maxiter = 10)
    σx = TensorMap(scalartype(ψ)[0 1; 1 0], ℂ^2, ℂ^2)
    Magn = LocalOperator(fill(ℂ^2, 1, 1), (CartesianIndex(1, 1),) => σx)

    env0 = CTMRGEnv(ψ, ComplexSpace(χenv));
    E = 0
    for i = 1:maxiter
        println("In step $i, the norm = $(norm(ψ))")
    
        envs, = leading_boundary(env0, ψ, ctm_alg);
    
        E = expectation_value(ψ, H, envs)
        magn = expectation_value(ψ, Magn, envs)
        println("Energy is $(E), Magnetization is $(magn)")
        ψ = InfinitePEPS(apply_PEPO(ψ[1,1], O))
    end
    return ψ, E, magn
end

p = 5
β = 1e-1
D = 2
χenv = 40

symmetry = "C4"
critical = false

setprecision(128)
T = Complex{BigFloat}

if critical
    J = 1.0
    g = 3.1
    e = -1.6417 * 2
    mˣ = 0.91
else
    J = 1.0
    g = 2.0
    e = -1.2379 * 2
    mˣ = 0.524
end

N1, N2 = (1,1)

ctm_alg = SimultaneousCTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10))
)

pspace = ℂ^2
vspace = ℂ^D
ψ = InfinitePEPS(pspace, vspace, vspace)

twosite_op = rmul!(σᶻᶻ(T), -J)
onesite_op = rmul!(σˣ(T), g * -J)

H = transverse_field_ising(InfiniteSquare(); g)

# spaces = i -> (i >= 0) ? ℂ^(2*i+1) : ℂ^30
# spaces = i -> (i >= 0) ? ℂ^(2*i) : ℂ^30
spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^30

O, O_clust = clusterexpansion(T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = "C4", verbosity = 2)
    
println(a)
ψ, E, magn = imaginary_time_evolution(ψ, O_clust, χenv; maxiter = 1000)

@test E ≈ e atol = 0.2
@test imag(magn) ≈ 0 atol = 1e-6
@test abs(magn) ≈ mˣ atol = 5e-2
