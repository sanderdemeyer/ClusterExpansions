using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfiniteSquareNetwork, InfinitePartitionFunction, LocalOperator, vertices
using Random
# include("apply_PEPO_exactenv.jl")

function imaginary_time_evolution(ψ, O, χenv; maxiter = 10)
    for i = 1:maxiter
        ψnew = InfinitePEPS(apply_PEPO(ψ[1,1], O))
        error = abs(1.0 - (norm(ψ)*exp(2))/norm(ψnew))
        ψ = copy(ψnew)
        if (i > 3)
            @test error < 1e-5
        end
    end
    return ψ
end

Random.seed!(2928528935)

p = 2
β = 1e-1
D = 2
χenv = 10

symmetry = "C4"
critical = false

# setprecision(128)
T = Complex{Float64}

g = 2.0

N1, N2 = (1,1)

ctm_alg = SimultaneousCTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10))
)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg, optimizer=LBFGS(4; gradtol=1e-3, verbosity=3)
)

pspace = ℂ^2
vspace = ℂ^D
vspaceO = ℂ^1
ψ = InfinitePEPS(pspace, vspace, vspace)

onesite_op = rmul!(σˣ(T), g)
O = zeros(scalartype(ψ), pspace ⊗ pspace', vspaceO ⊗ vspaceO ⊗ vspaceO' ⊗ vspaceO')
O[:,:,1,1,1,1] = exp(onesite_op)[]
H = LocalOperator(fill(ℂ^2, 1, 1), (CartesianIndex(1, 1),) => onesite_op)

env, = leading_boundary(CTMRGEnv(ψ, ComplexSpace(χenv)), ψ, ctm_alg)
ψ, env, E, = fixedpoint(H, ψ, env, opt_alg)
ψ = ψ / norm(ψ)

@test abs(abs(expectation_value(ψ, H, env)) - 2.0) < 1e-6

ψ = imaginary_time_evolution(ψ, O, χenv; maxiter = 10)

@test abs(abs(expectation_value(ψ, H, env)) - 2.0) < 1e-6

