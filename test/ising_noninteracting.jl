using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction, LocalOperator, vertices

function imaginary_time_evolution(ψ, O, χenv; maxiter = 10)
    pspace = ψ[1,1].codom[1]
    spaces = fill(pspace, (1,1))

    σz = TensorMap(scalartype(ψ)[1 0; 0 -1], ℂ^2, ℂ^2)
    Magn = LocalOperator(spaces, (CartesianIndex(1, 1),) => σz)

    operator = TensorMap(convert(Array, O_clust[][:,:,1,1,1,1]), pspace, pspace)

    env0 = CTMRGEnv(ψ, ComplexSpace(χenv));
    E = 0
    for i = 1:maxiter
        println("In step $i, the norm = $(norm(ψ))")
    
        envs = leading_boundary(env0, ψ, ctm_alg);
    
        A = flip_arrows(ψ[1,1])
        if norm(A - rotl90(A)) / norm(A) > 1e-10
            A_symm = (A + rotl90(A) + rotl90(rotl90(A)) + rotl90(rotl90(rotl90(A))))/4
            ψ = InfinitePEPS(flip_arrows(A_symm))
            @warn "State was not rotationally invariant. Error = $(norm(A - rotl90(A)) / norm(A)). New error = $(norm(A_symm - rotl90(A_symm)) / norm(A_symm))"
        end
        ψ_copy = copy(ψ)
        A_new, Ws = apply(ψ[1,1], O; verbosity = 0)
        ψ_apply = InfinitePEPS(A_new)
        if norm(ψ_copy - ψ)/norm(ψ) > 1e-15
            @warn "The apply function is not working properly. Error = $(norm(ψ_apply - ψ))"
        end
        @tensor A_other[-1; -2 -3 -4 -5] := ψ[1,1][1; -2 -3 -4 -5] * operator[-1; 1]
        println("Difference between exact and apply = $(norm(A_other - ψ_apply[1,1])/norm(A_other))")
        ψ = InfinitePEPS(A_other / norm(A_other))

        E = expectation_value(ψ, H, envs)
        magn = expectation_value(ψ, Magn, envs)
        println("Exact: Energy is $(E), Magnetization is $(magn)")

        E = expectation_value(ψ_apply, H, envs)
        magn = expectation_value(ψ_apply, Magn, envs)
        println("Apply: Energy is $(E), Magnetization is $(magn)")
    end


    return ψ, E, Magn
end

p = 1
β = 1e-2
D = 4
χenv = χenv_approx = 10

maxiter = 50

J = 0.0
g = 1.0 # 3.1
e = -1.0 # -1.6417 * 2
mˣ = 0.0 # 0.91
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
onesite_op = rmul!(σˣ(), g)
# σz = TensorMap(scalartype(ψ)[1 0; 0 1], ℂ^2, ℂ^2)
# σz = TensorMap(scalartype(ψ)[0 1; 1 0], ℂ^2, ℂ^2)

spaces_op = fill(domain(onesite_op)[1], (N1, N2))
H = LocalOperator(spaces_op, (CartesianIndex(1, 1),) => onesite_op)

spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^10
_, O_clust = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = "C4")

pspace = ℂ^2

A = TensorMap(randn, pspace, pspace ⊗ pspace ⊗ pspace' ⊗ pspace')
A_symm = flip_arrows(make_translationally_invariant(flip_arrows(A)))
ψ = InfinitePEPS(A_symm)

ψ, E, magn = imaginary_time_evolution(ψ, O_clust, χenv_approx; maxiter = maxiter)


println(a)

ctm_alg = CTMRG(; maxiter=300, tol=1e-7)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:diffgauge),
    reuse_env=true,
)

envs0 = CTMRGEnv(randn, ComplexF64, ψ, ℂ^χenv)
envs = leading_boundary(envs0, ψ, ctm_alg)

result = fixedpoint(ψ, ham, opt_alg, envs)
println("E = $(result.E)")

@test E ≈ e atol = 1e-2
@test imag(magn) ≈ 0 atol = 1e-6
@test abs(magn) ≈ mˣ atol = 5e-2
