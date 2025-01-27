using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction, LocalOperator, vertices
import PEPSKit: S_xx, S_yy, S_zz

function imaginary_time_evolution(ψ, O, χenv; maxiter = 10)
    pspace = ℂ^2
    spaces = fill(pspace, (1,1))
    
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

        A = flip_arrows(ψ[1,1])
        if norm(A - rotl90(A)) / norm(A) > 1e-10
            A = (A + rotl90(A) + rotl90(rotl90(A)) + rotl90(rotl90(rotl90(A))))/4
            @warn "State was not rotationally invariant. Error = $(norm(A - rotl90(A)) / norm(A)). New error = $(norm(A_symm - rotl90(A_symm)) / norm(A_symm))"
        end
        ψ = InfinitePEPS(flip_arrows(A))

        A_new, Ws = apply(ψ[1,1], O; verbosity = 0)
        ψ = InfinitePEPS(A_new)
    end
    return ψ, E, Magn
end

p = 3
β = 1e-2
D = 2
χenv = χenv_approx = 16

(Jx, Jy, Jz) = (-1.0, 1.0, -1.0)
h = 0.0
e = -0.6602310934799577
unitcell = (1,1)

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

H = heisenberg_XYZ(InfiniteSquare(unitcell...))

pspace = ℂ^2
T, S = ComplexF64, Trivial
twosite_op = rmul!(S_xx(T, S; spin=1//2), Jx) + rmul!(S_yy(T, S; spin=1//2), Jy) + rmul!(S_zz(T, S; spin=1//2), Jz)
onesite_op = rmul!(σᶻ(), h)


spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^10
_, O_clust = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = "C4")


A = TensorMap(randn, pspace, pspace ⊗ pspace ⊗ pspace' ⊗ pspace')
A_symm = flip_arrows(make_translationally_invariant(flip_arrows(A)))
ψ = InfinitePEPS(A_symm)

ψ, E, magn = imaginary_time_evolution(ψ, O_clust, χenv_approx; maxiter = 100)

@test E ≈ e atol = 1e-2