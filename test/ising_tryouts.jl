using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction
using Zygote


p = 2
β = 1e-3
D = 2
χenv = 8

symmetry = "C4"

J = 1.0
g = 0.0
N1, N2 = (1,1)

pspace = ℂ^2
twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)
# function spaces(i)
#     if i == -1
#         return ℂ^(10)
#     elseif i == 0
#         return ℂ^(1)
#     elseif i == 1
#         return ℂ^(2)
#     elseif i == 2
#         return ℂ^(4)
#     elseif i == 3
#         return ℂ^(8)
#     else
#         return ℂ^(2^(2*i))
#     end
# end

spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^(-10*i)
# spaces = i -> (i >= 0) ? ℂ^(2^(2*i) - (i == 1)*2) : ℂ^10
# spaces = i -> (i >= 0) ? ℂ^(1+2*i) : ℂ^(-10*i)
O, O_clust = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry, verbosity = 1)
Magn = TensorMap([1.0 0.0; 0.0 -1.0], pspace, pspace)
PEPO = InfinitePEPO(O_clust)


vspace = ℂ^3
# A = TensorMap(randn, pspace, pspace ⊗ pspace ⊗ pspace' ⊗ pspace')
# A = flip_arrows(make_translationally_invariant(flip_arrows(A)))
A = TensorMap(randn, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
A = flip_arrows(make_translationally_invariant(flip_arrows(A)))
# O_clust = flip_arrows(O_clust)
# W, error = find_truncation(A, O_clust; step_size = 1e-3, ϵ = 1e-10, max_iter = 10000, line_search = false, linesearch_options = 1, verbosity = 2)
Ws, A_trunc = find_truncation(A, O_clust)
println("Relative error = $(norm(A-A_trunc)/norm(A))")


A = apply(A, O_clust, Ws; spaces = [vspace], verbosity = 3)

println(a)

@tensor Z[-3 -4; -1 -2] := O_clust[1 1; -1 -2 -3 -4]

pspace = ℂ^2
T = TensorMap(randn, pspace, pspace ⊗ pspace ⊗ pspace' ⊗ pspace')
test = InfinitePEPS(T)

partfunc = InfinitePartitionFunction(Z)
@tensor A_M[-3 -4; -1 -2] := O_clust[1 2; -1 -2 -3 -4] * Magn[2; 1]
partfunc_M = InfinitePartitionFunction(A_M)

env0 = CTMRGEnv(partfunc, ComplexSpace(χenv));

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)

env = leading_boundary(env0, partfunc, ctm_alg);
Z_pf = norm(partfunc, env);

@tensor E_pf[-1 -2 -3; -4 -5 -6] := O_clust[1 2; -1 3 -2 -3] * O_clust[4 5; -4 -5 -6 3] * twosite_op[2 5; 1 4]
O_E_L, Σ, O_E_R = tsvd(E_pf; trunc = truncdim(dim(O_clust.dom[1])));
O_E_new = permute(O_E_L * sqrt(Σ), ((3,2),(1,4)))
println("O_E_L = $(summary(O_E_L))");
println("O_E_R = $(summary(O_E_R))");
println("O_E_new = $(summary(O_E_new))");


magn = norm(partfunc_M, env);
E = norm(InfinitePartitionFunction(O_E_new), env);
println("magn should be zero, is $(magn)");
println("E = $(E)");
println("Z should be $(2*(cosh(β*J)^2)), is $(Z_pf)");
println("relative error on Z = $(abs(Z_pf - 2*(cosh(β*J)^2))/Z_pf)");

"""
ctm_alg = SimultaneousCTMRG(; tol=1e-10, trscheme=truncdim(χenv))
# ctm_alg = SequentialCTMRG(; maxiter=300, tol=1e-7, trscheme=truncdim(χenv))

opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=2),
    gradient_alg=LinSolver(),
    reuse_env=true,
)

# ground state search
state = InfinitePEPS(2, D)
ctm = leading_boundary(CTMRGEnv(ψ₀, ComplexSpace(χenv)), ψ₀, ctm_alg)
result = fixedpoint(ψ₀, PEPO, opt_alg, ctm)

env = CTMRGEnv(PEPO, ComplexSpace(χenv));

"""
