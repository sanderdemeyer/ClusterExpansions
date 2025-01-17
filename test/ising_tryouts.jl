using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction

p = 3
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

spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^10
# spaces = i -> (i >= 0) ? ℂ^(1+2*i) : ℂ^(-10*i)
O_clust = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry)
Magn = TensorMap([1.0 0.0; 0.0 -1.0], pspace, pspace)
PEPO = InfinitePEPO(O_clust)

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
