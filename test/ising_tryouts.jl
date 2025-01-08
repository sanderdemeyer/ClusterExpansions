using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction

p = 4
β = 1e-3
D = 2
χenv = 6

J = 1.0
g = 0.0
N1, N2 = (1,1)

# PEPO_dict = get_all_indices(3, β)
# O = get_PEPO(ℂ^2, PEPO_dict)
# println("done")

twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)

pspace = ℂ^2
H = transverse_field_ising(ComplexF64, Trivial, InfiniteSquare(N1,N2); J = J, g = g)
ψ₀ = InfinitePEPS(2, D; unitcell=(N1,N2))

O = clusterexpansion(p, β, twosite_op, onesite_op; levels_convention = "tree_depth")
Magn = TensorMap([1.0 0.0; 0.0 -1.0], pspace, pspace)
PEPO = InfinitePEPO(O)

@tensor Z[-3 -4; -1 -2] := O[1 1; -1 -2 -3 -4]

pspace = ℂ^2
T = TensorMap(randn, pspace, pspace ⊗ pspace ⊗ pspace' ⊗ pspace')
test = InfinitePEPS(T)

partfunc = InfinitePartitionFunction(Z)
@tensor A_M[-3 -4; -1 -2] := O[1 2; -1 -2 -3 -4] * Magn[2; 1]
partfunc_M = InfinitePartitionFunction(A_M)

envtest = CTMRGEnv(test, ComplexSpace(χenv));
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
contracted = norm(partfunc, env)

Z = norm(partfunc, env);
magn = norm(partfunc_M, env);
println("magn should be zero, is $(magn)");
println("Z should be $(2*(cosh(β*J)^2)), is $(Z)");
println("relative error on Z = $(abs(Z - 2*(cosh(β*J)^2))/Z)");

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
