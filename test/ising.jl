using TensorKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare
p = 2
β = 1
D = 2
χenv = 12

J = 1.0
g = 1.0
N1, N2 = (1,1)

# PEPO_dict = get_all_indices(3, β)
# O = get_PEPO(ℂ^2, PEPO_dict)
# println("done")

twosite_op = rmul!(σᶻᶻ(), -1.0)
onesite_op = rmul!(σˣ(), g * -J)

pspace = ℂ^2
H = transverse_field_ising(ComplexF64, Trivial, InfiniteSquare(N1,N2); J = J, g = g)
ψ₀ = InfinitePEPS(2, D; unitcell=(N1,N2))

O = clusterexpansion(p, β, twosite_op, onesite_op)
Magn = TensorMap([1.0 0.0; 0.0 -1.0], pspace, pspace)
PEPO = InfinitePEPO(O)

# @tensor Z[-1 -2 -3 -4] := O[1 1; -1 -2 -3 -4]


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