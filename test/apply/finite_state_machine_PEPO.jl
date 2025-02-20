using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfiniteSquareNetwork, InfinitePartitionFunction, LocalOperator, vertices

J = 1.0
twosite_op = rmul!(σᶻᶻ(T), -J)

U, S, V = tsvd(twosite_op, ((1,3),(2,4)))
L = U * sqrt(S)
R = permute(sqrt(S) * V, ((2,3), (1,)))

@assert dim(domain(S)) == 4

pspace = ℂ^2
vspace = ℂ^D
χenv = 10

# vspaceO = ℂ^11
# O = zeros(pspace ⊗ pspace', vspaceO ⊗ vspaceO ⊗ vspaceO' ⊗ vspaceO')

# unit = ComplexF64[1.0 0.0; 0.0 1.0]

# O[][:,:,1,1,1,1] = unit
# O[][:,:,3,3,3,3] = unit
# O[][:,:,4,4,4,4] = unit
# O[][:,:,2,2,2,2] = unit

# O[][:,:,7,7,7,1] = unit
# O[][:,:,7,3,7,7] = unit
# O[][:,:,3,6,4,6] = unit
# O[][:,:,7,4,7,7] = unit
# O[][:,:,7,7,7,2] = unit
# O[][:,:,1,5,2,5] = unit

# O[][:,:,7,8:11,7,5] = L[]
# O[][:,:,7,6,7,8:11] = R[]

# O[][:,:,1,7,7,7] = unit
# O[][:,:,7,7,3,7] = unit
# O[][:,:,6,3,6,4] = unit
# O[][:,:,7,7,4,7] = unit
# O[][:,:,2,7,7,7] = unit
# O[][:,:,5,1,5,2] = unit

# O[][:,:,5,7,8:11,7] = L[]
# O[][:,:,8:11,7,6,7] = R[]

vspaceO = ℂ^1
O = zeros(T, pspace ⊗ pspace', vspaceO ⊗ vspaceO ⊗ vspaceO' ⊗ vspaceO')

O[][:,:,1,1,1,1] = rmul!(σˣ(T), g)[]

PEPO = InfinitePEPO(O)
peps = InfinitePEPS(pspace, vspace)

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

network = InfiniteSquareNetwork(peps, PEPO)
# env2, = leading_boundary(CTMRGEnv(peps, ComplexSpace(χenv)), peps, ctm_alg)
# env3, = leading_boundary(CTMRGEnv(network, ComplexSpace(χenv)), network, ctm_alg)
env2, = leading_boundary(CTMRGEnv(peps, χenv), peps, ctm_alg)
env3, = leading_boundary(CTMRGEnv(network, χenv), network, ctm_alg)

result = fixedpoint_triple(PEPO, peps, env2, env3, opt_alg)

