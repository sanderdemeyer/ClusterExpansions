using ClusterExpansions
using OptimKit

# β = 1
# D = 2
# χenv = 12

# PEPO_dict = get_all_indices(3, β)
# O = get_PEPO(ℂ^2, PEPO_dict)
# println("done")

H = heisenberg_XYZ(InfiniteSquare(); Jx=-1, Jy=1, Jz=-1) # sublattice rotation to obtain single-site unit cell
ψ₀ = InfinitePEPS(2, D)
env₀ = CTMRGEnv(ψ₀, ComplexSpace(χenv));

ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)
## Optimization algorithm
optim_method = ConjugateGradient
optim_tol = 1e-5
optim_maxiter = 100
verbosity = 2
boundary_maxiter = 50
tol_min = 1e-12
tol_max = 1e-4
tol_factor = 1e-3
symm = Full()
hermitian = true
boundary_method = VUMPS(; tol_galerkin=tol_max, maxiter=boundary_maxiter, dynamical_tols=true, eigs_tolfactor=1e-3, envs_tolfactor=1e-3, gauge_tolfactor=1e-6, tol_max=1e-4, verbose=verbosity >= 5)
pepo_alg = PEPOOptimize(;
    optim_method,
    optim_tol,
    optim_maxiter,
    verbosity,
    boundary_method,
    boundary_maxiter,
    tol_min,
    tol_max,
    tol_factor,
    symm,
    hermitian,
)

env_init = leading_boundary(env₀, ψ₀, ctm_alg)
result = fixedpoint(ψ₀, H, opt_alg, env_init)
println("E = $(result.E)")

# result2 = fixedpoint(ψ₀, O, opt_alg, env_init)

## Initialize PEPS and MPS fixed points
peps = copy(ψ₀) #INITIALIZE PEPS
envs = pepo_opt_environments(peps, O, pepo_alg.boundary_method; vspaces=[ℂ^χenv], hermitian=pepo_alg.hermitian) #INITIALIZE ENVS
normalize!(peps, envs.peps_boundary)

## Perform optimization
x, f, normgrad = leading_boundary(peps, O, pepo_alg, envs)
