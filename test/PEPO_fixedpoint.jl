
using Test
using Random
using LinearAlgebra
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit
using Zygote
import PEPSKit: rmul!, σᶻᶻ, σˣ, σᶻ #, InfiniteSquare, InfiniteSquareNetwork, InfinitePartitionFunction, LocalOperator, vertices
using ClusterExpansions

p = 4
β = 1e-1

symmetry = "C4"

setprecision(128)
T = Complex{BigFloat}

J = T(1.0)
g = T(2.0)

if g == 0.0
    e = -1.0 * 2
    mˣ = 0.0
    mᶻ = 1.0
elseif g == 3.1
    e = -1.6417 * 2
    mˣ = 0.91
    mᶻ = 1.0
elseif g == 2.0
    e = -1.2379 * 2
    mˣ = 0.524
    mᶻ = 1.0
elseif g == 1.0
    
end
    
N1, N2 = (1,1)

twosite_op = rmul!(σᶻᶻ(T), -J)
onesite_op = rmul!(σˣ(T), g * -J)

H = transverse_field_ising(InfiniteSquare(); g = Float64(g))

spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : ℂ^10

# _, O_clust = clusterexpansion(T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = "C4", verbosity = 2)

# O = zeros(ComplexF64, codomain(O_clust), domain(O_clust))
# for (i,tens) in enumerate(O_clust.data)
#     O.data[i][] = O_clust.data[i][]
# end
# O = convert(TensorMap, O)

# @tensor Mz[-1 -2; -3 -4 -5 -6] := O[1 -2; -3 -4 -5 -6] * σᶻ(T)[-1; 1]
# @tensor Mx[-1 -2; -3 -4 -5 -6] := O[1 -2; -3 -4 -5 -6] * σˣ(T)[-1; 1]

# initialize
χpeps = ℂ^2
χenv = ℂ^12

# cover all different flavors
ctm_styles = [:sequential, :simultaneous]
projector_algs = [:halfinfinite]

Random.seed!(81812781144)

# prep
ctm_alg = SimultaneousCTMRG(; maxiter=150, tol=1e-8, verbosity=2)
alg_rrule = EigSolver(;
    solver_alg=KrylovKit.Arnoldi(; maxiter=30, tol=1e-6, eager=true),
    iterscheme=:diffgauge,
)
opt_alg = LBFGS(32; maxiter=200, gradtol=1e-5, verbosity=3)

function pepo_retract(x, η, α)
    peps = deepcopy(x[1])
    peps.A .+= η.A .* α
    env2 = deepcopy(x[2])
    env3 = deepcopy(x[3])
    return (peps, env2, env3), η
end

# contract
T = InfinitePEPO(O) # unitcell=(1, 1, 1)
psi0 = initializePEPS(T, χpeps)
env2_0 = CTMRGEnv(InfiniteSquareNetwork(psi0), χenv)
env3_0 = CTMRGEnv(InfiniteSquareNetwork(psi0, T), χenv)

# optimize free energy per site
(psi_final, env2_final, env3_final), f, = optimize(
    (psi0, env2_0, env3_0), opt_alg; retract=pepo_retract, inner=PEPSKit.real_inner
) do (psi, env2, env3)
    E, gs = withgradient(psi) do ψ
        n2 = InfiniteSquareNetwork(ψ)
        env2′, info = PEPSKit.hook_pullback(
            leading_boundary, env2, n2, ctm_alg; alg_rrule
        )
        n3 = InfiniteSquareNetwork(ψ, T)
        env3′, info = PEPSKit.hook_pullback(
            leading_boundary, env3, n3, ctm_alg; alg_rrule
        )
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env2, env2′)
            PEPSKit.update!(env3, env3′)
        end
        λ3 = network_value(n3, env3)
        λ2 = network_value(n2, env2)
        return -real(log(λ3 / λ2))
        # return -log(real(λ3 / λ2))
        # return (real(λ3 / λ2))
    end
    g = only(gs)
    return E, g
end

# check energy
n3_final = InfiniteSquareNetwork(psi_final, T)
energy_final = expectation_value(psi_final, H, env2_final)
mz = PEPSKit.contract_local_tensor((1, 1, 1), Mz, n3_final, env3_final)
mx = PEPSKit.contract_local_tensor((1, 1, 1), Mx, n3_final, env3_final)
nrm3 = PEPSKit._contract_site((1, 1), n3_final, env3_final)

# compare to Monte-Carlo result from https://www.worldscientific.com/doi/abs/10.1142/S0129183101002383
@test e ≈ energy_final rtol = 1e-3
@test abs(abs(mx / nrm3) - mˣ) < 1e-4
@test Float64(abs(mz / nrm3)) ≈ mᶻ rtol = 1e-4


file = jldopen(name, "w")
file["peps"] = copy(peps)
close(file)