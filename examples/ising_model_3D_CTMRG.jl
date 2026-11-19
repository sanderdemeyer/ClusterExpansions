using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using Plots
using Random
using LinearAlgebra

Random.seed!(123456789)
setprecision(100)

T = Float64
spin_symmetry = Trivial
J = 1.0
g = 2.0

function evolve_CE(obs, Δβ_iter, Δβ_obs, maxiter_obs, D, chipeps, chienv; J = 1.0, g = 0.0, spin_symmetry = Trivial)
    if spin_symmetry == Trivial
        pepsspace = ℂ^chipeps
        envspace = ℂ^chienv
    else
        pepsspace = Vect[Z2Irrep](0 => chipeps - div(chipeps,2), 1 => div(chipeps,2))
        envspace = Vect[Z2Irrep](0 => chienv - div(chienv,2), 1 => div(chienv,2))
    end
            
    nsteps = round(Int, Δβ_obs / Δβ_iter)
    @assert abs(nsteps - Δβ_obs / Δβ_iter) < 1e-10 "Δβ_obs and Δβ_iter are incompatible"

    # define time-evolution (truncation) algorithm
    trunc_alg = NoEnvTruncation(truncrank(D))

    # define cluster-expansion algorithm. p = 3 is default.
    ce_alg = ising_operators(J, g, 0.0; T = BigFloat, symmetry = "Oh", verbosity = 1, p = 3)
    ΔO = evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64);
    
    # start with density operator at β = 0
    O = evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64);

    ms = []
    for i = 1:maxiter_obs
        @info "At step $i / $(maxiter_obs)"
        # Do the time evolution
        for _ = 1:nsteps
            O, = approximate_state((O,ΔO), trunc_alg; rot_invariance = false)
        end
        # Calculate the CTMRG environment and the expectation value
        m, = MPSKit.expectation_value(O, [obs], pepsspace, envspace; miniter=3, maxiter=80, tol=1e-7, verbosity=3, sqrts = true)
        push!(ms, m)
        @info "magn is $(ms[end])"
    end
    βs = [Δβ_obs*i for i in 1:maxiter_obs]
    return βs, ms
end

# values of the bond dimension of the density operator, and the two environment bond dimensions of CTMRG (these are quite low)
D_CE = 4
D_SU = 2
chipeps = 3
chienv = 5

# Which observable do you want to calculate?
obs = SpinOperators.σᶻ(T, spin_symmetry)

# parameters for time evolution
Δβ_iter = 0.005
Δβ_obs = 0.01
maxiter_obs = 3

βs, ms = evolve_CE(obs, Δβ_iter, Δβ_obs, maxiter_obs, D_CE, chipeps, chienv; J = 1.0, g, spin_symmetry)

# The exact results for the critical temperature
βc = if g == 0.0
    0.2216546
elseif g == 2.0
    1 / 4.212488872894219
elseif g == 4.5
    1 / 2.6092116221586945
else
    error("Unknown g value")
end

# Make plot
plt = plot()
scatter!(plt, βs, abs.(ms), label = "CE + CTMRG")
vline!(plt, [βc], label = "QMC")
xlabel!("β")
ylabel!("Magnetization")
display(plt)
