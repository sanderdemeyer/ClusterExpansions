using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using PEPSKit
using Plots
using JLD2

# Set up the SF model
spin_symmetry = Trivial
t = 1.0
V = 0.0
ce_alg = spinless_fermion_operators(t, V, 0.0; symmetry = "C4", T = Complex{BigFloat})

χenv = 6  # Environment bond dimension used in the calculation of expectation values
χenv_approx = 6 # Environment bond dimension used in the truncation scheme

# Parameters in the truncation scheme
Dcut = 4
maxiter = 10
ftol = 1e-7
gradnormtol = 1e-9
ctm_alg = SimultaneousCTMRG(; maxiter = 250, verbosity = 1)
envspace = Vect[fℤ₂](0 => div(χenv_approx,2), 1 => div(χenv_approx,2))
truncspace = Vect[fℤ₂](0 => div(Dcut,2), 1 => div(Dcut,2))

trunc_alg = VOPEPO_CTMRG(ctm_alg, envspace, truncspace, ftol, gradnormtol, maxiter; verbosity = 2)

# Set up time evolution algorithm
β₀ = 0.05
Δβ = 0.05
maxiter = 5
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

# Define observables
ctm_alg = SimultaneousCTMRG(; maxiter = 100)
observables = PEPO_observables([FermionOperators.f_hop()], [ctm_alg]; funcs = [x -> x])
obs_function = O -> calculate_observables(O, χenv, observables)

βs, expvals, Os = time_evolve(ce_alg, time_alg, trunc_alg, obs_function);

hops = [e[1] for e in expvals]

plt = scatter()
scatter!(βs, real.(hops) ./ 4, label = "CE")
file = jldopen("analytical_solution_SF.jld2", "r")
plot!(file["βs"], file["values"], label = "Exact")
close(file)
xlabel!("β")
ylabel!("hopping operator")
title!("Spinless fermion model for V = $V")
display(plt)
