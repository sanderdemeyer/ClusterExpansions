using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using PEPSKit
using Plots
using JLD2
using Test

# Set up the SF model
spin_symmetry = Trivial
t = 1.0
V = 0.0
ce_alg = spinless_fermion_operators(t, V, 0.0; symmetry = "C4", T = Complex{BigFloat}, p = 3, loop_space = Vect[fℤ₂](0 => 2, 1 => 2), svd = true)

χenv = 24  # Environment bond dimension used in the calculation of expectation values
χenv_approx = 24 # Environment bond dimension used in the truncation scheme

# Parameters in the truncation scheme
Dcut = 2
# Dcut_CTM = 8
ftol =  1e-10
gradnormtol = -Inf #1e-9
ctm_alg = SimultaneousCTMRG(; maxiter = 500, verbosity = 1)
envspace = Vect[fℤ₂](0 => div(χenv_approx,2), 1 => div(χenv_approx,2))
truncspace = Vect[fℤ₂](0 => Dcut - div(Dcut,2), 1 => div(Dcut,2))

maxiter_LBFGS = 20
trunc_alg_local = ClusterExpansions.NoEnvTruncation(truncdim(Dcut))
trunc_alg_vomps = VOPEPO_CTMRG(ctm_alg, envspace, truncspace, ftol, gradnormtol, maxiter_LBFGS; verbosity = 2, c₁ = 1e-3)

# Set up time evolution algorithm
β₀ = 0.0
Δβ = 0.1
maxiter = 2
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

# Define observables
ctm_alg = SimultaneousCTMRG(; maxiter = 500)
vumps_alg = VUMPS(; maxiter = 500, verbosity = 1)

observables = PEPO_observables([FermionOperators.f_hop()], [ctm_alg])
obs_function = O -> calculate_observables(O, χenv, observables)

canoc_alg = nothing

βs_local, expvals_local, Os_local = time_evolve(ce_alg, time_alg, trunc_alg_local, obs_function; skip_first = false, canoc_alg);
βs_vomps, expvals_vomps, Os_vomps = time_evolve(ce_alg, time_alg, trunc_alg_vomps, obs_function; skip_first = false, initial_guesses = i -> nothing, canoc_alg); #Os[i]);

hops_local = [e[1] for e in expvals_local]
hops_vomps = [e[1] for e in expvals_vomps]

file = jldopen("analytical_solution_SF.jld2", "r")
βs_exact = file["βs"]
values_exact = file["values"]
close(file)

if β₀ == 0.1
    errors_local = [hops_local[i] / 4 - values_exact[20*i+1] for i = eachindex(βs_local)]
    errors_vomps = [hops_vomps[i] / 4 - values_exact[20*i+1] for i = eachindex(βs_vomps)]
elseif β₀ == 0.0
    errors_local = [hops_local[i] / 4 - values_exact[20*(i-1)+1] for i = eachindex(βs_local)]
    errors_vomps = [hops_vomps[i] / 4 - values_exact[20*(i-1)+1] for i = eachindex(βs_vomps)]
end

plt = scatter()
scatter!(βs_local, real.(hops_local), label = "Local")
scatter!(βs_vomps, real.(hops_vomps), label = "Variational")
plot!(βs_exact, real.(values_exact) .* 4, label = "Exact")
xlabel!("β")
ylabel!("Hopping term")
title!("Spinless fermion model for V = $V")
display(plt)

plt = scatter()
scatter!(βs_local, abs.(errors_local), label = "Local", yscale = :log10)
scatter!(βs_vomps, abs.(errors_vomps), label = "Variational")
xlabel!("β")
ylabel!("Error")
title!("Spinless fermion model for V = $V")
display(plt)
