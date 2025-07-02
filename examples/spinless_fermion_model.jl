using TensorKit
using TensorKitTensors
using ClusterExpansions
using PEPSKit
using Plots

# Set up the Ising model
model = spinless_fermion_operators
t = 1.0 # Nearest-neighbour hopping term
V = -2.5 # Interaction term. Repulsive if positive.
μ = 0.0 # Extra chemical potential on top of half-filling
model_param = (t, V, μ)

χenv = 30 # Environment bond dimension used in the calculation of expectation values

# Parameters in the truncation scheme
schmidt_cut = 1.5
Dcut = 4
trunc = truncdim(Dcut)
ctm_alg = SimultaneousCTMRG(; maxiter = 300, verbosity = 0)
χenv_approx = 2 # Environment bond dimension used in the calculation of the truncation isometry
envspace_approx = Vect[fℤ₂](0 => div(χenv_approx, 2), 1 => div(χenv_approx, 2))

trscheme = ApproximateEnvTruncation
trscheme_parameters = (ctm_alg, envspace_approx, trunc)
trscheme_kwargs = ()

# Set up time evolution algorithm
β₀ = 0.1
Δβ = 0.02
maxiter = ceil(Int, (1.25 - β₀) / Δβ) # Go up to a value of β = 0.7
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

ce_kwargs = (symmetry = nothing, T = Complex{BigFloat})

# Define observables, which can be both LocalOperators and functions
pspace = Vect[fℤ₂](0 => 1, 1 => 1)
H_num = localoperator_model(pspace, FermionOperators.f_num())

# A0 = permute(id(T, pspace ⊗ trivspace ⊗ trivspace), ((1,4),(5,6,2,3))) * (1 / sqrt(2))

βs, expvals, As = time_evolve_model(spinless_fermion_operators, (t, V, μ), time_alg, χenv, trscheme, trscheme_parameters; observables = [H_num], ce_kwargs);

# Extract the expectation values
filling = [abs(e[1]-0.5) for e in expvals]

# Expected phase transition according to https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.155116
if V == -2.5
    TPT_mean_field = [0.45, 0.5]
    TPT_QMC = [0.85]
end

Ts = 1 ./ βs
plt = scatter(Float64.(βs), real.(filling), label = "CE - Dcut = $(Dcut)", legend=:bottomright)
vspan!(TPT_mean_field, label = "Expected phase transition - Mean field", alpha = 0.5)
vline!(TPT_QMC, label = "Expected phase transition - QMC")
xlabel!("β")
ylabel!("Deviation from half-filling")
title!("Spinless fermion model with V = $(real(V))")
display(plt)
