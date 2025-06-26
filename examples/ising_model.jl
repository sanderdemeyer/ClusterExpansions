using TensorKit
using MPSKitModels
using ClusterExpansions
using PEPSKit
using Plots

# Set up the Ising model
model = ising_operators
J = 1.0 # nearest-neighbour ZZ interaction
g = 0.0 # transverse field along the x-axis
z = 0.0 # explicit symmetry-breaking fielda along the z-axis
model_param = (J, g, z)

χenv = 30 # Environment bond dimension used in the calculation of expectation values

# Parameters in the truncation scheme
schmidt_cut = 1.5
Dcut = 4
trscheme = NoEnvTruncation
trunc = truncbelow(10.0^(-schmidt_cut)) & truncdim(Dcut)
trscheme_parameters = (trunc,)
trscheme_kwargs = ()

# Set up time evolution algorithm
β₀ = 0.1
Δβ = 0.1
maxiter = 9
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

# Define the arguments of the cluster cluster
# Default is T = BigFloat{Float64}, which is recommended for small values of β
ce_kwargs = (T = ComplexF64,)

# Define observables, which can be both LocalOperators and functions
pspace = ℂ^2
Hz = localoperator_model(pspace, σᶻ())
Hx = localoperator_model(pspace, σˣ())

# Perform the actual time evolution.
# `times` is a list of times at which the expectation values are computed
# 'expvals' are the expectation values. Every element of `expvals` is a list of the expectation values at the corresponding time
# `Os` is a list of PEPO density operators
times, expvals, Os = time_evolve_model(model, model_param, time_alg, χenv, trscheme, trscheme_parameters; trscheme_kwargs, ce_kwargs, observables = [Hz Hx]);

# Extract the expectation values
mz = [e[1] for e in expvals]
mx = [e[2] for e in expvals]

# Plot them
Tc = 2/(log(1+sqrt(2))) # Critical temperature for the Ising model without transverse field
Ts = 1 ./ times
plt = scatter(Float64.(Ts), abs.(mx), label = "<X>")
scatter!(Float64.(Ts), abs.(mz), label = "<Z>")
vline!([Tc], label = "Tc")
xlabel!("T")
ylabel!("Magnetization")
title!("Ising model with g = $(real(g))")
display(plt)
