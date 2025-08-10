using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using PEPSKit
using Plots

# Set up the Ising model
J = 1.0 # nearest-neighbour ZZ interaction
g = 0.0 # transverse field along the x-axis
z = 0.0 # explicit symmetry-breaking fielda along the z-axis
ce_alg = ising_operators(J, g, z; T = Complex{BigFloat}, symmetry = "C4")

χenv = 8 # Environment bond dimension used in the calculation of expectation values

# Parameters in the truncation scheme
Dcut = 4
trunc_alg = NoEnvTruncation(truncdim(Dcut); verbosity = 0)

# Set up time evolution algorithm
β₀ = 0.1
Δβ = 0.1
maxiter = 8
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

# Define observables
vumps_alg = VUMPS(; maxiter = 100, verbosity = 0)
observables = PEPO_observables([SpinOperators.σᶻ(), SpinOperators.σˣ()], vumps_alg)
observable = O -> ClusterExpansions.calculate_observables(O, χenv, observables)

# Perform the actual time evolution.
# `βs` is a list of times/βs at which the expectation values are computed
# 'expvals' are the expectation values. Every element of `expvals` is a list of the expectation values at the corresponding time
# `Os` is a list of PEPO density operators
βs, expvals, Os = time_evolve(ce_alg, time_alg, trunc_alg, observable);

# Extract the expectation values
mz = [e[1] for e in expvals]
mx = [e[2] for e in expvals]

# Plot them
Tc = 2/(log(1+sqrt(2))) # Critical temperature for the Ising model without transverse field
Ts = 1 ./ βs
plt = scatter(Float64.(Ts), abs.(mx), label = "<X>")
scatter!(Float64.(Ts), abs.(mz), label = "<Z>")
vline!([Tc], label = "Tc")
xlabel!("T")
ylabel!("Magnetization")
title!("Ising model with g = $(real(g))")
display(plt)
