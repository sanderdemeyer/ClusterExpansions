using TensorKit
using MPSKitModels
using ClusterExpansions
using PEPSKit
using Plots

# Set up the Heisenberg model
model = heisenberg_operators
Jx = 1.0
Jy = 1.0
Jz = 1.0
z = 0.0 # explicit symmetry-breaking fielda along the z-axis
model_param = (Jx, Jy, Jz, z)

χenv = 16 # Environment bond dimension used in the calculation of expectation values

# Parameters in the truncation scheme
Dcut = 4
trscheme = NoEnvTruncation
trunc = truncdim(Dcut)
trscheme_parameters = (trunc,)
trscheme_kwargs = ()

# Set up time evolution algorithm
β₀ = 1e-1
Δβ = 1e-1
maxiter = 30
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

# Define the arguments of the cluster expansion algorithm
# Default is T = BigFloat{Float64}, which is recommended for small values of β
spin_symmetry = U1Irrep
ce_kwargs = (T = Complex{BigFloat}, symmetry = "C4", spin_symmetry = spin_symmetry)

# Define observables, which can be both LocalOperators and functions
H = heisenberg_XYZ(ComplexF64, spin_symmetry, InfiniteSquare(); Jx=1, Jy=1, Jz=1)
# Perform the actual time evolution.
# `βs` is a list of times/βs at which the expectation values are computed
# 'expvals' are the expectation values. Every element of `expvals` is a list of the expectation values at the corresponding time
# `Os` is a list of PEPO density operators
βs, expvals, Os = time_evolve_model(model, model_param, time_alg, χenv, trscheme, trscheme_parameters; trscheme_kwargs, ce_kwargs, observables = [H]);

# Extract the expectation values
energies = [e[1] for e in expvals]

# Plot them
# plt = scatter(Float64.(βs), real.(energies), label = "Energies CE")
plt = scatter(Float64.(βs), real.(energies), label = "Energies CE")
xlabel!("T")
ylabel!("E")
title!("Heisenberg model")
display(plt)
