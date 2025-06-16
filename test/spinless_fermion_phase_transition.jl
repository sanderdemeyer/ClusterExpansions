using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Plots
using TensorKitTensors

t = 1.0
V = -1.0
μ = 2*V

β₀, Δβ = 0.01, 0.01
maxiter = ceil(Int, (1.0 - β₀) / Δβ)

χenv, χenv_approx = 24, 2
trscheme = truncdim(4)


pspace = Vect[fℤ₂](0 => 1, 1 => 1)
(Nr, Nc) = (1, 1)
pspaces = fill(pspace, Nr, Nc)
lattice = InfiniteSquare(Nr, Nc)

H_hop = PEPSKit.LocalOperator(pspaces, (neighbor => FermionOperators.f_hop() for neighbor in PEPSKit.nearest_neighbours(lattice))...,)
H_num = PEPSKit.LocalOperator(pspaces, ((idx,) => FermionOperators.f_num() for idx in PEPSKit.vertices(lattice))...,)

time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

times, expvals, As = time_evolve_model(spinless_fermion_operators, (t, V, μ), time_alg, χenv; trscheme, observables = [H_num H_hop], T = Complex{BigFloat});

Ts = 1 ./ times
plt = scatter(Float64.(times), [real(expvals_6[i][1]) for i = 1:length(expvals_6)], label = "D = 6")
scatter!(Float64.(times), [real(expvals_7[i][1]) for i = 1:length(expvals)], label = "D = 7")
scatter!(Float64.(times), [real(expvals_4[i][1]) for i = 1:length(expvals)], label = "D = 4")
scatter!(Float64.(times), [real(expvals_3[i][1]) for i = 1:length(expvals)], label = "D = 3")
scatter!(Float64.(times), [real(expvals_2[i][1]) for i = 1:length(expvals)], label = "D = 2")
xlabel!("T")
# xlims!(plt, (1.0, 4.0))
ylabel!("Magnetization")
title!("Spinless fermion model with V = $(real(V))")
display(plt)


plt = scatter(Float64.(times), [real(expvals[i][2]) for i = 1:length(expvals)])
xlabel!("T")
# xlims!(plt, (1.0, 4.0))
ylabel!("Hopping")
title!("Spinless fermion model with V = $(real(V))")
display(plt)
