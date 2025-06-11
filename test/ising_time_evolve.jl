using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Plots

J = 1.0
g = 0.0

χenv, χenv_approx = 50, 6
trscheme = truncbelow(1e-8) & truncdim(4)
time_alg = UniformTimeEvolution(0.1, 0.1, 9)
H = localoperator_model(ℂ^2, σᶻ())
times, expvals, As = time_evolve_model(ising_operators, (J, g), time_alg, χenv; χenv_approx, trscheme, observables = [H]);

magnetizations = [e[1] for e in expvals]

Tc = 2/(log(1+sqrt(2)))
Ts = 1 ./ times
plt = scatter(Float64.(Ts), abs.(magnetizations))
vline!([Tc])
xlabel!("T")
xlims!(plt, (1.0, 4.0))
ylabel!("Magnetization")
title!("Ising model with g = $(real(g))")

display(plt)
