using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Plots

g = 0.0
β₀, Δβ = 0.1, 0.05
maxiter = ceil(Int, (0.55 - β₀) / Δβ)

χenv, χenv_approx = 2, 2
Dcut = 2

D = 3
pspace = ℂ^2
vspace = ℂ^D
A = InfinitePEPS(pspace, vspace, vspace)[1,1]

H = PEPSKit.transverse_field_ising(PEPSKit.InfiniteSquare(); g)
times, expvals = time_evolve_model(ising_operators, (1.0, g), β₀, Δβ, maxiter, χenv; χenv_approx, Dcut, verbosity = 0, O₀ = A)

# @test E ≈ e atol = 0.2
# @test imag(magn) ≈ 0 atol = 1e-6
# @test abs(magn) ≈ mˣ atol = 5e-2

Ts = 1 ./ times
plt = scatter(Float64.(Ts), abs.(expvals))
xlabel!("T")
ylabel!("Magnetization")
title!("Ising model with g = $(real(g))")
display(plt)
