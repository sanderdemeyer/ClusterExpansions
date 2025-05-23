using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Plots

(Jx, Jy, Jz) = (-1.0, 1.0, -1.0)
h = 0.0

e = -0.6602310934799577

β₀, Δβ = 0.1, 0.05
maxiter = 10

χenv, χenv_approx = 2, 2
Dcut = 2

D = 3
pspace = ℂ^2
vspace = ℂ^D
A = InfinitePEPS(pspace, vspace, vspace)[1,1]

if h != 0.0
    @warn "LocalOperator defined for h = 0.0, whereas true value is $(h)"
end
H = PEPSKit.heisenberg_XYZ(; Jx, Jy, Jz)
times, expvals = time_evolve_model(heisenberg_operators, (Jx, Jy, Jz, h), β₀, Δβ, maxiter, χenv; χenv_approx, Dcut, verbosity = 3, O₀ = nothing)

# @test E ≈ e atol = 0.2
# @test imag(magn) ≈ 0 atol = 1e-6
# @test abs(magn) ≈ mˣ atol = 5e-2

Ts = 1 ./ times
plt = scatter(Float64.(Ts), abs.(expvals))
xlabel!("T")
ylabel!("Magnetization")
title!("Heisenberg model with J = ($(Jx), $(Jy), $(Jz)), h = $(h)")
display(plt)

