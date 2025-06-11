using Test
using TensorKit
using TensorKitTensors
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Plots
using JLD2

T = Complex{BigFloat}
setprecision(128)

J = 1.0
g = 0.1


d = Dict()

for g = [1.0]
    ce_alg = ising_operators(J, g; T = T, verbosity = 0)
    for β = [0.01 0.02 0.05]
        O = evolution_operator(ce_alg, β)
        d[(g, β)] = copy(O)
    end
end

file = jldopen("ising_CE_lambda_1.0.jld2", "w")
file["dict"] = d;
close(file)