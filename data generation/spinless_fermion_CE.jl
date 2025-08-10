using TensorKit
using TensorKitTensors
using ClusterExpansions
using PEPSKit
using Plots
using MPSKit
using JLD2

# Set up time evolution algorithm
β₀ = 0.1
Δβ = 0.02
max_beta = 1.5
maxiter = ceil(Int, (max_beta - β₀) / Δβ) # Go up to a value of β = 0.9
# time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)
time_alg = StaticTimeEvolution(β₀, [0.1, 0.005], vcat(fill(1, 4), fill(2, 80), fill(1, 6)))

V = -2.5

χenvs = [10]
Dcuts = [4]
for Dcut in Dcuts
    trunc_alg = NoEnvTruncation(truncdim(Dcut))
    for χenv in χenvs
        name = "Spinless_Fermions_vumps_V_$(V)_Dcut_$(Dcut)_χ_$(χenv)_max_$(max_beta).jld2"
        spinless_fermion_model_CE(time_alg, trunc_alg, χenv; V, name, saving = true)
    end
end
