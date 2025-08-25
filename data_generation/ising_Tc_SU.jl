using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using PEPSKit
using Plots
using JLD2

function get_data_ising_Tc(Dcut, χenv)
    g = 2.5
    trscheme = truncdim(Dcut)

    Δβ = 0.05
    dt = 1e-3
    max_beta = 1.0
    maxiter_SU = ceil(Int, max_beta / Δβ)
    time_alg_SU = (dt = dt, Δt = Δβ, maxiter = maxiter_SU)

    βs, mzs, mxs_SU, ξs, δs, As_SU = data_generation_ising_SU(time_alg_SU, trscheme, χenv; g = g)

    file = jldopen("Ising_Tc_Trotter_vs_CE_D_$(Dcut)_χenv_$(χenv).jld2", "w")
    file["mz"] = mzs
    file["ξs"] = ξs
    file["δs"] = δs
    file["βs"] = βs
    close(file)

    plt = scatter(βs, abs.(mzs), label = "SU Dcut=$Dcut χenv=$χenv", markersize = 3)
    display(plt)

    plt = scatter(βs, ξs, label = "SU Dcut=$Dcut χenv=$χenv", markersize = 3)
    display(plt)
end

Dcuts = [2 4 6 8 10 12 14]
χenvs = [16]

for Dcut = Dcuts
    for χenv = χenvs
        get_data_ising_Tc(Dcut, χenv)
    end
end
