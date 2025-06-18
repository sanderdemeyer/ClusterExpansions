using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Plots
using JLD2

function data_collapse_generation(χenv, schmidt_cut, Dcut)
    J = 1.0
    g = 3.04438 # Quantum phase transition dynamics in the two-dimensional transverse-field Ising model. https://www.science.org/doi/10.1126/sciadv.abl6850
    z = 0.0
    trscheme = truncbelow(10.0^(-schmidt_cut)) & truncdim(Dcut)

    β₀ = 0.1
    maxiter = 20
    time_alg = UniformTimeEvolution(β₀, β₀, maxiter; verbosity = 3)

    H = localoperator_model(ℂ^2, σᶻ())

    times, expvals, As = time_evolve_model(ising_operators, (J, g, z), time_alg, χenv; trscheme, observables = [H], T = Complex{BigFloat});

    magnetizations = [e[1] for e in expvals]

    Tc = 0.0
    ν = 0.62997097

    Ts = 1 ./ times

    ξs = []
    δs = []
    for A = As
        pf = PEPSKit.trace_out(InfinitePEPO(A));
        env,  = leading_boundary(CTMRGEnv(pf, ℂ^χenv), pf, SimultaneousCTMRG(; maxiter = 1000));
        ξ_h, _, λ_h, _ = correlation_length(pf, env; num_vals = 4);
        println("ξ_h = $ξ_h, λ_h = $λ_h")
        println(map(λ -> -1 / log(abs(λ[3])), λ_h)[1])
        println(map(λ -> log(abs(λ[2])) - log(abs(λ[3])), λ_h)[1])
        if ξ_h[1] > 1e14
            push!(ξs, map(λ -> -1 / log(abs(λ[3])), λ_h)[1])
        else
            push!(ξs, ξ_h[1])
        end
        push!(δs, map(λ -> log(abs(λ[2])) - log(abs(λ[3])), λ_h)[1])
    end
    # ξs[1:4] .= 0
    xdata = [(T-Tc)*δ^(-1/ν) for (T, δ) in zip(Ts, δs)]
    ydata = [log(ξ*δ) for (ξ, δ) in zip(ξs, δs)]

    file = jldopen("Ising_critical_exponents_g_$(g)_z_$(z)_cut_$(schmidt_cut)_Dcut_$(Dcut)_χ_$(χenv).jld2", "w")
    file["Ts"] = Float64.(Ts)
    file["magnetizations"] = Float64.(abs.(magnetizations))
    file["ξs"] = Float64.(ξs)
    file["δs"] = Float64.(δs)
    file["xdata"] = Float64.(xdata)
    file["ydata"] = Float64.(ydata)
    close(file)

    plt = scatter(Float64.(Ts), abs.(magnetizations), label = "4")
    # scatter!(Float64.(Ts), abs.(magnetizations_3), label = "3")
    vline!([Tc])
    xlabel!("T")
    xlims!(plt, (1.0, 1.6))
    ylabel!("Magnetization")
    title!("Ising model with g = $(real(g))")
    savefig("ising_critical_exponents_g_$(g)_z_$(z)_cut_$(schmidt_cut)_Dcut_$(Dcut)_χ_$(χenv)_magnetization.png")
    display(plt)

    plt = scatter(Float64.(Ts), ξs, label = "4")
    # scatter!(Float64.(Ts), abs.(magnetizations_3), label = "3")
    vline!([Tc])
    xlims!(plt, (0.0, 1.0))
    ylims!(plt, (0.0, 10.0))
    xlabel!("T")
    ylabel!("Correlation length")
    title!("Ising model with g = $(real(g))")
    savefig("ising_critical_exponents_g_$(g)_z_$(z)_cut_$(schmidt_cut)_Dcut_$(Dcut)_χ_$(χenv)_corrlength.png")
    display(plt)

    plt = scatter(xdata[6:end], ydata[6:end])
    xlabel!("(T-Tc)*δ^(-1/ν)")
    ylabel!("log(ξδ)")
    title!("Ising model with g = $(real(g))")
    xlims!(plt, (0, 250))
    savefig("ising_critical_exponents_g_z_$(z)_$(g)_cut_$(schmidt_cut)_Dcut_$(Dcut)_χ_$(χenv)_datacollaps.png")
    display(plt)
end

χenvs = [10 20 30 40]
Dcuts = [4 6 8 10 12]
schmidt_cuts = [1.0 1.25 1.5 1.75 2.0]
for χenv in χenvs
    for (schmidt_cut, Dcut) in zip(schmidt_cuts, Dcuts)
        data_collapse_generation(χenv, schmidt_cut, Dcut)
    end
end