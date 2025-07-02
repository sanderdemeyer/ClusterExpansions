using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Plots
using MPSKit
using JLD2

J = 1.0
g = 3.04438
z = 0.0

χenv = 50
time_alg = UniformTimeEvolution(0.1, 0.02, 45; verbosity = 2)

pspace = ℂ^2
H = localoperator_model(pspace, σᶻ())

@tensor M[-1 -2; -3 -4] := σᶻ()[-1; -3] * σᶻ()[-2; -4]

# ce_alg = ising_operators_Z2(J; T = ComplexF64, verbosity = 0)
ce_alg = ising_operators(J, g, z; T = Complex{BigFloat}, verbosity = 0)
envspace = ce_alg.envspace(χenv)

vumps_alg = VUMPS(; verbosity = 0, maxiter = 500)
ctm_alg = SimultaneousCTMRG(; maxiter = 150)

function nothing_function(a)
    return 0.0
end

function data_collapse_generation(ce_alg, time_alg, trunc_alg, χenv, name)
    # times, expvals, As = ClusterExpansions.time_evolve(ce_alg, time_alg, trunc_alg, O -> observable_time_evolve(O, σᶻ(), χenv, vumps_alg))
    times, expvals, As = ClusterExpansions.time_evolve(ce_alg, time_alg, trunc_alg, nothing_function)
    ξs = []
    δs = []
    for O = As[5:end]
        pf_asym = PEPSKit.trace_out(InfinitePEPO(O))
        T = InfiniteMPO([pf_asym[1,1]])
        
        # @tensor MO[-4 -3; -1 -2] := twist(O, 2)[1 2; -1 -2 -3 -4] * M[2; 1]
        # T2 = InfiniteMPO([MO])
        
        virtualspace = ℂ^χenv
        pspace = domain(pf_asym[1,1])[2]
        
        mps = InfiniteMPS([
            randn(
                ComplexF64,
                virtualspace * pspace,
                virtualspace,
            )])
        # env_CTMRG, = leading_boundary(CTMRGEnv(pf_asym, virtualspace), pf_asym, ctm_alg);
        mps, env, ϵ = leading_boundary(mps, T, vumps_alg);
        # @tensor M[-1 -2; -3 -4] := id(ℂ^2)[-1; -3] * σᶻ()[-2; -4]

        # E_num = PEPSKit.@autoopt @tensor O[dL1 dL2; DLN Dc DLS DLW] * O[dR1 dR2; DRN DRE DRS Dc] * mps.AC[1][DtL DLN; Dt] * 
        # conj(mps.AC[1][DbL DLS; Db]) * mps.AR[2][Dt DRN; DtR] * conj(mps.AR[2][Db DRS; DbR]) * M[dL2 dR2; dL1 dR1] * 
        # env.GLs[1][DbL DLW; DtL] * env.GRs[1][DtR DRE; DbR]
        # E_denom = PEPSKit.@autoopt @tensor O[dL dL; DLN Dc DLS DLW] * O[dR dR; DRN DRE DRS Dc] * mps.AC[1][DtL DLN; Dt] * 
        # conj(mps.AC[1][DbL DLS; Db]) * mps.AR[2][Dt DRN; DtR] * conj(mps.AR[2][Db DRS; DbR]) *
        # env.GLs[1][DbL DLW; DtL] * env.GRs[1][DtR DRE; DbR]
        # println(E_num / E_denom)
        # push!(ms, E_num/E_denom)
        # λ = correlation_length(mps)
        # λ_CTMRG = correlation_length(pf_asym, env_CTMRG; num_vals = 5)
        # println("Correlation length CTMRG: ", λ_CTMRG)
        ϵ, δ, = marek_gap(mps; num_vals = 20)
        push!(ξs, 1 / ϵ)
        push!(δs, δ)
    end
    file = jldopen(name, "w")
    file["Ts"] = Float64.(times)
    file["ξs"] = Float64.(ξs)
    file["δs"] = Float64.(δs)
    # file["xdata"] = Float64.(xdata)
    # file["ydata"] = Float64.(ydata)
    close(file)
end

χenvs = [10 20 30 40]
Dcuts = [4]
schmidt_cuts = [Inf Inf]
for (schmidt_cut, Dcut) in zip(schmidt_cuts, Dcuts)
    trscheme = truncdim(Dcut) # & truncbelow(schmidt_cut)
    trunc_alg = NoEnvTruncation(trscheme)
    for χenv in χenvs
        name = "Ising_vumps_critical_exponents_g_$(g)_z_$(z)_cut_$(schmidt_cut)_Dcut_$(Dcut)_χ_$(χenv).jld2"
        println(name)
        data_collapse_generation(ce_alg, time_alg, trunc_alg, χenv, name)
    end
end