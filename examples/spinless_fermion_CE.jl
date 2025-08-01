using TensorKit
using TensorKitTensors
using ClusterExpansions
using PEPSKit
using Plots
using JLD2
using MPSKit

function get_both_observables(O::InfinitePEPO, virtualspace, vumps_alg, M)
    pf = PEPSKit.trace_out(O)
    T = InfiniteMPO([pf[1,1]])

    pspace = domain(pf[1,1])[2]

    mps = InfiniteMPS([
        randn(
            ComplexF64,
            virtualspace * pspace,
            virtualspace,
        )])
    mps, env, _ = leading_boundary(mps, T, vumps_alg);
    ϵ, δ, = marek_gap(mps; num_vals = 20)
    
    t = O[1,1]
    E_num = PEPSKit.@autoopt @tensor twist(t,2)[d1 d2; DN DE DS DW] * mps.AC[1][DtL DN; DtR] * 
    conj(mps.AC[1][DbL DS; DbR]) * M[d2; d1] * 
    env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR]
    E_denom = PEPSKit.@autoopt @tensor twist(t,2)[d d; DN DE DS DW] * mps.AC[1][DtL DN; DtR] * 
    conj(mps.AC[1][DbL DS; DbR]) *
    env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR]
    return E_num / E_denom, 1 / ϵ

end

function get_data_CE(Dcut, χenv)
    t = 1.0 # Nearest-neighbour hopping term
    V = -2.5 # Interaction term. Repulsive if positive.
    μ = 0.0 # Extra chemical potential on top of half-filling
    @assert μ == 0.0 "mu set to zero to have no onsite interaction"

    envspace = Vect[fℤ₂](0 => div(χenv,2), 1 => div(χenv,2))

    model = spinless_fermion_operators
    model_param = (t, V, μ)

    # Parameters in the truncation scheme
    trunc = truncdim(Dcut)

    trscheme = NoEnvTruncation
    trscheme_parameters = (trunc,)

    # Set up time evolution algorithm
    β₀ = 0.1
    Δβ = 0.01
    maxiter = 80
    time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

    ce_kwargs = (symmetry = nothing, T = Complex{BigFloat})

    # Define observables, which can be both LocalOperators and functions
    H_both = (O, envspace, ctm_alg) -> get_both_observables(InfinitePEPO(O), envspace, VUMPS(; maxiter = 500, verbosity = 0), FermionOperators.f_num())

    βs, expvals, As = time_evolve_model(model, model_param, time_alg, χenv, trscheme, trscheme_parameters; observables = [H_both], ce_kwargs, verbosity_ctm = 2);

    # Extract the expectation values
    occupancies = [e[1][1] for e in expvals]
    corrlengths = [e[1][2] for e in expvals]
    println(occupancies)
    println(corrlengths)

    file = jldopen("SF_CE_Dcut_$(Dcut)_χenv_$(χenv).jld2", "w")
    file["βs"] = βs
    file["occupancies"] = occupancies
    file["corrlengths"] = corrlengths
end


Dcuts = [3]
χenvs = [16]

for Dcut = Dcuts
    for χenv = χenvs
        get_data_CE(Dcut, χenv)
    end
end