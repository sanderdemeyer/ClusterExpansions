using TensorKit
using TensorKitTensors
using ClusterExpansions
using PEPSKit
using Plots
using MPSKit: VUMPS
using JLD2

# Set up time evolution algorithm
β₀ = 0.1
Δβ = 0.02
max_beta = 1.5
maxiter = ceil(Int, (max_beta - β₀) / Δβ) # Go up to a value of β = 0.9
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

V = -2.5

function SF_data_generation(time_alg, trscheme, trscheme_parameters, χenv, name; V = -2.5)
    # Set up the SF model
    model = spinless_fermion_operators
    t = 1.0 # Nearest-neighbour hopping term
    μ = 0.0 # Extra chemical potential on top of half-filling
    model_param = (t, V, μ)

    ce_kwargs = (symmetry = nothing, T = Complex{BigFloat})

    # Define observables, which can be both LocalOperators and functions
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    H_num = localoperator_model(pspace, FermionOperators.f_num())
    H_ξs = (O, envspace, ctm_alg) -> get_marek_gap(O, envspace, VUMPS(; maxiter = 500, verbosity= 0))

    βs, expvals, As = time_evolve_model(model, model_param, time_alg, χenv, trscheme, trscheme_parameters; observables = [H_num, H_ξs], ce_kwargs);

    # Extract the expectation values
    fillings = [abs(e[1]-0.5) for e in expvals]
    ξs = [e[2][1] for e in expvals]
    δs = [e[2][2] for e in expvals]

    file = jldopen(name, "w")
    file["βs"] = Float64.(βs)
    file["ns"] = Float64.(fillings)
    file["ξs"] = Float64.(ξs)
    file["δs"] = Float64.(δs)
    close(file)

end

χenvs = [10]
Dcuts = [4]
for Dcut in Dcuts
    trscheme = NoEnvTruncation
    trscheme_parameters = (truncdim(Dcut),)
    for χenv in χenvs
        name = "SF_vumps_V_$(V)_Dcut_$(Dcut)_χ_$(χenv)_max_$(max_beta).jld2"
        SF_data_generation(time_alg, trscheme, trscheme_parameters, χenv, name; V)
    end
end