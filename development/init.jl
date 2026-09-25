using TensorKit
using TensorKitTensors
using ClusterExpansions
using PEPSKit
using MPSKit
using JLD2
using LinearAlgebra

BLAS.set_num_threads(1)

function evolve(ce_alg, trunc_alg, Δβ, maxiter, χpeps, χenv, name, frequency)
    vumps_alg = VUMPS(; maxiter = 200, verbosity = 1)
    ctm_alg = SimultaneousCTMRG(; maxiter = 200, verbosity = 1)
    envspace = ℂ^χenv
    ΔO = ClusterExpansions.evolution_operator_cubic(ce_alg, Δβ; T_conv = Float64);
    O = ClusterExpansions.evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64);
    # println("imag/real = $(norm(imag(O))) / $(norm(real(O)))")
    # println("imag/real = $(norm(imag(ΔO))) / $(norm(real(ΔO)))")
    # ΔO = real(ΔO)
    # O = real(O)
    ns = []
    for i = 1:maxiter
        @info "i = $i / $maxiter"
        O, = approximate_state((O, ΔO), trunc_alg)
        O /= norm(O)
        # println(test_rotinvariance(O))
        if (i % frequency == 0)
            file = jldopen(name(round(i*Δβ, digits = 2)), "w")
            file["O"] = O
            close(file)
        end        
    end
    return O
end

function get_data(Dcut, max_beta, μ, V; delta = 0.0)
    t = 1.0
    U = 40.0
    cutoff = 2
    χpeps = 4
    χenv = 16

    trunc_alg = NoEnvTruncation(truncdim(Dcut))

    β₀ = 0.0
    Δβ = 0.01
    maxiter = ceil(Int, (max_beta - β₀) / Δβ)
    time_alg_CE = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 2)

    frequency = 2
    ce_alg = bose_hubbard_operators(t, U, V, μ; δ = delta, T = Float64, cutoff, symmetry = "C4_manifest", svd = false)
    name = β -> "bh_init_symmetry_broken_delta_$(δ)/D_$(Dcut)/bose_Hubbard_cutoff_$(cutoff)_U_40_V_$(round(V,digits = 3))_mu_$(round(μ, digits = 1))_β_$(β)_Dcut_$(Dcut).jld2"
    O = evolve(ce_alg, trunc_alg, Δβ, maxiter, χpeps, χenv, name, frequency)
    return
end

beta = 0.6
Dcut = 5
for mu = [34.0, 40.0]
    for V = [0.0, 0.667]
        for delta = [0.01, 0.02, 0.04]
            get_data(Dcut, beta, mu, V; delta)
        end
    end
end