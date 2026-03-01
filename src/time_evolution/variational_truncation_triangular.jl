struct TTNTruncation <: EnvTruncation
    trscheme::TruncationScheme
    ctm_alg::PEPSKit.CTMRGAlgorithm
    envspace::ElementarySpace
    n::Int
    tol::Float64
    k::Int # currently, k = always 2
    maxiter::Int
    check_fidelity::Bool
    verbosity::Int
end

function TTNTruncation(
        trscheme::TruncationScheme, ctm_alg::PEPSKit.CTMRGAlgorithm, envspace::ElementarySpace, n::Int;
        tol::Float64 = 1.0e-10, k::Int = 2, maxiter::Int = 100, check_fidelity::Bool = false, verbosity::Int = 0
    )
    return TTNTruncation(trscheme, ctm_alg, envspace, n, tol, k, maxiter, check_fidelity, verbosity)
end

function calculate_Ew120(T, E, W120, W60, W0)
    @tensor opt = true Ew[-1 -2; -3] := twist(E, (6, 7, 8))[18 17; -3 12 13 14 15 16] * T[18 1; -2 3 5 7 9 11] * T[1 17; -1 2 4 6 8 10] *
        W60[2 3; 12] * W0[4 5; 13] * conj(W120[6 7; 14]) * conj(W60[8 9; 15]) * conj(W0[10 11; 16])
    return Ew / norm(Ew)
end

function calculate_Ew60(T, E, W120, W60, W0)
    @tensor opt = true Ew[-1 -2; -3] := twist(E, (6, 7, 8))[18 17; 12 -3 13 14 15 16] * T[18 1; 3 -2 5 7 9 11] * T[1 17; 2 -1 4 6 8 10] *
        W120[2 3; 12] * W0[4 5; 13] * conj(W120[6 7; 14]) * conj(W60[8 9; 15]) * conj(W0[10 11; 16])
    return Ew / norm(Ew)
end

function calculate_Ew0(T, E, W120, W60, W0)
    @tensor opt = true Ew[-1 -2; -3] := twist(E, (6, 7, 8))[18 17; 12 13 -3 14 15 16] * T[18 1; 3 5 -2 7 9 11] * T[1 17; 2 4 -1 6 8 10] *
        W120[2 3; 12] * W60[4 5; 13] * conj(W120[6 7; 14]) * conj(W60[8 9; 15]) * conj(W0[10 11; 16])
    return Ew / norm(Ew)
end

function update_W(Ew, trunc_alg)
    Ew_perm = permute(Ew', ((2, 3), (1,)))
    U, _, Vᴴ = tsvd(Ew_perm; trunc = trunc_alg.trscheme)
    return U * Vᴴ
end

function upward_sweep!(Ts, Es, (W120s, W60s, W0s), (Ew120s, Ew60s, Ew0s), trunc_alg::TTNTruncation)
    for m in 1:trunc_alg.n
        Ew120m = calculate_Ew120(Ts[m], Es[m], W120s[m], W60s[m], W0s[m])
        Ew60m = calculate_Ew60(Ts[m], Es[m], W120s[m], W60s[m], W0s[m])
        Ew0m = calculate_Ew0(Ts[m], Es[m], W120s[m], W60s[m], W0s[m])
        W120m = update_W(Ew120m, trunc_alg)
        W60m = update_W(Ew60m, trunc_alg)
        W0m = update_W(Ew0m, trunc_alg)

        @tensor opt = true T[-1 -2; -3 -4 -5 -6 -7 -8] := Ts[m][-1 1; 3 5 7 9 11 13] * Ts[m][1 -2; 2 4 6 8 10 12] *
            W120m[2 3; -3] * W60m[4 5; -4] * W0m[6 7; -5] * conj(W120m[8 9; -6]) * conj(W60m[10 11; -7]) * conj(W0m[12 13; -8])
        twist!(T, (6, 7, 8))
        T /= norm(T)

        Ew120s[m] = Ew120m
        Ew60s[m] = Ew60m
        Ew0s[m] = Ew0m
        W120s[m] = W120m
        W60s[m] = W60m
        W0s[m] = W0m
        Ts[m + 1] = T
    end
    return Ts, (W120s, W60s, W0s), (Ew120s, Ew60s, Ew0s)
end

function calculate_Et(T, trunc_alg)
    @tensor pf_tensor[-6 -5 -4; -1 -2 -3] := twist(T, 2)[1 1; -1 -2 -3 -4 -5 -6]

    # pf = InfinitePartitionFunction(pf_tensor)
    # env, = leading_boundary(CTMRGEnv(pf, trunc_alg.envspace), pf, trunc_alg.ctm_alg)

    scheme = ClusterExpansions.CTM_triangular(pf_tensor)
    ClusterExpansions.run!(scheme, truncdim(dim(trunc_alg.envspace)), 100)

    pspace = codomain(T)[1]
    PEPSKit.@autoopt @tensor Et[db dt; D120 D60 D0 D300 D240 D180] := scheme.C[1][χNW D120; χN] * scheme.C[2][χN D60; χNE] * scheme.C[3][χNE D0; χSE] *
        flip(scheme.C[4], 2)[χSE D300; χS] * flip(scheme.C[5], 2)[χS D240; χSW] * flip(scheme.C[6], 2)[χSW D180; χNW] *
        twist(id(pspace), 2)[dt db]

    # env.edges[4,1,1][χ_WSW D_W; χ_WNW] *
    # env.corners[1,1,1][χ_WNW; χ_NNW] *
    # env.edges[1,1,1][χ_NNW D_N; χ_NNE] *
    # env.corners[2,1,1,][χ_NNE; χ_ENE] *
    # env.edges[2,1,1][χ_ENE D_E; χ_ESE] *
    # env.corners[3,1,1][χ_ESE; χ_SSE] *
    # env.edges[3,1,1][χ_SSE D_S; χ_SSW] *
    # env.corners[4,1,1][χ_SSW; χ_WSW] *
    # twist(id(pspace), 2)[dt db]
    return Et / norm(Et), abs(network_value_triangular(scheme))
end

function calculate_En(Et)
    return Et # Is this true for the density-operator (non-purified) case?
end

function calculate_E(E_prev, T, W120, W60, W0)
    @tensor opt = true E_new[-1 -2; -3 -4 -5 -6 -7 -8] := twist(E_prev, (6, 7, 8))[-1 1; 3 5 7 9 11 13] * T[-2 1; 2 4 6 8 10 12] *
        W120[2 -3; 3] * W60[4 -4; 5] * W0[6 -5; 7] * conj(W120[8 -6; 9]) * conj(W60[10 -7; 11]) * conj(W0[12 -8; 13])
    return E_new /= norm(E_new)
end

function downward_sweep!(Ts, Es, (W120s, W60s, W0s), (Ew120s, Ew60s, Ew0s), trunc_alg)
    for m in trunc_alg.n:-1:1
        Ew120m = calculate_Ew120(Ts[m], Es[m], W120s[m], W60s[m], W0s[m])
        Ew60m = calculate_Ew60(Ts[m], Es[m], W120s[m], W60s[m], W0s[m])
        Ew0m = calculate_Ew0(Ts[m], Es[m], W120s[m], W60s[m], W0s[m])
        Ew120s[m] = Ew120m
        Ew60s[m] = Ew60m
        Ew0s[m] = Ew0m

        W120m = update_W(Ew120m, trunc_alg)
        W60m = update_W(Ew60m, trunc_alg)
        W0m = update_W(Ew0m, trunc_alg)
        W120s[m] = W120m
        W60s[m] = W60m
        W0s[m] = W0m

        if m > 1
            Emm1 = calculate_E(Es[m], Ts[m], W120m, W60m, W0m)
            Es[m - 1] = Emm1
        end
    end
    return Es, (W120s, W60s, W0s), (Ew120s, Ew60s, Ew0s)
end

function check_convergence(Ws, Ews)
    E = scalartype(Ws)
    Zs = Vector{E}(undef, length(Ws))
    for (i, (W, Ew)) in enumerate(zip(Ws, Ews))
        Zs[i] = @tensor Ew[1 2; 3] * W[1 2; 3]
    end
    return sum([abs(Zs[i] - Zs[1]) for i in 2:length(Zs)]) / length(Zs)
    # println("Zs = ", Zs)
    # return maximum(real.(Zs))
end

function approximate_state(
        T₀::AbstractTensorMap{E, S, 2, 6},
        trunc_alg::TTNTruncation
    ) where {E, S <: ElementarySpace}
    T = scalartype(T₀)
    pspace = codomain(T₀)[1]
    vspace = domain(T₀)[1]
    E₀ = randn(T, pspace' ⊗ pspace, vspace' ⊗ vspace' ⊗ vspace' ⊗ vspace ⊗ vspace ⊗ vspace)
    Ts = fill(T₀, trunc_alg.n + 1)
    Es = fill(E₀, trunc_alg.n)
    W120s = W60s = W0s = fill(randn(T, vspace ⊗ vspace, vspace), trunc_alg.n)
    Ew120s = Ew60s = Ew0s = fill(randn(T, vspace' ⊗ vspace', vspace'), trunc_alg.n)
    ϵ = Inf
    ϵs = [ϵ]
    for iter in 1:trunc_alg.maxiter
        if trunc_alg.verbosity > 1
            @info "Iteration $iter, ϵ = $ϵ"
        end
        Ts, (W120s, W60s, W0s), (Ew120s, Ew60s, Ew0s) = upward_sweep!(Ts, Es, (W120s, W60s, W0s), (Ew120s, Ew60s, Ew0s), trunc_alg)
        Et, ϵ = calculate_Et(Ts[end], trunc_alg)
        En = calculate_En(Et)
        Es[end] = En
        Es, (W120s, W60s, W0s), (Ew120s, Ew60s, Ew0s) = downward_sweep!(Ts, Es, (W120s, W60s, W0s), (Ew120s, Ew60s, Ew0s), trunc_alg)

        # ϵ = calculate_Z(Ts[end])
        # ϵ = check_convergence(W0s, Ew0s)
        push!(ϵs, ϵ)
        if ϵ < -trunc_alg.tol
            if trunc_alg.verbosity > 1
                @info "Converged after $iter iterations with ϵ = $ϵ"
            end
            return Ts
        end

    end
    if trunc_alg.verbosity > 0
        @warn "Not converged after $(trunc_alg.maxiter) iterations with ϵ = $ϵ"
    end
    println("epsilons: ", ϵs)
    return Ts
end
