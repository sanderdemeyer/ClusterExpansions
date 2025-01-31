using LinearAlgebra
abstract type stopcrit end

struct maxiter_TNR <: stopcrit
    n::Int
end

struct convcrit_TNR <: stopcrit
    Δ::Float64
    f::Function
end

struct MultipleCrit_TNR <: stopcrit
    crits::Vector{stopcrit}
end

(crit::maxiter_TNR)(steps::Int, data) = steps < crit.n
(crit::convcrit_TNR)(steps::Int, data) = crit.Δ < crit.f(steps, data)

function pseudopow(t, a::Real; tol=eps(scalartype(t))^(3 / 4))
    t′ = copy(t)
    for (c, b) in blocks(t′)
        @inbounds for I in LinearAlgebra.diagind(b)
            b[I] = b[I] < tol ? b[I] : b[I]^a
        end
    end
    return t′
end

struct EntanglementFiltering
    criterion::stopcrit
    function EntanglementFiltering(criterion::stopcrit)
        return new(criterion)
    end
    function EntanglementFiltering()
        f = (steps, data) -> data
        crit = maxiter_TNR(10) & convcrit_TNR(1e-10, f)
        return new(crit)
    end
end

struct LoopOptimization
    criterion::stopcrit
    verbosity::Int
    function LoopOptimization(criterion::stopcrit)
        return new(criterion)
    end
    function LoopOptimization()
        f = (steps, data) -> data
        crit = maxiter_TNR(50) & convcrit_TNR(1e-12, f)
        return new(crit)
    end
end

mutable struct LoopTNR
    TA::TensorMap
    TB::TensorMap

    entanglement_alg::EntanglementFiltering
    loop_alg::LoopOptimization

    finalize!::Function
    function LoopTNR(T; entanglement_alg=EntanglementFiltering(),
                     loop_alg=LoopOptimization(), finalize=finalize_TNR!)
        return new(copy(T), copy(T), entanglement_alg, loop_alg, finalize)
    end
end

# Entanglement filtering step 
function QR_L(L::TensorMap, T::AbstractTensorMap{S,2,2}) where {S}
    @tensor temp[-1 -2; -3 -4] := L[-2; 1] * T[-1 1; -3 -4]
    _, Rt = leftorth(temp, (1, 2, 4), (3,))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{S,2,2}) where {S}
    @tensor temp[-1 -2; -3 -4] := T[-1 -2; 1 -4] * R[1; -3]
    Lt, _ = rightorth(temp, (2,), (1, 3, 4))
    return Lt
end

function QR_L(L::TensorMap, T::AbstractTensorMap{S,1,3}) where {S}
    @tensor temp[-1; -2 -3 -4] := L[-1; 1] * T[1; -2 -3 -4]
    _, Rt = leftorth(temp, (1, 3, 4), (2,))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{S,1,3}) where {S}
    @tensor temp[-1; -2 -3 -4] := T[-1; 1 -3 -4] * R[1; -2]
    Lt, _ = rightorth(temp, (1,), (2, 3, 4))
    return Lt
end

function QR_L(L::TensorMap, T::AbstractTensorMap{S,1,2}) where {S}
    @tensor temp[-1; -2 -3] := L[-1; 1] * T[1; -2 -3]
    _, Rt = leftorth(temp, (1, 3), (2,))
    return Rt
end

function QR_R(R::TensorMap, T::AbstractTensorMap{S,1,2}) where {S}
    @tensor temp[-1; -2 -3] := T[-1; 1 -3] * R[1; -2]
    Lt, _ = rightorth(temp, (1,), (2, 3))
    return Lt
end

function find_L(pos::Int, ψ::Array, scheme::LoopTNR)
    L = id(space(ψ[pos])[1])

    crit = true
    steps = 0
    error = Inf
    n = length(ψ)

    while crit
        new_L = copy(L)
        for i in (pos - 1):(pos + n - 2)
            new_L = QR_L(new_L, ψ[i % n + 1])
        end
        new_L = new_L / norm(new_L)

        if space(new_L) == space(L)
            error = abs(norm(new_L - L))
        end

        L = new_L
        steps += 1
        crit = scheme.entanglement_alg.criterion(steps, error)
    end
    return L
end

function find_R(pos::Int, ψ::Array, scheme::LoopTNR)
    R = id(space(ψ[mod(pos - 2, 4) + 1])[2]')
    crit = true
    steps = 0
    error = Inf
    n = length(ψ)

    while crit
        new_R = copy(R)

        for i in (pos - 2):-1:(pos - n - 1)
            new_R = QR_R(new_R, ψ[mod(i, n) + 1])
        end

        new_R = new_R / norm(new_R)

        if space(new_R) == space(R)
            error = abs(norm(new_R - R))
        end
        R = new_R
        steps += 1
        crit = scheme.entanglement_alg.criterion(steps, error)
    end

    return R
end

function P_decomp(R::TensorMap, L::TensorMap)
    @tensor temp[-1; -2] := L[-1; 1] * R[1; -2]
    U, S, V, _ = tsvd(temp, (1,), (2,))
    re_sq = pseudopow(S, -0.5)

    @tensor PR[-1; -2] := R[-1, 1] * adjoint(V)[1; 2] * re_sq[2, -2]
    @tensor PL[-1; -2] := re_sq[-1, 1] * adjoint(U)[1; 2] * L[2, -2]

    return PR, PL
end

function find_projectors(ψ::Array, scheme::LoopTNR)
    PR_list = []
    PL_list = []

    for i in eachindex(ψ)
        L = find_L(i, ψ, scheme)

        R = find_R(i, ψ, scheme)

        pr, pl = P_decomp(R, L)

        push!(PR_list, pr)
        push!(PL_list, pl)
    end
    return PR_list, PL_list
end

function finalize_TNR!(scheme::LoopTNR)
    n = norm(@tensor opt = true scheme.TA[1 2; 3 4] * scheme.TB[3 5; 1 6] *
                                scheme.TB[7 4; 8 2] * scheme.TA[8 6; 7 5])

    scheme.TA /= n^(1 / 4)
    scheme.TB /= n^(1 / 4)
    return n^(1 / 4)
end

function entanglement_filtering(A; ϵ = 1e-14, maxiter = 10, verbosity = 1)

    entanglement_alg = EntanglementFiltering(maxiter_TNR(maxiter))
    loop_alg = LoopOptimization(maxiter_TNR(maxiter))
    loop_tnr = LoopTNR(A; entanglement_alg = entanglement_alg, loop_alg = loop_alg)
    
    A_unfiltered = deepcopy(A)

    A = permute(A, (3,),(4,1,2))
    psi_A = AbstractTensorMap[A,A,A,A]
    PR_list, PL_list = find_projectors(psi_A, loop_tnr);
    @tensor A[-1; -2 -3 -4] := A[1; 2 -3 -4] * PR_list[1][2; -2] * PL_list[1][-1; 1];
    A = permute(A, (3,4), (2,1));

    @tensor A_joined[-2; -1] := A[2 2; -1 1] * A[3 3; 1 -2]

    loop_unfiltered = contract_tensors_symmetric(A_unfiltered)
    cut = 3
    error = Inf
    for cut = 1:3:61
        _, _, V = tsvd(A_joined; trunc = truncbelow(10.0^(-cut)))
        @tensor A_truncated[-1 -2; -3 -4] := conj(V[-3; 1]) * A[-1 -2; 1 2] * (V[-4; 2])
        loop_truncated = contract_tensors_symmetric(A_truncated)
        error = norm(loop_truncated - loop_unfiltered)/norm(loop_unfiltered)
        if error < ϵ
            if verbosity >= 2
                @info "Entanglement filtering converged:\n Error = $(error) for Schmidt-cut 1e-$(cut) and D = $(dim(A_truncated.dom[1]))"
            end
            return A_truncated, error
        end
    end
    if verbosity >= 1
        @warn "Entanglement filtering failed"
    end
    return A_truncated, Inf
end
