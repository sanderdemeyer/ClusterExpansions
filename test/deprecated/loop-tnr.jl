using LinearAlgebra
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
        crit = maxiter(10) & convcrit(1e-10, f)
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
        crit = maxiter(50) & convcrit(1e-12, f)
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

# function psiA(scheme::LoopTNR)
#     ψ = AbstractTensorMap[permute(scheme.TA, ((2,), (3, 4, 1))),
#                           permute(scheme.TB, ((1,), (2, 3, 4))),
#                           permute(scheme.TA, ((4,), (1, 2, 3))),
#                           permute(scheme.TB, ((3,), (4, 1, 2)))]
#     return ψ
# end

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

# function entanglement_filtering!(scheme::LoopTNR)
#     ψ = psiA(scheme)
#     PR_list, PL_list = find_projectors(ψ, scheme)

#     TA = copy(scheme.TA)
#     TB = copy(scheme.TB)

#     @tensor scheme.TA[-1 -2; -3 -4] := TA[1 2; 3 4] * PR_list[4][1; -1] *
#                                        PL_list[1][-2; 2] * PR_list[2][3; -3] *
#                                        PL_list[3][-4; 4]
#     @tensor scheme.TB[-1 -2; -3 -4] := TB[1 2; 3 4] * PL_list[2][-1; 1] *
#                                        PR_list[3][2; -2] * PL_list[4][-3; 3] *
#                                        PR_list[1][4; -4]

#     return scheme
# end

# function one_loop_projector(phi::Array, pos::Int)
#     L = id(space(phi[1])[1])
#     R = id(space(phi[end])[2]')
#     for i in 1:pos
#         L = QR_L(L, phi[i])
#     end
#     for i in length(phi):-1:(pos + 1)
#         R = QR_R(R, phi[i])
#     end
#     PR, PL = P_decomp(R, L)
#     return PR, PL
# end

# function SVD12(T::AbstractTensorMap{S,1,3}, trunc::TensorKit.TruncationScheme) where {S}
#     U, s, V, _ = tsvd(T, ((1, 4), (2, 3)); trunc=trunc)
#     @tensor S1[-1; -2 -3] := U[-1 -3; 1] * sqrt(s)[1; -2]
#     @tensor S2[-1; -2 -3] := sqrt(s)[-1; 1] * V[1; -2 -3]
#     return S1, S2
# end

# function psiB(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
#     psi = psiA(scheme)
#     psiB_new = []

#     for i in 1:4
#         s1, s2 = SVD12(psi[i], trunc)
#         phi = copy(psi)
#         popat!(phi, i)
#         insert!(phi, i, s1)
#         insert!(phi, i + 1, s2)

#         pr, pl = one_loop_projector(phi, i)

#         @tensor B1[-1; -2 -3] := s1[-1; 1 -3] * pr[1; -2]
#         @tensor B2[-1; -2 -3] := pl[-1; 1] * s2[1; -2 -3]
#         push!(psiB_new, B1)
#         push!(psiB_new, B2)
#     end
#     return psiB_new
# end

# function const_C(psiA)
#     @tensor tmp[-1 -2; -3 -4] := psiA[1][-1; -3 1 2] * adjoint(psiA[1])[-4 1 2; -2]
#     for i in 2:4
#         @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * psiA[i][1; -3 3 4] *
#                                      adjoint(psiA[i])[-4 3 4; 2]
#     end
#     return @tensor tmp[1 2; 1 2]
# end

# function tN(pos, psiB)
#     @tensor tmp[-1 -2; -3 -4] := psiB[mod(pos, 8) + 1][-1; -3 1] *
#                                  adjoint(psiB[mod(pos, 8) + 1])[-4 1; -2]
#     for i in (pos + 1):(pos + 6)
#         @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * psiB[mod(i, 8) + 1][1; -3 3] *
#                                      adjoint(psiB[mod(i, 8) + 1])[-4 3; 2]
#     end
#     return tmp
# end

# function tW(pos, psiA, psiB)
#     next_a = mod(ceil(Int, pos / 2), 4) + 1
#     next_b = mod(2 * ceil(Int, pos / 2) + 1, 8)

#     @tensor tmp[-1 -2; -3 -4] := psiA[next_a][-1; -3 1 2] * adjoint(psiB[next_b])[3 2; -2] *
#                                  adjoint(psiB[next_b + 1])[-4 1; 3]
#     for i in next_a:(next_a + 1)
#         @tensor tmp[-1 -2; -3 -4] := tmp[-1 -2; 1 2] * psiA[mod(i, 4) + 1][1; -3 3 4] *
#                                      adjoint(psiB[2 * (mod(i, 4) + 1) - 1])[5 4; 2] *
#                                      adjoint(psiB[2 * (mod(i, 4) + 1)])[-4 3; 5]
#     end

#     if pos % 2 == 0
#         @tensor W[-1; -2 -3] := tmp[1 -1; 2 3] * psiA[ceil(Int, pos / 2)][2; 1 -3 4] *
#                                 adjoint(psiB[pos - 1])[-2 4; 3]
#     else
#         @tensor W[-1; -2 -3] := tmp[1 3; 2 -2] * psiA[ceil(Int, pos / 2)][2; 1 4 -3] *
#                                 adjoint(psiB[pos + 1])[3 4; -1]
#     end

#     return W
# end

# function cost_func(pos, psiA, psiB)
#     C = const_C(psiA)
#     N = tN(pos, psiB)
#     TNT = norm(@tensor N[1 2; 3 4] * psiB[pos][3; 1 5] * adjoint(psiB[pos])[2 5; 4])
#     W = tW(pos, psiA, psiB)
#     WdT = norm(@tensor W[3; 1 2] * adjoint(psiB[pos])[3 2; 1])
#     dWT = norm(@tensor adjoint(W)[1 2; 3] * psiB[pos][1; 3 2])

#     return C + TNT - WdT - dWT
# end

# function opt_T(N, W)
#     function apply_f(x::TensorMap, ::Val{false})
#         @tensor b[-1; -2 -3] := N[1 -1; 2 -2] * x[2; 1 -3]
#         return b
#     end
#     function apply_f(b::TensorMap, ::Val{true})
#         @tensor x[-1; -2 -3] := adjoint(N)[-1 1; -2 2] * b[2; 1 -3]
#         return x
#     end

#     new_T, info = lssolve(apply_f, W, LSMR(; maxiter=500, verbosity=0))

#     return new_T
# end

# function loop_opt!(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
#     psi_A = psiA(scheme)
#     psi_B = psiB(scheme, trunc)

#     crit = true
#     cost = Inf
#     sweep = 0
#     while crit
#         for i in 1:8
#             N = tN(i, psi_B)
#             W = tW(i, psi_A, psi_B)
#             new_T = opt_T(N, W)
#             psi_B[i] = new_T
#         end
#         sweep += 1
#         # @show sweep
#         cost = cost_func(1, psi_A, psi_B)
#         # @show cost
#         crit = scheme.loop_alg.criterion(sweep, cost)
#     end
#     @tensor scheme.TB[-1 -2; -3 -4] := psi_B[5][4; -1 1] * psi_B[8][-2; 2 1] *
#                                        psi_B[1][2; -3 3] * psi_B[4][-4; 4 3]
#     @tensor scheme.TA[-1 -2; -3 -4] := psi_B[2][-1; 1 4] * psi_B[3][1; -2 2] *
#                                        psi_B[6][-3; 3 2] * psi_B[7][3; -4 4]
#     return scheme
# end

# function step!(scheme::LoopTNR, trunc=TensorKit.TruncationScheme)
#     entanglement_filtering!(scheme)
#     loop_opt!(scheme, trunc)
#     return scheme
# end

function finalize_TNR!(scheme::LoopTNR)
    n = norm(@tensor opt = true scheme.TA[1 2; 3 4] * scheme.TB[3 5; 1 6] *
                                scheme.TB[7 4; 8 2] * scheme.TA[8 6; 7 5])

    scheme.TA /= n^(1 / 4)
    scheme.TB /= n^(1 / 4)
    return n^(1 / 4)
end

# # show functions
# function Base.show(io::IO, scheme::LoopTNR)
#     println(io, "Loop TNR")
#     println(io, "  * TA: $(summary(scheme.TA))")
#     println(io, "  * TB: $(summary(scheme.TB))")
#     println(io, "  * Entanglement Filtering: $(summary(scheme.entanglement_alg.criterion))")
#     println(io, "  * Loop Optimization: $(summary(scheme.loop_alg.criterion))")
#     return nothing
# end