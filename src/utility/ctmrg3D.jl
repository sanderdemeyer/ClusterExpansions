# Implementation of the 3C-CTMRG algorithm from Orus 2012 (Phys. Rev. B 85, 205117 – Published 10 May, 2012)

mutable struct CTMRG3D{T, S, TT <: AbstractTensorMap{T, S, 0, 6}, TA <: AbstractTensorMap{T, S, 1, 4}, TE <: AbstractTensorMap{T, S, 2, 2}, TC <: AbstractTensorMap{T, S, 0, 3}} <: CTMRG3D_alg
    O::TT
    A::Array{TA,1}
    E::Array{TE,1}
    C::Array{TC,1}
    trunc_bpeps::MatrixAlgebraKit.TruncationStrategy
    trunc_env::MatrixAlgebraKit.TruncationStrategy
    miniter::Int
    maxiter::Int
    tol::Float64
    verbosity::Int

    function CTMRG3D(O::TT; trunc_bpeps = notrunc(), trunc_env = notrunc(), Vp = space(O)[1]', Vbp = oneunit(Vp), Vv = oneunit(Vp), miniter::Int = 5, maxiter::Int = 150, tol::Float64 = 1e-8, verbosity::Int = 0) where {T, S, TT <: AbstractTensorMap{T, S, 0, 6}}
        A, E, C = CTMRG3D_init(O; Vp, Vbp, Vv)
        new{T, S, TT, typeof(A[1]), typeof(E[1]), typeof(C[1])}(O, A, E, C, trunc_bpeps, trunc_env, miniter, maxiter, tol, verbosity)
    end
end

function CTMRG3D_init(O::TensorMap{T,S,0,6}; Vp = space(O)[1]', Vbp = oneunit(Vp), Vv = oneunit(Vp)) where {T,S}
    S_type = scalartype(O)
    
    A = fill(ones(S_type, Vp, Vbp' ⊗ Vbp' ⊗ Vbp' ⊗ Vbp'), 6)
    E = fill(ones(S_type, Vv' ⊗ Vbp', Vv ⊗ Vbp), 12)
    C = ones(S_type, Vv, Vv' ⊗ Vv')
    C = fill(permute(C, ((),(1,2,3))), 8)
    return A, E, C
end

envspace(env::CTMRG3D) = domain(env.E[1])[1]

function rotl_90!(env::CTMRG3D, p_corners, p_edges, p_as)
    permute!(env.C, p_corners)
    permute!(env.E, p_edges)
    permute!(env.A, p_as)
end

function rot_ud_90(env::CTMRG3D)
    # negative rotation along the down -> up axis
    p_corners = [4, 1, 2, 3, 8, 5, 6, 7]
    p_edges = [4, 1, 2, 3, 8, 5, 6, 7, 12, 9, 10, 11]
    p_as = [1, 5, 2, 3, 4, 6]
    rotl_90!(env, p_corners, p_edges, p_as)
    env.O = permute(env.O, ((),(1,6,2,4,3,5)))
end

function rot_ns_90(env::CTMRG3D)
    # positive rotation along the south -> north axis
    p_corners = [5, 1, 4, 8, 6, 2, 3, 7]
    p_edges = [5, 4, 8, 12, 9, 1, 3, 11, 6, 2, 7, 10]
    p_as = [5, 2, 1, 4, 6, 3]
    rotl_90!(env, p_corners, p_edges, p_as)
    env.O = permute(env.O, ((),(6,2,1,3,5,4)))
end

function rot_we_90(env::CTMRG3D)
    # positive rotation along the east -> west axis
    p_corners = [4, 3, 7, 8, 1, 2, 6, 5]
    p_edges = [3, 7, 11, 8, 4, 2, 10, 12, 1, 6, 9, 5]
    p_as = [4, 1, 3, 6, 5, 2]
    rotl_90!(env, p_corners, p_edges, p_as)
    env.O = permute(env.O, ((),(5,1,3,2,4,6)))
end

function enlarged_corners(env::CTMRG3D)
    @tensor opt = true ec1[UP1 UP2 NORTH1 NORTH2 EAST] := 
                        env.C[4][UP1 NORTH1 1] * flip(env.E[3], [2 3 4])[1 NORTH2; EAST UP2]
    @tensor opt = true ec2[UP1 UP2 NORTH1 NORTH2 EAST] := 
                        env.C[1][UP1 NORTH1 1] * flip(env.E[1], [2 3 4])[1 NORTH2; EAST UP2]
    @tensor opt = true ec3[UP1 UP2 NORTH1 NORTH2 EAST] := 
                        env.C[5][UP1 NORTH1 1] * flip(env.E[9], [2 3 4])[1 NORTH2; EAST UP2]
    @tensor opt = true ec4[UP1 UP2 NORTH1 NORTH2 EAST] := 
                        env.C[8][UP1 NORTH1 1] * flip(env.E[11], [2 3 4])[1 NORTH2; EAST UP2]
    
    return [permute(ec, ((),(1,2,3,4,5))) for ec in [ec1 ec2 ec3 ec4]]
end

function enlarged_edges(env::CTMRG3D)
    @tensor opt = true ee1[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[4][UP1 NORTH1 DOWN1 1] * flip(env.A[1], [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    @tensor opt = true ee2[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[5][UP1 NORTH1 DOWN1 1] * flip(env.A[2], [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    @tensor opt = true ee3[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[12][UP1 NORTH1 DOWN1 1] * flip(env.A[6], [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    @tensor opt = true ee4[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[8][UP1 NORTH1 DOWN1 1] * flip(env.A[4], [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    return ee1, ee2, ee3, ee4
end

function enlarged_side(env::CTMRG3D)
    @tensor opt = true es[EAST; UP1 UP2 NORTH1 NORTH2 DOWN1 DOWN2 SOUTH1 SOUTH2] :=
                        env.A[5][1; UP1 NORTH1 DOWN1 SOUTH1] * flip(env.O, [1 2 3 4 5])[UP2 NORTH2 EAST; DOWN2 SOUTH2 1]
    return es
end

function get_projectors_corner(ecs, trunc)
    Uys = []
    Uzs = []
    for i = 1:4
        Uy, Sy, Vyᴴ = svd_trunc(permute(ecs[i], ((1,2),(3,4,5))); trunc)
        Uz, Sz, Vzᴴ = svd_trunc(permute(ecs[i], ((3,4),(1,2,5))); trunc)

        push!(Uys, Uy * sqrt(Sy))
        push!(Uzs, Uz * sqrt(Sz))
    end
    return Uys, Uzs
end

function get_projectors_edge(ees, trunc)
    Ws = []
    for i = 1:4
        W, S, M = svd_trunc(permute(ees[i], ((3,4),(1,2,5,6,7))); trunc)

        push!(Ws, W * sqrt(S))
    end
    return Ws
end


function partial_ctmrgstep(env::CTMRG3D)
    ecs = enlarged_corners(env)
    ees = enlarged_edges(env)
    es = enlarged_side(env)

    Uys, Uzs = get_projectors_corner(ecs, env.trunc_env)
    Ws = get_projectors_edge(ees, env.trunc_bpeps)

    renormalize_corners(env, ecs, Uys, Uzs)
    renormalize_edges(env, ees, Uys, Uzs, Ws)
    renormalize_side(env, es, Ws)

    normalize!(env)

    return
end

function ctmrg_4steps(env::CTMRG3D)
    for i = 1:4
        partial_ctmrgstep(env)
        rot_we_90(env)
    end
end

function ctmrgstep(env::CTMRG3D)
    for i = 1:4
        ctmrg_4steps(env)
        rot_ud_90(env)
    end
    rot_ns_90(env)
    ctmrg_4steps(env)
    rot_ns_90(env)
    rot_ns_90(env)
    ctmrg_4steps(env)
    rot_ns_90(env)
    _, S, _ = svd_trunc(permute(env.C[1], ((1,2),(3,))))
    return S
end

function renormalize_corners(env::CTMRG3D, ecs, Uys, Uzs)
    corners = [4, 1, 5, 8]
    for i = 1:4
        @tensor opt = true C′[u n e] :=
                            ecs[i][UP1 UP2 NORTH1 NORTH2 e] * 
                            conj(Uys[i][UP1 UP2; u]) * conj(Uzs[i][NORTH1 NORTH2; n])
        env.C[corners[i]] = permute(C′, ((),(1,2,3)))
    end
end

function renormalize_edges(env::CTMRG3D, ees, Uys, Uzs, Ws)
    edges = [4, 5, 12, 8]
    for i = 1:4
        @tensor opt = true E′[u n; d e] := 
                            ees[i][UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 e] * 
                            Uzs[mod1(i+1,4)][UP1 UP2; u] * Uys[i][DOWN1 DOWN2; d] *
                            conj(Ws[i][NORTH1 NORTH2; n])
        env.E[edges[i]] = flip(E′, 2)
    end
end

function renormalize_side(env::CTMRG3D, es, Ws)
    @tensor opt = true A′[e; u n d s] := 
                        es[e; UP1 UP2 NORTH1 NORTH2 DOWN1 DOWN2 SOUTH1 SOUTH2] * 
                        Ws[1][UP1 UP2; u] * Ws[2][NORTH1 NORTH2; n] * 
                        Ws[3][DOWN1 DOWN2; d] * Ws[4][SOUTH1 SOUTH2; s]
    env.A[5] = flip(A′, [2 3 4 5])
end

function normalize!(env::CTMRG3D)
    for i = 1:8
        env.C[i] /= norm(env.C[i])
    end
    for i = 1:12
        env.E[i] /= norm(env.E[i])
    end
    for i = 1:6
        env.A[i] /= norm(env.A[i])
    end
end

# function contract_onesite(env::CTMRG3D, M)
#     # add 3x3x3 contraction
#     return @tensor opt=true M[UP NORTH EAST; DOWN SOUTH WEST] * 
#                     env.A[1][UP; U9 U7 U4 U6] * env.A[6][DOWN; L4 L7 L9 L6] *
#                     env.A[5][WEST; A4 M3 B4 M8] * env.A[3][EAST; A6 M10 B6 M5] *
#                     env.A[2][NORTH; A2 M2 B2 M1] * env.A[4][SOUTH; A8 M11 B8 M12] * 
#                     env.E[8][A7 M8; B7 M11] * env.E[5][A1 M1; B1 M3] *
#                     env.E[6][A3 M5; B3 M2] * env.E[7][A9 M12; B9 M10] *
#                     env.E[4][U3 A4; U8 U6] * env.E[1][U2 A2; U1 U4] * 
#                     env.E[2][U10 A6; U5 U7] * env.E[3][U11 A8; U12 U9] *
#                     env.E[12][L8 B4; L3 L6] * env.E[9][L1 B2; L2 L4] *
#                     env.E[10][L5 B6; L10 L7] * env.E[11][L12 B8; L11 L9] * 
#                     env.C[8][B7 L8 L11] * env.C[5][B1 L1 L3] *
#                     env.C[6][B3 L5 L2] * env.C[7][B9 L12 L10] *
#                     env.C[4][U8 A7 U11] * env.C[1][U1 A1 U3] * 
#                     env.C[2][U5 A3 U2] * env.C[3][U12 A9 U10] 
# # env.A[1][UP; U9 U7 U4 U6] * env.A[6][DOWN; L4 L7 L9 L6] *
#     # env.A[5][WEST; A4 M3 B4 M8] * env.A[3][EAST; A6 M10 B6 M5] *
#     # env.A[2][NORTH; A2 M2 B2 M1] * env.A[4][SOUTH; A8 M11 B8 M12] * 
#     # env.E[8][A7 M8; B7 M11] * env.E[11][L11 L9; L12 B8] * 
#     # env.E[7][B9 M12; A9 M10] * env.E[3][U12 U9; U11 A8] *
#     # env.E[5][A1 M1; B1 M3] * env.E[1][U2 A2; U1 U4] * 
#     # env.E[6][B3 M2; A3 M5] * env.E[9][L1 B2; L2 L4] * 
#     # env.E[4][U3 U6; U8 A4] * env.E[2][U5 U7; U10 A6] * 
#     # env.E[10][L5 B6; L10 L7] * env.E[12][L3 L6; L8 B4] *
#     # env.C[4][U8 A7 U11] * env.C[1][U1 A1 U3] * 
#     # env.C[2][U5 A3 U2] * env.C[3][U12 A9 U10] * 
#     # env.C[8][B7 L8 L11] * env.C[5][B1 L1 L3] *
#     # env.C[6][L5 L2 B3] * env.C[7][L12 L10 B9]
# end

# Memory-minimal contraction order for the closed cube network of snippet.txt.
#
# At χ = 8, D = 5 (corner legs χ, all other legs D):
#
#                     flops      largest intermediate   live intermediates (peak)
#   opt = true        1.52e12    119.2 GiB (χ⁴D⁹)       238.4 GiB
#   this order        2.27e12     23.8 GiB (χ⁴D⁸)        39.1 GiB
#
# parentheses enforce contraction order, so no `opt` is needed.

function contract_onesite(env::CTMRG3D, M; allocator = ManualAllocator())
    return @tensor allocator = allocator (
        env.C[1][U1 A1 U3] *
        (
            env.E[1][U2 A2; U1 U4] *
            (
                env.C[2][U5 A3 U2] *
                (
                    (
                        (env.E[5][A1 M1; B1 M3] * env.C[5][B1 L1 L3]) *
                        (
                            env.A[2][NORTH; A2 M2 B2 M1] *
                            (env.E[6][A3 M5; B3 M2] * (env.E[9][L1 B2; L2 L4] * env.C[6][B3 L5 L2]))
                        )
                    ) *
                    (
                        (
                            (env.E[2][U10 A6; U5 U7] * env.C[3][U12 A9 U10]) *
                            (
                                env.A[1][UP; U9 U7 U4 U6] *
                                (env.E[4][U3 A4; U8 U6] * (env.E[3][U11 A8; U12 U9] * env.C[4][U8 A7 U11]))
                            )
                        ) *
                        (
                            (
                                env.A[5][WEST; A4 M3 B4 M8] *
                                (env.E[8][A7 M8; B7 M11] * (env.E[12][L8 B4; L3 L6] * env.C[8][B7 L8 L11]))
                            ) *
                            (
                                M[UP NORTH EAST; DOWN SOUTH WEST] *
                                (
                                    env.A[3][EAST; A6 M10 B6 M5] *
                                    (
                                        env.A[4][SOUTH; A8 M11 B8 M12] *
                                        (
                                            env.E[7][A9 M12; B9 M10] *
                                            (
                                                env.A[6][DOWN; L4 L7 L9 L6] *
                                                (env.E[11][L12 B8; L11 L9] * (env.E[10][L5 B6; L10 L7] * env.C[7][B9 L12 L10]))
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
end

function transfer_matrix(env::CTMRG3D)
    @tensor opt = true TM[A1 A3 A9 A7; B1 B3 B9 B7] := 
                            flip(env.E[1], 1)[A7 1; B7 4] * flip(env.E[9], [1 2 4])[A1 2; B1 1] * 
                            flip(env.E[11], 1)[A3 3; B3 2] * flip(env.E[3], [1 2 4])[A9 4; B9 3]
    return TM
end

function calculate_universal_3D_spectrum(env::CTMRG3D)
    @tensor opt=true mat_e[M1a M4a; M1b M4b] := env.C[8][M4b B4 B3] * flip(env.C[5], [1 2 3])[M1b B1 B4] * env.C[6][M2 B2 B1] * flip(env.C[7], [1 2 3])[M3 B3 B2] * 
                                flip(env.C[4], [1 2 3])[A4 M4a A3] * env.C[1][A1 M1a A4] * flip(env.C[2], [1 2 3])[A2 M2 A1] * env.C[3][A3 M3 A2]

    @tensor opt=true mat_c[M1a; M1b] := mat_e[M1a M4; M1b M4]
    _, S_e, V = svd_full(permute(mat_e, ((1,3),(2,4))))
    S_e /= tr(S_e)
    _, S_c, V = svd_full(mat_c)
    S_c /= tr(S_c)

    EE_e = -tr(S_e * log(S_e))
    EE_c = -tr(S_c * log(S_c))
    return EE_e - EE_c
end