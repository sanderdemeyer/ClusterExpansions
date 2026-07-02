mutable struct CTMRG3D{T, S, TT <: AbstractTensorMap{T, S, 0, 6}, TA <: AbstractTensorMap{T, S, 1, 4}, TE <: AbstractTensorMap{T, S, 2, 2}, TC <: AbstractTensorMap{T, S, 0, 3}} <: CTMRG3D_alg
    O::TT
    A::Array{TA,1}
    E::Array{TE,1}
    C::Array{TC,1}
    trunc_bpeps::TruncationScheme
    trunc_env::TruncationScheme
    maxiter::Int
    tol::Float64
    verbosity::Int

    function CTMRG3D(O::TT; trunc_bpeps = notrunc(), trunc_env = notrunc(), Vp = space(O)[1]', Vbp = oneunit(Vp), Vv = oneunit(Vp), maxiter::Int = 150, tol::Float64 = 1e-8, verbosity::Int = 0) where {T, S, TT <: AbstractTensorMap{T, S, 0, 6}}
        A, E, C = CTMRG3D_init(O; Vp, Vbp, Vv)
        new{T, S, TT, typeof(A[1]), typeof(E[1]), typeof(C[1])}(O, A, E, C, trunc_bpeps, trunc_env, maxiter, tol, verbosity)
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
    p_corners = [4, 1, 2, 3, 8, 5, 6, 7]
    p_edges = [4, 1, 2, 3, 8, 5, 6, 7, 12, 9, 10, 11]
    p_as = [1, 5, 2, 3, 4, 6]
    rotl_90!(env, p_corners, p_edges, p_as)
end

function rot_ns_90(env::CTMRG3D)
    p_corners = [5, 1, 4, 8, 6, 2, 3, 7]
    p_edges = [5, 4, 8, 12, 9, 1, 3, 11, 6, 2, 7, 10]
    p_as = [5, 2, 1, 4, 6, 3]
    rotl_90!(env, p_corners, p_edges, p_as)
end

function rot_we_90(env::CTMRG3D)
    p_corners = [4, 3, 7, 8, 1, 2, 6, 5]
    p_edges = [3, 7, 11, 8, 4, 2, 10, 12, 1, 6, 9, 5]
    p_as = [4, 1, 3, 6, 5, 2]
    rotl_90!(env, p_corners, p_edges, p_as)
end

function enlarged_corner(env::CTMRG3D)
    @tensor opt = true ec[UP1 UP2 NORTH1 NORTH2 EAST] := 
                        env.C[4][UP1 NORTH1 1] * flip(env.E[3], [2 3 4])[1 NORTH2; EAST UP2]
    return permute(ec, ((),(1,2,3,4,5)))
end

function enlarged_edge(env::CTMRG3D)
    @tensor opt = true ee1[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[4][UP1 NORTH1 DOWN1 1] * flip(env.A[1], [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    @tensor opt = true ee3[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[8][UP1 NORTH1 DOWN1 1] * flip(env.A[4], [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    return ee1, ee3
end

function enlarged_side(env::CTMRG3D)
    @tensor opt = true es[EAST; UP1 UP2 NORTH1 NORTH2 DOWN1 DOWN2 SOUTH1 SOUTH2] :=
                        env.A[5][1; UP1 NORTH1 DOWN1 SOUTH1] * flip(env.O, [1 2 3 4 5])[UP2 NORTH2 EAST; DOWN2 SOUTH2 1]
    return es
end

function partial_ctmrgstep(env::CTMRG3D)
    ec = enlarged_corner(env)
    ee1, ee3 = enlarged_edge(env)
    es = enlarged_side(env)


    Uy, S₁, Vyᴴ, = tsvd(permute(ec, ((1,2),(3,4,5))); trunc = env.trunc_env)
    Uz, S₂, Vzᴴ, = tsvd(permute(ec, ((3,4),(1,2,5))); trunc = env.trunc_env)
    Wz, S₃, Mzᴴ, = tsvd(permute(ee1, ((3,4),(1,2,5,6,7))); trunc = env.trunc_bpeps)
    Wy, S₄, Myᴴ, = tsvd(permute(ee3, ((3,4),(1,2,5,6,7))); trunc = env.trunc_bpeps)

    Uy, Vyᴴ = Uy * sqrt(S₁), sqrt(S₁) * Vyᴴ
    Uz, Vzᴴ = Uz * sqrt(S₂), sqrt(S₂) * Vzᴴ
    Wz, Mzᴴ = Wz * sqrt(S₃), sqrt(S₃) * Mzᴴ
    Wy, Myᴴ = Wy * sqrt(S₄), sqrt(S₄) * Myᴴ

    Wy = flip(Wy, [3])
    Wz = flip(Wz, [3])
    Myᴴ = flip(Myᴴ, [1])
    Mzᴴ = flip(Mzᴴ, [1])

    env.C[4] = renormalize_corner(ec, Uy, Uz)
    env.E[8] = renormalize_edge(ee3, Uz, Wy)
    env.E[4] = renormalize_edge(ee1, Uy, Wz)
    env.A[5] = renormalize_side(es, Wy, Wz)

    normalize!(env)

    return S₁
end

function ctmrg_4steps(env::CTMRG3D)
    local S
    for i = 1:4
        S = partial_ctmrgstep(env)
        rot_we_90(env)
    end
    return S
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
    S = ctmrg_4steps(env)
    return S
end

function renormalize_corner(ec, Uy, Uz)
    @tensor opt = true C′[u n e] :=
                        ec[UP1 UP2 NORTH1 NORTH2 e] * 
                        conj(Uy[UP1 UP2; u]) * conj(Uz[NORTH1 NORTH2; n])
    return permute(C′, ((),(1,2,3)))
end

function renormalize_edge(ee, U, W)
    @tensor opt = true E′[u n; d e] := 
                        ee[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 e] * 
                        U[UP1 UP2; u] * U[DOWN1 DOWN2; d] *
                        conj(W[NORTH1 NORTH2; n])
    return E′
end

function renormalize_side(es, Wy, Wz)
    @tensor opt = true A′[e; u n d s] := 
                        es[e; UP1 UP2 NORTH1 NORTH2 DOWN1 DOWN2 SOUTH1 SOUTH2] * 
                        Wz[UP1 UP2; u] * Wy[NORTH1 NORTH2; n] * 
                        Wz[DOWN1 DOWN2; d] * Wy[SOUTH1 SOUTH2; s]
    return A′
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

function contract_onesite(env::CTMRG3D, M)
    # add 3x3x3 contraction
    return @tensor opt = true M[UP NORTH EAST; DOWN SOUTH WEST] * 
                        env.A[1][UP; U9 U7 U4 U6] * env.A[6][DOWN; L4 L7 L9 L6] *
                        env.A[5][WEST; A4 M3 B4 M8] * env.A[3][EAST; A6 M10 B6 M5] *
                        env.A[2][NORTH; A2 M2 B2 M1] * env.A[4][SOUTH; A8 M11 B8 M12] * 
                        env.E[8][A7 M8; B7 M11] * env.E[11][L11 L9; L12 B8] * 
                        env.E[7][B9 M12; A9 M10] * env.E[3][U12 U9; U11 A8] *
                        env.E[5][A1 M1; B1 M3] * env.E[1][U2 A2; U1 U4] * 
                        env.E[6][B3 M2; A3 M5] * env.E[9][L1 B2; L2 L4] * 
                        env.E[4][U3 U6; U8 A4] * env.E[2][U5 U7; U10 A6] * 
                        env.E[10][L5 B6; L10 L7] * env.E[12][L3 L6; L8 B4] *
                        env.C[4][U8 A7 U11] * env.C[1][U1 A1 U3] * 
                        env.C[2][U5 A3 U2] * env.C[3][U12 A9 U10] * 
                        env.C[8][B7 L8 L11] * env.C[5][B1 L1 L3] *
                        env.C[6][L5 L2 B3] * env.C[7][L12 L10 B9]
end

function transfer_matrix(env::CTMRG3D)
    @tensor opt = true TM[A1 A3 A9 A7; B1 B3 B9 B7] := 
                            flip(env.E[1], 1)[A7 1; B7 4] * flip(env.E[9], [1 2 4])[A1 2; B1 1] * 
                            flip(env.E[11], 1)[A3 3; B3 2] * flip(env.E[3], [1 2 4])[A9 4; B9 3]
    return TM
end
