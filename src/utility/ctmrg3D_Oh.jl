mutable struct CTMRG3D_Oh{T, S, TT <: AbstractTensorMap{T, S, 0, 6}, TA <: AbstractTensorMap{T, S, 1, 4}, TE <: AbstractTensorMap{T, S, 2, 2}, TC <: AbstractTensorMap{T, S, 0, 3}} <: CTMRG3D_alg
    O::TT
    A::TA
    E::TE
    C::TC
    trunc_bpeps::TruncationScheme
    trunc_env::TruncationScheme
    maxiter::Int
    tol::Float64
    verbosity::Int

    function CTMRG3D_Oh(O::TT; trunc_bpeps = notrunc(), trunc_env = notrunc(), Vp = space(O)[1]', Vbp = oneunit(Vp), Vv = oneunit(Vp), maxiter::Int = 150, tol::Float64 = 1e-8, verbosity::Int = 0) where {T, S, TT <: AbstractTensorMap{T, S, 0, 6}}
        A, E, C = CTMRG3D_Oh_init(O; Vbp, Vp, Vv)
        new{T, S, TT, typeof(A), typeof(E), typeof(C)}(O, A, E, C, trunc_bpeps, trunc_env, maxiter, tol, verbosity)
    end
end

function CTMRG3D_Oh_init(O::TensorMap{T,S,0,6}; Vp = space(O)[1]', Vbp = oneunit(Vp), Vv = oneunit(Vp)) where {T,S}
    @assert norm(O - permute(O, ((),(1,6,2,4,3,5)))) < 1e-8
    @assert norm(O - permute(O, ((),(5,1,3,2,4,6)))) < 1e-8
    @assert norm(O - permute(O, ((),(6,2,1,3,5,4)))) < 1e-8
    S_type = scalartype(O)
    
    A = ones(S_type, Vp, Vbp' ⊗ Vbp' ⊗ Vbp' ⊗ Vbp')
    E = ones(S_type, Vv' ⊗ Vbp', Vv ⊗ Vbp)
    C = ones(S_type, Vv, Vv' ⊗ Vv')
    C = permute(C, ((),(1,2,3)))
    return A, E, C
end

envspace(env::CTMRG3D_Oh) = domain(env.E)[1]

function enlarged_corner(env::CTMRG3D_Oh)
    @tensor opt = true ec[UP1 UP2 NORTH1 NORTH2 EAST] := 
                        env.C[UP1 NORTH1 1] * flip(env.E, [2 3 4])[1 NORTH2; EAST UP2]
    return permute(ec, ((),(1,2,3,4,5)))
end

function enlarged_edge(env::CTMRG3D_Oh)
    @tensor opt = true ee[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[UP1 NORTH1 DOWN1 1] * flip(env.A, [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    return ee
end

function enlarged_side(env::CTMRG3D_Oh)
    @tensor opt = true es[EAST; UP1 UP2 NORTH1 NORTH2 DOWN1 DOWN2 SOUTH1 SOUTH2] :=
                        env.A[1; UP1 NORTH1 DOWN1 SOUTH1] * flip(env.O, [1 2 3 4 5])[UP2 NORTH2 EAST; DOWN2 SOUTH2 1]
    return es
end

function ctmrgstep(env::CTMRG3D_Oh)
    ec = enlarged_corner(env)
    ee = enlarged_edge(env)
    es = enlarged_side(env)

    Uy, S₁, Vyᴴ, = tsvd(permute(ec, ((1,2),(3,4,5))); trunc = env.trunc_env)
    Uz, S₂, Vzᴴ, = tsvd(permute(ec, ((3,4),(1,2,5))); trunc = env.trunc_env)
    Wy, S₄, Myᴴ, = tsvd(permute(ee, ((3,4),(1,2,5,6,7))); trunc = env.trunc_bpeps)

    Uy, Vyᴴ = Uy * sqrt(S₁), sqrt(S₁) * Vyᴴ
    Uz, Vzᴴ = Uz * sqrt(S₂), sqrt(S₂) * Vzᴴ
    Wy, Myᴴ = Wy * sqrt(S₄), sqrt(S₄) * Myᴴ

    Wy = flip(Wy, [3])
    Myᴴ = flip(Myᴴ, [1])

    renormalize_corner(env, ec, Uy, Uz)
    renormalize_edge(env, ee, Uz, Wy)
    renormalize_side(env, es, Wy)

    normalize!(env)

    return S₁
end

function renormalize_corner(env::CTMRG3D_Oh, ec, Uy, Uz)
    @tensor opt = true C′[u n e] :=
                        ec[UP1 UP2 NORTH1 NORTH2 e] * 
                        conj(Uy[UP1 UP2; u]) * conj(Uz[NORTH1 NORTH2; n])
    env.C = permute(C′, ((),(1,2,3)))
end

function renormalize_edge(env::CTMRG3D_Oh, ee, U, W)
    @tensor opt = true E′[u n; d e] := 
                        ee[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 e] * 
                        U[UP1 UP2; u] * U[DOWN1 DOWN2; d] *
                        conj(W[NORTH1 NORTH2; n])
    env.E = E′
end

function renormalize_side(env::CTMRG3D_Oh, es, W)
    @tensor opt = true A′[e; u n d s] := 
                        es[e; UP1 UP2 NORTH1 NORTH2 DOWN1 DOWN2 SOUTH1 SOUTH2] * 
                        W[UP1 UP2; u] * W[NORTH1 NORTH2; n] * 
                        W[DOWN1 DOWN2; d] * W[SOUTH1 SOUTH2; s]
    env.A = A′
end

function normalize!(env::CTMRG3D_Oh)
    env.C /= norm(env.C)
    env.E /= norm(env.E)
    env.A /= norm(env.A)
end

function check_rotinv(env::CTMRG3D_Oh)
    @assert norm(env.C - permute(env.C, ((),(2,3,1)))) < 1e-4
    @assert norm(env.C - permute(env.C, ((),(3,2,1)))) < 1e-4
    @assert norm(env.E - permute(env.E, ((3,4),(1,2)))) < 1e-4
    @assert norm(env.A - permute(env.A, ((1,),(4,5,2,3)))) < 1e-4
end

function contract_onesite(env::CTMRG3D_Oh, M)
    # add 3x3x3 contraction
    return @tensor opt = true M[UP NORTH EAST; DOWN SOUTH WEST] * 
                                env.A[UP; U9 U7 U4 U6] * env.A[DOWN; L4 L7 L9 L6] *
                                env.A[WEST; A4 M3 B4 M8] * env.A[EAST; A6 M10 B6 M5] *
                                env.A[NORTH; A2 M2 B2 M1] * env.A[SOUTH; A8 M11 B8 M12] * 
                                env.E[A7 M8; B7 M11] * env.E[L11 L9; L12 B8] * 
                                env.E[B9 M12; A9 M10] * env.E[U12 U9; U11 A8] *
                                env.E[A1 M1; B1 M3] * env.E[U2 A2; U1 U4] * 
                                env.E[B3 M2; A3 M5] * env.E[L1 B2; L2 L4] * 
                                env.E[U3 U6; U8 A4] * env.E[U5 U7; U10 A6] * 
                                env.E[L5 B6; L10 L7] * env.E[L3 L6; L8 B4] *
                                env.C[U8 U11 A7] * env.C[U1 U3 A1] * 
                                env.C[U2 U5 A3] * env.C[U12 U10 A9] * 
                                env.C[L8 L11 B7] * env.C[L1 L3 B1] *
                                env.C[L2 L5 B3] * env.C[L12 L10 B9]
end

function transfer_matrix(env::CTMRG3D_Oh)
    @tensor opt = true TM[A1 A3 A9 A7; B1 B3 B9 B7] := 
                            flip(env.E, 1)[A7 1; B7 4] * flip(env.E, [1 2 4])[A1 2; B1 1] * 
                            flip(env.E, 1)[A3 3; B3 2] * flip(env.E, [1 2 4])[A9 4; B9 3]
    return TM
end
