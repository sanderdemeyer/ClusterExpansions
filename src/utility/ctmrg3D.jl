# Implementation of the 3C-CTMRG algorithm from Orus 2012 (Phys. Rev. B 85, 205117 – Published 10 May, 2012)
#
# ============================================================================================
#  Conventions
# ============================================================================================
#
# The environment of a single site O is a 3x3x3 cube: O in the centre, surrounded by
# 6 face tensors A, 12 edge tensors E and 8 corner tensors C.
#
# Directions: EAST = +x, NORTH = +y, UP = +z (right-handed), WEST/SOUTH/DOWN the opposites.
#
#   O (0,6):  O[; UP NORTH EAST DOWN SOUTH WEST]
#
# Positions (letters give the direction(s) of the tensor as seen from the centre):
#
#   A1 = U    A2 = N    A3 = E    A4 = S    A5 = W    A6 = D
#
#   upper ring:     E1 = UN   E2 = UE   E3 = US   E4 = UW
#   vertical edges: E5 = NW   E6 = NE   E7 = SE   E8 = SW
#   lower ring:     E9 = DN   E10 = DE  E11 = DS  E12 = DW
#
#   upper corners:  C1 = UNW  C2 = UNE  C3 = USE  C4 = USW
#   lower corners:  C5 = DNW  C6 = DNE  C7 = DSE  C8 = DSW
#
#   Seen from above (N on top), both layers look like
#
#       C1 ─── E1 ─── C2        E5 ─── A2 ─── E6                C5 ─── E9 ─── C6
#       │              │        │              │                │              │
#       E4     A1     E2        A5      O     A3               E12     A6     E10       
#       │              │        │              │                │              │  
#       C4 ─── E3 ─── C3        E8 ─── A4 ──  E7                C8 ─── E11 ── C7
#           upper layer           middle layer                 lower layer
#
# Leg order. Every environment tensor lists its virtual legs counterclockwise as seen from
# outside the cube. The first leg is fixed by the rules below, which are chosen relative to
# the W-E axis, i.e. the direction in which `partial_ctmrgstep` absorbs (it grows the WEST side).
#
#   A (1,4):  A[O; l1 l2 l3 l4], O = the leg contracting with the central tensor.
#             A1, A2, A4, A6 (faces parallel to x): l2 = the edge on the west side,
#                                                   l4 = the edge on the east side.
#             A3, A5 (faces on the x axis):         l1 = the edge on the upper side.
#   E (2,2):  E[c1 f1; c2 f2], c = corner legs, f = face legs.
#             E1, E3, E9, E11 (edges parallel to x): c1 = west corner, c2 = east corner.
#             all other edges (in the W or E plane): f1 = A5 (west) resp. A3 (east).
#   C (0,3):  C[; e1 e2 e3], e1 and e2 are the edges in the corner's own (W or E) plane,
#             e3 is the edge along x.
#
# Explicitly:
#
#   A1 = [O; E1 E4 E3 E2]     E1 = [C1 A1; C2 A2]     E7  = [C7 A3; C3 A4]     C1 = [E5 E4 E1]
#   A2 = [O; E9 E5 E1 E6]     E2 = [C3 A3; C2 A1]     E8  = [C4 A5; C8 A4]     C2 = [E2 E6 E1]
#   A3 = [O; E2 E7 E10 E6]    E3 = [C4 A4; C3 A1]     E9  = [C5 A2; C6 A6]     C3 = [E7 E2 E3]
#   A4 = [O; E3 E8 E11 E7]    E4 = [C1 A5; C4 A1]     E10 = [C6 A3; C7 A6]     C4 = [E4 E8 E3]
#   A5 = [O; E4 E5 E12 E8]    E5 = [C5 A5; C1 A2]     E11 = [C8 A6; C7 A4]     C5 = [E12 E5 E9]
#   A6 = [O; E11 E12 E9 E10]  E6 = [C2 A3; C6 A2]     E12 = [C8 A5; C5 A6]     C6 = [E6 E10 E9]
#                                                                              C7 = [E10 E7 E11]
#                                                                              C8 = [E8 E12 E11]
#
# This convention is covariant under the rotation around the W-E axis (S -> U -> N -> D):
# the tensors on the west side (C4, C1, C5, C8), (E3, E1, E9, E11), (E4, E5, E12, E8),
# (A1, A2, A6, A4) are mapped onto each other with identical leg order, which is why
# `enlarged_corners`, `enlarged_edges`, `renormalize_corners` and `renormalize_edges` can use
# the same index pattern for all four of them.
#
# Spaces (see `CTMRG3D_init`): 
# O ∈ one ← Vp ⊗ Vp ⊗ Vp ⊗ Vp ⊗ Vp ⊗ Vp. 
# A ∈ Vp ← Vbp' ⊗ Vbp' ⊗ Vbp' ⊗ Vbp', 
# E ∈ Vv' ⊗ Vbp' ← Vv ⊗ Vbp,
# C ∈ one ← Vv' ⊗ Vv' ⊗ Vv'. 
# Corner legs may contract with either c1 or c2 and A legs with either
# f1 or f2, so cyclic permutations of legs (see `cycle_legs`) never break the arrow structure.
#
# Rotations. A full CTMRG step (`ctmrgstep`) absorbs from the west, rotating the whole network
# in between with `rot_ud_90`, `rot_ns_90` and `rot_we_90` such that every side is grown.
# A rotation moves each tensor to a new position (permutation p) and, because no leg convention
# can be covariant under all cube rotations (e.g. the 3-fold rotation around a corner diagonal
# cyclically permutes that corner's legs), it also cyclically shifts the legs of the tensor by
# s such that it obeys the convention of its new position:
#
#   new.X[i] = cycle_legs(old.X[p[i]], s[i])      (leg k of the new tensor = leg k + s[i] of the old one)
#
# The shifts were derived from the geometry above. A rotation is an exact symmetry of the
# closed network, so `contract_onesite(env, M)` is invariant under any rotation of `env`
# together with the same permutation of the legs of M.
# ============================================================================================

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

# Cyclically shift the virtual legs of an environment tensor: leg k of the result is leg k + s
# of the input. Since all tensors are ordered counterclockwise as seen from outside the cube,
# a proper rotation can only change the starting leg, so this is all that is needed.
cycle_legs(C::AbstractTensorMap{T,S,0,3}, s) where {T,S} =
    iszero(mod(s, 3)) ? C : permute(C, ((), Tuple(mod1(k + s, 3) for k in 1:3)))
cycle_legs(A::AbstractTensorMap{T,S,1,4}, s) where {T,S} =
    iszero(mod(s, 4)) ? A : permute(A, ((1,), Tuple(1 + mod1(k + s, 4) for k in 1:4)))
function cycle_legs(E::AbstractTensorMap{T,S,2,2}, s) where {T,S}
    iszero(mod(s, 4)) && return E
    @assert mod(s, 4) == 2 "edges can only be shifted by 2 (corner legs are at positions 1 and 3)"
    return permute(E, ((3, 4), (1, 2)))
end

# Move the tensor at position p[i] to position i and shift its legs by s[i] such that it
# satisfies the leg convention of its new position (see top of this file).
function rotl_90!(env::CTMRG3D, p_corners, p_edges, p_as, s_corners, s_edges, s_as)
    env.C[:] = [cycle_legs(env.C[p], s) for (p, s) in zip(p_corners, s_corners)]
    env.E[:] = [cycle_legs(env.E[p], s) for (p, s) in zip(p_edges, s_edges)]
    env.A[:] = [cycle_legs(env.A[p], s) for (p, s) in zip(p_as, s_as)]
end

function rot_ud_90(env::CTMRG3D)
    # negative (clockwise seen from above) rotation along the down -> up axis: W -> N -> E -> S -> W
    p_corners = [4, 1, 2, 3, 8, 5, 6, 7]
    p_edges = [4, 1, 2, 3, 8, 5, 6, 7, 12, 9, 10, 11]
    p_as = [1, 5, 2, 3, 4, 6]
    s_corners = [1, 2, 1, 2, 2, 1, 2, 1]
    s_edges = [2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2]
    s_as = [1, 2, 2, 0, 0, 3]
    rotl_90!(env, p_corners, p_edges, p_as, s_corners, s_edges, s_as)
    env.O = permute(env.O, ((),(1,6,2,4,3,5)))
end

function rot_ns_90(env::CTMRG3D)
    # positive rotation along the south -> north axis: U -> E -> D -> W -> U
    p_corners = [5, 1, 4, 8, 6, 2, 3, 7]
    p_edges = [5, 4, 8, 12, 9, 1, 3, 11, 6, 2, 7, 10]
    p_as = [5, 2, 1, 4, 6, 3]
    s_corners = [2, 1, 2, 1, 1, 2, 1, 2]
    s_edges = [0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 0, 2]
    s_as = [1, 3, 1, 1, 1, 1]
    rotl_90!(env, p_corners, p_edges, p_as, s_corners, s_edges, s_as)
    env.O = permute(env.O, ((),(6,2,1,3,5,4)))
end

function rot_we_90(env::CTMRG3D)
    # positive rotation along the east -> west axis: S -> U -> N -> D -> S.
    # The leg convention is covariant under this rotation, except for the faces on the axis.
    p_corners = [4, 3, 7, 8, 1, 2, 6, 5]
    p_edges = [3, 7, 11, 8, 4, 2, 10, 12, 1, 6, 9, 5]
    p_as = [4, 1, 3, 6, 5, 2]
    s_corners = zeros(Int, 8)
    s_edges = zeros(Int, 12)
    s_as = [0, 0, 1, 0, 3, 0]
    rotl_90!(env, p_corners, p_edges, p_as, s_corners, s_edges, s_as)
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

# The projectors are isometries U: one side of a new bond gets conj(U), the other side U, so
# the bond carries the projector conj(U) * transpose(U), which is exact without truncation.
# (Using U * sqrt(S) instead would insert conj(U) * S * transpose(U), biasing every absorption.)
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
        env.C[1][A1 U3 U1] *
        (
            env.E[1][U1 U4; U2 A2] *
            (
                env.C[2][U5 A3 U2] *
                (
                    (
                        (env.E[5][B1 M3; A1 M1] * env.C[5][L3 B1 L1]) *
                        (
                            env.A[2][NORTH; B2 M1 A2 M2] *
                            (env.E[6][A3 M5; B3 M2] * (env.E[9][L1 B2; L2 L4] * env.C[6][B3 L5 L2]))
                        )
                    ) *
                    (
                        (
                            (env.E[2][U10 A6; U5 U7] * env.C[3][A9 U10 U12]) *
                            (
                                env.A[1][UP; U4 U6 U9 U7] *
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
                                            env.E[7][B9 M10; A9 M12] *
                                            (
                                                env.A[6][DOWN; L9 L6 L4 L7] *
                                                (env.E[11][L11 L9; L12 B8] * (env.E[10][L5 B6; L10 L7] * env.C[7][L10 B9 L12]))
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
                            flip(env.E[1], 1)[A7 1; B7 4] * flip(env.E[3], [1 2 4])[A1 2; B1 1] * 
                            flip(env.E[11], 1)[A3 3; B3 2] * flip(env.E[9], [1 2 4])[A9 4; B9 3]
    return TM
end

function calculate_universal_3D_spectrum(env::CTMRG3D)
    @tensor opt=true mat_e[M1a M4a; M1b M4b] := env.C[8][M4b B4 B3] * flip(env.C[5], [1 2 3])[B4 M1b B1] * env.C[6][M2 B2 B1] * flip(env.C[7], [1 2 3])[B2 M3 B3] * 
                                flip(env.C[4], [1 2 3])[A4 M4a A3] * env.C[1][M1a A4 A1] * flip(env.C[2], [1 2 3])[A2 M2 A1] * env.C[3][M3 A2 A3]

    @tensor opt=true mat_c[M1a; M1b] := mat_e[M1a M4; M1b M4]
    _, S_e, V = svd_full(permute(mat_e, ((1,3),(2,4))))
    S_e /= tr(S_e)
    _, S_c, V = svd_full(mat_c)
    S_c /= tr(S_c)

    EE_e = -tr(S_e * log(S_e))
    EE_c = -tr(S_c * log(S_c))
    return EE_e - EE_c
end