struct Cubic <: Lattice end

# QR decomposition
function QR_two_pepo_left(::Cubic, O1, O2, ind)
    pb = (1, ind)
    _, Rb = left_orth(permute(O1, (Tuple(setdiff(1:8, pb)), pb)))
    pt = (2, ind)
    _, Rt = left_orth(permute(O2, (Tuple(setdiff(1:8, pt)), pt)))
    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = left_orth(permute(M, ((3, 4), (1, 2))))
    return R
end

function QR_two_pepo_right(::Cubic, O1, O2, ind)
    pb = (1, ind)
    Rb, _ = right_orth(permute(O1, (pb, Tuple(setdiff(1:8, pb)))))
    pt = (2, ind)
    Rt, _ = right_orth(permute(O2, (pt, Tuple(setdiff(1:8, pt)))))
    @tensor M[-1 -2; -3 -4] := Rt[1 -1;-3] * Rb[1 -2;-4]
    R, _ = right_orth(permute(M, ((1, 2), (3, 4))))
    return R
end

function approximate_state(
        A::Tuple{AbstractTensorMap{E, S, 2, 6}, AbstractTensorMap{E, S, 2, 6}},
        trunc_alg::NoEnvTruncation; rot_invariance::Bool = false
    ) where {E, S <: ElementarySpace}
    proj = rot_invariance ? :takagi : :svd
    lattice = Cubic()
    PU,PD = find_P1P2(lattice,A,3,6,trunc_alg.trscheme; proj)
    # @info "$(norm(PU.data - (PD').data)) / $(norm(PU))"
    # @info "$(norm(PU.data - conj.((PD').data))) / $(norm(PU))"
    if rot_invariance
        PN = PE = copy(PU)
        PS = PW = copy(PD)
    else
        PN,PS = find_P1P2(lattice,A,4,7,trunc_alg.trscheme; proj)
        PE,PW = find_P1P2(lattice,A,5,8,trunc_alg.trscheme; proj)
    end
    @tensor opt = true Onew[-1 -2; -3 -4 -5 -6 -7 -8] := A[1][1 -2; 8 9 10 11 12 13] * A[2][-1 1; 2 3 4 5 6 7] * PU[2 8; -3] * PN[3 9; -4] * PE[4 10; -5] * PD[-6; 5 11] * PS[-7; 6 12] * PW[-8; 7 13]
    return Onew, nothing
end