# QR decomposition
function QR_two_pepo_left(lattice::Triangular, O1,O2,ind)
    pb = (1,ind)
    _, Rb= leftorth(O1,(Tuple(setdiff(1:8, pb)),pb))
    pt = (2,ind)
    _, Rt= leftorth(O2,(Tuple(setdiff(1:8, pt)),pt))
    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = leftorth(M, (3,4),(1,2))
    return R
end

function QR_two_pepo_right(lattice::Triangular, O1,O2,ind)
    pb = (1,ind)
    Rb, _ = rightorth(O1, (pb, Tuple(setdiff(1:8,pb))))
    pt = (2,ind)
    Rt, _ = rightorth(O2, (pt, Tuple(setdiff(1:8,pt))))
    @tensor M[-1 -2; -3 -4] := Rt[1 -1;-3] * Rb[1 -2;-4]
    R, _ = rightorth(M, (1,2),(3,4))
    return R
end

function QR_two_pepo(lattice::Triangular, O1,O2,ind;side=:left)
    if side==:left
        return QR_two_pepo_left(lattice,O1,O2,ind)
    elseif side==:right
        return QR_two_pepo_right(lattice,O1,O2,ind)
    else
        @error "side shoulde be :left or :right"
    end
end

function R1R2(lattice::Triangular, A1, A2, ind1, ind2; check_space=true)
    RA1 = QR_two_pepo(lattice,A1,A2,ind1);
    RA2 = QR_two_pepo(lattice,A1,A2,ind2;side=:right);
    if check_space
        if domain(RA1) != codomain(RA2)
            @error "space mismatch"
        end
    end
    return RA1, RA2
end

# Find the pair of oblique projectors acting on the indices p1 of A1 and p2 of A2
#=
   ┌──┐        ┌──┐   
   │  ├◄──  ─◄─┤  │   
─◄─┤P1│        │P2├◄──
   │  ├◄──  ─◄─┤  │   
   └──┘        └──┘   
=#

function find_P1P2(lattice::Triangular, A1, A2, ind1, ind2, trunc; check_space=true, proj = :svd)
    R1, R2 = R1R2(lattice, A1, A2, ind1, ind2; check_space=check_space)
    return oblique_projector(R1, R2, trunc; proj)
end

function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,2,6},AbstractTensorMap{E,S,2,6}},
    trunc_alg::NoEnvTruncation
) where {E,S<:ElementarySpace}
    lattice = Triangular()
    P120,P300 = find_P1P2(lattice,A[1],A[2],3,6,trunc_alg.trscheme; proj = trunc_alg.projectors);
    P60,P240 = find_P1P2(lattice,A[1],A[2],4,7,trunc_alg.trscheme; proj = trunc_alg.projectors);
    P0,P180 = find_P1P2(lattice,A[1],A[2],5,8,trunc_alg.trscheme; proj = trunc_alg.projectors);

    @tensor opt=true Onew[-1 -2; -3 -4 -5 -6 -7 -8] := A[1][1 -2; 8 9 10 11 12 13] * A[2][-1 1; 2 3 4 5 6 7] * P120[2 8; -3] * P60[3 9; -4] * P0[4 10; -5] * P300[-6; 5 11] * P240[-7; 6 12] * P180[-8; 7 13]
    return Onew, nothing
end
