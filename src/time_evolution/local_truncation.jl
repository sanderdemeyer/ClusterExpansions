struct NoEnvTruncation <: EnvTruncation
    trscheme::TruncationScheme
    check_fidelity::Bool
    verbosity::Int
end

function NoEnvTruncation(trscheme::TruncationScheme;
    check_fidelity::Bool = false, verbosity::Int = 0)
    NoEnvTruncation(trscheme, check_fidelity, verbosity)
end

# QR decomposition
function QR_two_pepo_left(O1,O2,ind)
    pb = (1,ind)
    _, Rb= leftorth(O1,(Tuple(setdiff(1:6, pb)),pb))
    pt = (2,ind)
    _, Rt= leftorth(O2,(Tuple(setdiff(1:6, pt)),pt))
    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = leftorth(M, (3,4),(1,2))
    return R
end

function QR_two_pepo_right(O1,O2,ind)
    pb = (1,ind)
    Rb, _ = rightorth(O1, (pb, Tuple(setdiff(1:6,pb))))
    pt = (2,ind)
    Rt, _ = rightorth(O2, (pt, Tuple(setdiff(1:6,pt))))
    @tensor M[-1 -2; -3 -4] := Rt[1 -1;-3] * Rb[1 -2;-4]
    R, _ = rightorth(M, (1,2),(3,4))
    return R
end

function QR_two_pepo(O1,O2,ind;side=:left)
    if side==:left
        return QR_two_pepo_left(O1,O2,ind)
    elseif side==:right
        return QR_two_pepo_right(O1,O2,ind)
    else
        @error "side shoulde be :left or :right"
    end
end

function R1R2(A1, A2, ind1, ind2; check_space=true)
    RA1 = QR_two_pepo(A1,A2,ind1);
    RA2 = QR_two_pepo(A1,A2,ind2;side=:right);
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

function find_P1P2(A1, A2, ind1, ind2, trunc; check_space=true)
    R1, R2 = R1R2(A1, A2, ind1, ind2; check_space=check_space)
    return oblique_projector(R1, R2, trunc)
end

function oblique_projector(R1, R2, trunc)
    mat = R1 * R2

    U, S, Vt = tsvd(mat; trunc)
    P1 = R2 * adjoint(Vt) * inv(sqrt(S))
    P2 = inv(sqrt(S)) * adjoint(U) * R1

    return P1, P2
end

function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    trunc_alg::NoEnvTruncation
) where {E,S<:ElementarySpace}
    PN,PS = find_P1P2(A[1],A[2],3,5,trunc_alg.trscheme);
    PE,PW = find_P1P2(A[1],A[2],4,6,trunc_alg.trscheme);
    @tensor opt=true Onew[-1 -2; -3 -4 -5 -6] := A[1][1 -2; 7 8 9 10] * A[2][-1 1; 3 4 5 6] * PN[3 7; -3] * PE[4 8; -4] * PS[-5; 5 9] * PW[-6; 6 10]
    return Onew, nothing
end

# Approximation of a single PEPO tensor
function QR_two_pepo_left(O,ind)
    _, R= leftorth(O,(Tuple(setdiff(1:6, ind)),(ind,)))
    return R
end

function QR_two_pepo_right(O,ind)
    R, _ = rightorth(O, ((ind,), Tuple(setdiff(1:6,ind))))
    return R
end

function QR_two_pepo(O,ind;side=:left)
    if side==:left
        return QR_two_pepo_left(O,ind)
    elseif side==:right
        return QR_two_pepo_right(O,ind)
    else
        @error "side shoulde be :left or :right"
    end
end

function R1R2(A, ind1, ind2; check_space=true)
    RA1 = QR_two_pepo(A,ind1);
    RA2 = QR_two_pepo(A,ind2;side=:right);
    if check_space
        if domain(RA1) != codomain(RA2)
            @error "space mismatch"
        end
    end
    return RA1, RA2
end

function find_P1P2(A, ind1, ind2, trunc; check_space=true)
    R1, R2 = R1R2(A, ind1, ind2; check_space=check_space)
    return oblique_projector(R1, R2, trunc)
end

function approximate_state(
    A::AbstractTensorMap{E,S,2,4},
    trunc_alg::NoEnvTruncation
) where {E,S<:ElementarySpace}
    PN,PS = find_P1P2(A,3,5,trunc_alg.trscheme);    
    PE,PW = find_P1P2(A,4,6,trunc_alg.trscheme);
    @tensor opt=true Onew[-1 -2; -3 -4 -5 -6] := A[-1 -2; 1 2 3 4] * PN[1; -3] * PE[2; -4] * PS[-5; 3] * PW[-6; 4]
    return Onew, nothing
end

# Application of PEPO on PEPS
function QR_two_pepo_left(ψ::AbstractTensorMap{E,S,1,4},O::AbstractTensorMap{E,S,2,4},ind) where {E,S}
    pb = (1,ind-1)
    _, Rb= leftorth(ψ,(Tuple(setdiff(1:5, pb)),pb))
    pt = (2,ind)
    _, Rt= leftorth(O,(Tuple(setdiff(1:6, pt)),pt))
    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = leftorth(M, (3,4),(1,2))
    return R
end

function QR_two_pepo_right(ψ::AbstractTensorMap{E,S,1,4},O::AbstractTensorMap{E,S,2,4},ind) where {E,S}
    pb = (1,ind-1)
    Rb, _ = rightorth(ψ, (pb, Tuple(setdiff(1:5,pb))))
    pt = (2,ind)
    Rt, _ = rightorth(O, (pt, Tuple(setdiff(1:6,pt))))
    @tensor M[-1 -2; -3 -4] := Rt[1 -1;-3] * Rb[1 -2;-4]
    R, _ = rightorth(M, (1,2),(3,4))
    return R
end

function QR_two_pepo(ψ::AbstractTensorMap{E,S,1,4},O::AbstractTensorMap{E,S,2,4},ind;side=:left) where {E,S}
    if side==:left
        return QR_two_pepo_left(ψ,O,ind)
    elseif side==:right
        return QR_two_pepo_right(ψ,O,ind)
    else
        @error "side shoulde be :left or :right"
    end
end

function R1R2(ψ::AbstractTensorMap{E,S,1,4},O::AbstractTensorMap{E,S,2,4}, ind1, ind2; check_space=true) where {E,S}
    RA1 = QR_two_pepo(ψ,O,ind1);
    RA2 = QR_two_pepo(ψ,O,ind2;side=:right);
    if check_space
        if domain(RA1) != codomain(RA2)
            @error "space mismatch"
        end
    end
    return RA1, RA2
end

function find_P1P2(ψ::AbstractTensorMap{E,S,1,4},O::AbstractTensorMap{E,S,2,4}, ind1, ind2, trunc; check_space=true) where {E,S}
    R1, R2 = R1R2(ψ,O, ind1, ind2; check_space=check_space)
    return oblique_projector(R1, R2, trunc)
end

function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},
    trunc_alg::NoEnvTruncation
) where {E,S<:ElementarySpace}
    PN,PS = find_P1P2(A[1],A[2],3,5,trunc_alg.trscheme);
    PE,PW = find_P1P2(A[1],A[2],4,6,trunc_alg.trscheme);
    @tensor opt=true Onew[-1; -3 -4 -5 -6] := A[1][1; 7 8 9 10] * A[2][-1 1; 3 4 5 6] * PN[3 7; -3] * PE[4 8; -4] * PS[-5; 5 9] * PW[-6; 6 10]
    return Onew, nothing
end
