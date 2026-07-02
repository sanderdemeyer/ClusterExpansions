abstract type Lattice end

struct Cubic <: Lattice
end

# QR decomposition
function QR_two_pepo_left(lattice::Cubic, O1, O2, ind)
    pb = (1, ind)
    _, Rb = leftorth(O1, (Tuple(setdiff(1:8, pb)), pb))
    pt = (2, ind)
    _, Rt = leftorth(O2, (Tuple(setdiff(1:8, pt)), pt))
    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = leftorth(M, (3, 4), (1, 2))
    return R
end

function QR_two_pepo_right(lattice::Cubic, O1, O2, ind)
    pb = (1, ind)
    Rb, _ = rightorth(O1, (pb, Tuple(setdiff(1:8, pb))))
    pt = (2, ind)
    Rt, _ = rightorth(O2, (pt, Tuple(setdiff(1:8, pt))))
    @tensor M[-1 -2; -3 -4] := Rt[1 -1;-3] * Rb[1 -2;-4]
    R, _ = rightorth(M, (1, 2), (3, 4))
    return R
end

function QR_two_pepo(lattice::Cubic, O1, O2, ind; side = :left)
    if side == :left
        return QR_two_pepo_left(lattice, O1, O2, ind)
    elseif side == :right
        return QR_two_pepo_right(lattice, O1, O2, ind)
    else
        @error "side shoulde be :left or :right"
    end
end

function R1R2(lattice::Cubic, A1, A2, ind1, ind2; check_space = true)
    RA1 = QR_two_pepo(lattice, A1, A2, ind1)
    RA2 = QR_two_pepo(lattice, A1, A2, ind2; side = :right)
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

function oblique_projector(lattice::Cubic, R1, R2, trunc; proj = :svd)
    mat = R1 * R2
    if proj == :svd
        U, S, Vt = tsvd(mat; trunc)
        P1 = R2 * adjoint(Vt) * inv(sqrt(S))
        P2 = inv(sqrt(S)) * adjoint(U) * R1
    elseif proj == :eig
        dims = minimum([dim(domain(mat)) trunc.dim])
        D, V = eig_with_truncation_triangular(mat, ℂ^(dims))
        P1 = R2 * V * inv(sqrt(D))
        P2 = inv(sqrt(D)) * adjoint(V) * R1
    end
    return P1, P2
end

function find_P1P2(lattice::Cubic, A1, A2, ind1, ind2, trunc; check_space = true, proj = :svd)
    R1, R2 = R1R2(lattice, A1, A2, ind1, ind2; check_space = check_space)
    return oblique_projector(lattice, R1, R2, trunc; proj)
end

# Functions to permute (flipped and unflipped) tensors under 60 degree rotation
function rotl60_pf(T::TensorMap{A, S, 3, 3}) where {A, S}
    return permute(T, ((4, 1, 2), (5, 6, 3)))
end

function rotl60_pf(T::TensorMap{A, S, 0, 6}) where {A, S}
    return permute(T, ((), (2, 3, 4, 5, 6, 1)))
end

function approximate_state(
        A::Tuple{AbstractTensorMap{E, S, 2, 6}, AbstractTensorMap{E, S, 2, 6}},
        trunc_alg::NoEnvTruncation; rot_invariant::Bool = false
    ) where {E, S <: ElementarySpace}
    lattice = Cubic()
    PU,PD = find_P1P2(lattice,A[1],A[2],3,6,trunc_alg.trscheme)
    if rot_invariant
        PN = PE = copy(PU)
        PS = PW = copy(PD)
    else
        PN,PS = find_P1P2(lattice,A[1],A[2],4,7,trunc_alg.trscheme);
        PE,PW = find_P1P2(lattice,A[1],A[2],5,8,trunc_alg.trscheme);
    end
    @tensor opt = true Onew[-1 -2; -3 -4 -5 -6 -7 -8] := A[1][1 -2; 8 9 10 11 12 13] * A[2][-1 1; 2 3 4 5 6 7] * PU[2 8; -3] * PN[3 9; -4] * PE[4 10; -5] * PD[-6; 5 11] * PS[-7; 6 12] * PW[-8; 7 13]
    return Onew, nothing
end