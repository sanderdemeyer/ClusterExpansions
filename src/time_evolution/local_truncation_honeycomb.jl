# QR decomposition
function QR_two_pepo_left(lattice::Honeycomb, O1, O2, ind)
    pb = (1, ind)
    _, Rb = leftorth(O1, (Tuple(setdiff(1:5, pb)), pb))
    pt = (2, ind)
    _, Rt = leftorth(O2, (Tuple(setdiff(1:5, pt)), pt))
    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = leftorth(M, (3, 4), (1, 2))
    return R
end

# function QR_two_pepo_right(lattice::Honeycomb, O1, O2, ind)
#     pb = (1, ind)
#     Rb, _ = leftorth(O1, (pb, Tuple(setdiff(1:5, pb))))
#     pt = (2, ind)
#     Rt, _ = leftorth(O2, (pt, Tuple(setdiff(1:5, pt))))
#     @tensor M[-1 -2; -3 -4] := Rt[1 -1;-3] * Rb[1 -2;-4]
#     R, _ = leftorth(M, (1, 2), (3, 4))
#     return R
# end

function QR_two_pepo(lattice::Honeycomb, O1, O2, ind; side = :left)
    if side == :left
        return QR_two_pepo_left(lattice, O1, O2, ind)
    elseif side == :right
        return QR_two_pepo_left(lattice, O1, O2, ind)
    else
        @error "side shoulde be :left or :right"
    end
end

function R1R2(lattice::Honeycomb, A1, A2, ind; check_space = true)
    RA1 = QR_two_pepo(lattice, A1, A2, ind)
    RA2 = flip(permute(RA1, ((2,3),(1,))), [1 2 3])
    # RA2 = QR_two_pepo(lattice, A1, A2, ind2; side = :right)
    if check_space
        if domain(RA1) != codomain(RA2)
            @error "space mismatch"
        end
    end
    return RA1, RA2
end

function eig_with_truncation_honeycomb(x, space)
    T = scalartype(x)
    D = dim(space)
    eigval, eigvec = eig(x)
    if space == domain(eigval)[1]
        return eigval, eigvec
    end
    eigval_trunc = zeros(T, space, space)
    eigvec_trunc = zeros(T, codomain(x), space)
    eigval_trunc[] = eigval[][1:D, 1:D]
    eigvec_trunc[] = eigvec[][:,:, 1:D]
    return eigval_trunc, eigvec_trunc
end

function eig_with_truncation_honeycomb_small(x, space)
    T = scalartype(x)
    D = dim(space)
    eigval, eigvec = eig(x)
    if space == domain(eigval)[1]
        return eigval, eigvec
    end
    eigval_trunc = zeros(T, space, space)
    eigvec_trunc = zeros(T, codomain(x), space)
    eigval_trunc[] = eigval[][1:D, 1:D]
    eigvec_trunc[] = eigvec[][:, 1:D]
    return eigval_trunc, eigvec_trunc
end

function oblique_projector(lattice::Honeycomb, R1, R2, trunc; proj = :svd)
    mat = R1 * R2
    if proj == :svd
        U, S, Vt = tsvd(mat; trunc)
        P1 = R2 * adjoint(Vt) * inv(sqrt(S))
        P2 = inv(sqrt(S)) * adjoint(U) * R1
    elseif proj == :eig
        dims = minimum([dim(domain(mat)) trunc.dim])
        D, V = eig_with_truncation_honeycomb_small(mat, ℂ^(dims))
        P1 = R2 * V * inv(sqrt(D))
        P2 = inv(sqrt(D)) * adjoint(V) * R1
    end
    return P1, P2
end

function find_P1P2(lattice::Honeycomb, A1, A2, ind, trunc; check_space = true, proj = :svd)
    R1, R2 = R1R2(lattice, A1, A2, ind; check_space = check_space)
    return oblique_projector(lattice, R1, R2, trunc; proj)
end

function approximate_state(
    A::Tuple{AbstractTensorMap{E, S, 2, 3}, AbstractTensorMap{E, S, 2, 3}},
    trunc_alg::NoEnvTruncation
) where {E, S <: ElementarySpace}
    lattice = Honeycomb()
    PNW,  = find_P1P2(lattice, A[1], A[2], 3, trunc_alg.trscheme; proj = trunc_alg.projectors)
    # PE,  = find_P1P2(lattice, A[1], A[2], 4, trunc_alg.trscheme; proj = trunc_alg.projectors)
    # PSW, = find_P1P2(lattice, A[1], A[2], 5, trunc_alg.trscheme; proj = trunc_alg.projectors)
    PE = flip(PNW, [1 2 3])
    PSW = flip(PNW, [1 2 3])
    @tensor opt = true Onew[-1 -2; -3 -4 -5] := A[1][1 -2; 5 6 7] * A[2][-1 1; 2 3 4] * PNW[2 5; -3] * PE[3 6; -4] * PSW[4 7; -5]
    return Onew, nothing
end
