abstract type Lattice end

struct Square <: Lattice end


struct NoEnvTruncation <: EnvTruncation
    trscheme::MatrixAlgebraKit.TruncationStrategy
    check_fidelity::Bool
    verbosity::Int
    rot_invariance::Bool
end

function NoEnvTruncation(trscheme::MatrixAlgebraKit.TruncationStrategy;
    check_fidelity::Bool = false, verbosity::Int = 0, rot_invariance::Bool = false)
    NoEnvTruncation(trscheme, check_fidelity, verbosity, rot_invariance)
end


# Find the pair of oblique projectors acting on the indices p1 of A1 and p2 of A2
#=
    ┌──┐      ┌──┐    
 ─◄─┤  │      │  ├◄── 
    │P1├◄───◄─┤P2│    
 ─◄─┤  │      │  ├◄── 
    └──┘      └──┘    
=#

# Transpose of X: A ← B as a map B ← A on the same (non-dual) spaces, keeping the order of the legs
transpose_flip(X) = flip(permute(X, (domainind(X), codomainind(X))), Tuple(1:numind(X)))

# Takagi factorisation of a symmetric mat = transpose_flip(mat) as mat ≈ W * S * transpose_flip(W'),
# with W an isometry and S ≥ 0 the (truncated) singular values
function takagi_trunc(mat; trunc)
    U, S, Vᴴ, ϵ = svd_trunc(mat; trunc)

    # For symmetric mat, conj(V) = U * Z with Z a symmetric unitary commuting with S
    Z = U' * transpose_flip(Vᴴ)
    W = U * sqrt(Z)
    return W, S, ϵ
end

function oblique_projector(R1, R2, trunc; proj = :svd)
    mat = R1 * R2
    if proj == :takagi
        W, S, ϵ = takagi_trunc(mat; trunc)
        P2 = inv(sqrt(S)) * W' * R1
        P1 = transpose_flip(P2) # = R2 * conj(W) * inv(sqrt(S)) when R2 = transpose_flip(R1)
    elseif proj == :svd
        U, S, Vt, ϵ = svd_trunc(mat; trunc)
        P1 = R2 * adjoint(Vt) * inv(sqrt(S))
        P2 = inv(sqrt(S)) * adjoint(U) * R1
    elseif proj == :eig
        D, V, ϵ = eig_trunc(mat; trunc)
        D = complex(D)
        P1 = R2 * V * inv(sqrt(D))
        P2 = inv(sqrt(D)) * adjoint(V) * R1
    end
    return P1, P2, ϵ
end

function QR_two_pepo_left(::Square,O1,O2,ind)
    pb = (1,ind)
    _, Rb= left_orth(permute(O1,(Tuple(setdiff(1:6, pb)),pb)))
    pt = (2,ind)
    _, Rt= left_orth(permute(O2,(Tuple(setdiff(1:6, pt)),pt)))
    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = left_orth(permute(M, ((3,4),(1,2))))
    return R
end

function QR_two_pepo_left(::Square,O,ind)
    _, R= left_orth(permute(O,(Tuple(setdiff(1:6, ind)),(ind,))))
    return R
end

function QR_two_pepo_left(::Square,ψ::AbstractTensorMap{E,S,1,4},O::AbstractTensorMap{E,S,2,4},ind) where {E,S}
    pb = (1,ind-1)
    _, Rb= left_orth(permute(ψ,(Tuple(setdiff(1:5, pb)),pb)))
    pt = (2,ind)
    _, Rt= left_orth(permute(O,(Tuple(setdiff(1:6, pt)),pt)))
    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = left_orth(permute(M, ((3,4),(1,2))))
    return R
end

function QR_two_pepo_right(::Square,O1,O2,ind)
    pb = (1,ind)
    Rb, _ = right_orth(permute(O1, (pb, Tuple(setdiff(1:6,pb)))))
    pt = (2,ind)
    Rt, _ = right_orth(permute(O2, (pt, Tuple(setdiff(1:6,pt)))))
    @tensor M[-1 -2; -3 -4] := Rt[1 -1;-3] * Rb[1 -2;-4]
    R, _ = right_orth(permute(M, ((1,2),(3,4))))
    return R
end

function QR_two_pepo_right(::Square,O,ind)
    R, _ = right_orth(permute(O, ((ind,), Tuple(setdiff(1:6,ind)))))
    return R
end

function QR_two_pepo_right(::Square,ψ::AbstractTensorMap{E,S,1,4},O::AbstractTensorMap{E,S,2,4},ind) where {E,S}
    pb = (1,ind-1)
    Rb, _ = right_orth(permute(ψ, (pb, Tuple(setdiff(1:5,pb)))))
    pt = (2,ind)
    Rt, _ = right_orth(permute(O, (pt, Tuple(setdiff(1:6,pt)))))
    @tensor M[-1 -2; -3 -4] := Rt[1 -1;-3] * Rb[1 -2;-4]
    R, _ = right_orth(permute(M, ((1,2),(3,4))))
    return R
end

function QR_two_pepo(lattice,A,ind;side=:left)
    if side==:left
        return QR_two_pepo_left(lattice,A...,ind)
    elseif side==:right
        return QR_two_pepo_right(lattice,A...,ind)
    else
        @error "side shoulde be :left or :right"
    end
end

function R1R2(lattice, A, ind1, ind2; check_space=true)
    RA1 = QR_two_pepo(lattice,A,ind1);
    RA2 = QR_two_pepo(lattice,A,ind2;side=:right);
    if check_space
        if domain(RA1) != codomain(RA2)
            @error "space mismatch"
        end
    end
    return RA1, RA2
end

function find_P1P2(lattice, A, ind1, ind2, trunc; check_space=true, proj = :svd)
    if proj == :takagi
        # For symmetric tensors, the LQ on ind2 equals the transposed QR on ind1 up to a gauge, which is fixed here
        R1 = QR_two_pepo(lattice, A, ind1)
        R2 = transpose_flip(R1)
    else
        R1, R2 = R1R2(lattice, A, ind1, ind2; check_space = check_space)
    end
    return oblique_projector(R1, R2, trunc; proj)
end

function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    trunc_alg::NoEnvTruncation; rot_invariance::Bool = false
) where {E,S<:ElementarySpace}
    proj = rot_invariance ? :takagi : :svd
    lattice = Square()
    PN,PS,ϵv = find_P1P2(lattice,A,3,5,trunc_alg.trscheme; proj);
    if rot_invariance
        PE = copy(PN)
        PW = copy(PS)
        ϵh = ϵv
    else
        PE,PW,ϵh = find_P1P2(lattice,A,4,6,trunc_alg.trscheme; proj);
    end
    @tensor opt=true Onew[-1 -2; -3 -4 -5 -6] := A[1][1 -2; 7 8 9 10] * A[2][-1 1; 3 4 5 6] * PN[3 7; -3] * PE[4 8; -4] * PS[-5; 5 9] * PW[-6; 6 10]
    return Onew, maximum([ϵv,ϵh])
end

function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},
    trunc_alg::NoEnvTruncation; rot_invariance::Bool = false
) where {E,S<:ElementarySpace}
    proj = rot_invariance ? :takagi : :svd
    lattice = Square()
    PN,PS,ϵv = find_P1P2(lattice,A,3,5,trunc_alg.trscheme; proj);
    if rot_invariance
        PE = copy(PN)
        PW = copy(PS)
        ϵh = ϵv
    else
        PE,PW,ϵh = find_P1P2(lattice,A,4,6,trunc_alg.trscheme; proj);
    end
    @tensor opt=true Onew[-1; -3 -4 -5 -6] := A[1][1; 7 8 9 10] * A[2][-1 1; 3 4 5 6] * PN[3 7; -3] * PE[4 8; -4] * PS[-5; 5 9] * PW[-6; 6 10]
    return Onew, maximum([ϵv,ϵh])
end

function approximate_state(
    A::AbstractTensorMap{E,S,2,4},
    trunc_alg::NoEnvTruncation; rot_invariance::Bool = false
) where {E,S<:ElementarySpace}
    proj = rot_invariance ? :takagi : :svd
    lattice = Square()
    PN,PS,ϵv = find_P1P2(lattice,(A,),3,5,trunc_alg.trscheme; proj);
    if rot_invariance
        PE = copy(PN)
        PW = copy(PS)
        ϵh = ϵv
    else
        PE,PW,ϵh = find_P1P2(lattice,(A,),4,6,trunc_alg.trscheme; proj);
    end
    @tensor opt=true Onew[-1 -2; -3 -4 -5 -6] := A[-1 -2; 1 2 3 4] * PN[1; -3] * PE[2; -4] * PS[-5; 3] * PW[-6; 4]
    return Onew, maximum([ϵv,ϵh])
end
