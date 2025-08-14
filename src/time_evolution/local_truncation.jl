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
function QR_proj(A, p1, p2; check_space=true)
    len = length(codomain(A)) + 8
    q1 = Tuple(setdiff(1:len, p1))
    _, RA1 = leftorth(A, (q1, p1))
    q2 = Tuple(setdiff(1:len, p2))
    RA2, _ = rightorth(A, (p2, q2))
    if check_space
        if domain(RA1) != codomain(RA2)
            throw(SpaceMismatch("domain and codomain of projectors do not match"))
        end
    end
    return RA1, RA2
end

function oblique_projector(R1, R2, trunc)
    mat = R1 * R2

    U, S, Vt = tsvd(mat; trunc)
    P1 = R2 * adjoint(Vt) * inv(sqrt(S))
    P2 = inv(sqrt(S)) * adjoint(U) * R1

    return P1, P2
end

function find_proj(A, p1, p2, trunc; check_space=true)
    R1, R2 = QR_proj(A, p1, p2; check_space=check_space)
    P1, P2 = oblique_projector(R1, R2, trunc)
    return P1, P2
end

function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    trunc_alg::NoEnvTruncation
) where {E,S<:ElementarySpace}
    @tensor Otmp[-1 -2; -3 -4 -5 -6 -7 -8 -9 -10] := A[1][1 -2; -7 -8 -9 -10] * A[2][-1 1; -3 -4 -5 -6];
    PN, PS = find_proj(Otmp, (3,7), (5,9), trunc_alg.trscheme);
    PE, PW = find_proj(Otmp, (4,8), (6,10), trunc_alg.trscheme);

    @tensor opt=true contractcheck=true Onew[-1 -2; -3 -4 -5 -6] := Otmp[-1 -2; 3 4 5 6 7 8 9 10] * PN[3 7; -3] * PE[4 8; -4] * PS[-5; 5 9] * PW[-6; 6 10]
    return Onew, nothing
end

