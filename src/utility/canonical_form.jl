struct Canonicalization
    decomposition_alg
    maxiter::Int
    tol_canonical::Float64
    tol_reconstruction::Float64
    verbosity::Int
end

function Canonicalization(; decomposition_alg = TensorKit.QR(), maxiter = 100, tol_canonical = 1e-7, tol_reconstruction = 1e-10, verbosity = 0)
    return Canonicalization(decomposition_alg, maxiter, tol_canonical, tol_reconstruction, verbosity)
end

function canonicalize_QR(A::TensorMap{T,E,2,4}, canoc_alg::Canonicalization) where {T,E}
    Q = copy(A)
    ϵs = []
    A_errors = []
    Rs = [id(scalartype(A), domain(A)[dir]) for dir = 1:4]
    _, S, _ = tsvd(A)
    for i = 1:canoc_alg.maxiter
        for dir = 3:6
            Q, R = leftorth(Q, (Tuple(setdiff(1:6, dir)),(dir,)); alg = canoc_alg.decomposition_alg)
            Q = permute(Q, ((1,2),Tuple(insert!([3,4,5], dir-2, 6))))

            if dir > 4
                @tensor Rs[dir-2][-1; -2] := twist(R, 1)[-1; 1] * Rs[dir-2][1; -2]
            else
                @tensor Rs[dir-2][-1; -2] := R[-1; 1] * Rs[dir-2][1; -2]
            end
            _, Snew, _ = tsvd(Q)
            ϵ = norm(Snew - S)
            S = copy(Snew)
            push!(ϵs, ϵ)
            @tensor A′[-1 -2; -3 -4 -5 -6] := Q[-1 -2; 1 2 3 4] * Rs[1][1; -3] * Rs[2][2; -4] * twist(Rs[3],1)[3; -5] * twist(Rs[4],1)[4; -6]
            ϵ_A = norm(A - A′)/norm(A)
            push!(A_errors, ϵ_A)
            if ϵ < canoc_alg.tol_canonical
                if canoc_alg.verbosity >= 2
                    @info "Canonicalization converged after $i iterations. Error is $ϵ with reconstruction error $(ϵ_A)"
                end            
                return Q, Rs
            end
            if ϵ_A > canoc_alg.tol_reconstruction
                if canoc_alg.verbosity >= 1
                    @warn "Canonicalization aborted after $i iterations. Error is $ϵ with reconstruction error $(ϵ_A)"
                end            
                return Q, Rs
            end
            if canoc_alg.verbosity >= 3
                @info "Iteration $i and dir = $dir. Error is $ϵ with reconstruction error $(ϵ_A)"
            end
        end
    end
    if canoc_alg.verbosity >= 1
        @warn "Canonicalization not converged after $(canoc_alg.maxiter) iterations. Error is $(ϵs[end])"
    end
    return Q, Rs

end

function canonicalize(A, canoc_alg::Canonicalization)
    Q, Rs = canonicalize_QR(A, canoc_alg)
    @tensor h_bond[-1; -2] := Rs[2][-1; 1] * twist(Rs[4],1)[-2; 1]
    @tensor v_bond[-1; -2] := Rs[1][-1; 1] * twist(Rs[3],1)[-2; 1]

    Uv, Sv, Vv = tsvd(v_bond)
    Uh, Sh, Vh = tsvd(h_bond)

    MN = Uv * sqrt(Sv)
    MS = sqrt(Sv) * Vv
    ME = Uh * sqrt(Sh)
    MW = sqrt(Sh) * Vh
    @tensor A′[-1 -2; -3 -4 -5 -6] := Q[-1 -2; 1 2 3 4] * MN[1; -3] * ME[2; -4] * MS[-5; 3] * MW[-6; 4]
    return A′
end

function canonicalize(A, ::Nothing)
    return A
end
