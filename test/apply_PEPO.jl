using PEPSKit
using TensorKit
using KrylovKit
import PEPSKit: @autoopt

function own_isometry(Ws)
    for dir in 1:4
        (D, DO) = dims(Ws[dir].codom)
        for k in 1:DO
            if dir == 1
                Oᵏ = sum(O[][:, :, k, :, :, :])
            elseif dir == 2
                Oᵏ = sum(O[][:, :, :, k, :, :])
            elseif dir == 3
                Oᵏ = sum(O[][:, :, :, :, k, :])
            elseif dir == 4
                Oᵏ = sum(O[][:, :, :, :, :, k])
            end
            for j in 1:D
                Ws[dir][j, k, j] = 1.0 / (Oᵏ * DO)
            end
        end
    end
    return Ws
end

function initialize_isometry(
    ψ::Union{AbstractTensorMap{S,1,4},AbstractTensorMap{S,2,4}},
    O::AbstractTensorMap{S,2,4};
    initial_guess="random",
    space=ψ.dom[1],
) where {S}
    if initial_guess == "random"
        return [isdual(ψ.dom[i]) ? TensorMap(randn, ψ.dom[i] ⊗ O.dom[i], space') : TensorMap(randn, ψ.dom[i] ⊗ O.dom[i], space) for i in 1:4]
    elseif initial_guess == "isometry"
        return [isdual(ψ.dom[i]) ? isometry(ψ.dom[i] ⊗ O.dom[i], space') : isometry(ψ.dom[i] ⊗ O.dom[i], space) for i in 1:4]
    elseif initial_guess == "zeros"
        @warn "This will probably give errors"
        return [isdual(ψ.dom[i]) ? TensorMap(zeros, ψ.dom[i] ⊗ O.dom[i], space') : TensorMap(zeros, ψ.dom[i] ⊗ O.dom[i], space) for i in 1:4]
    elseif initial_guess == "SVD"
        Ws = [TensorMap(zeros, ψ.dom[i] ⊗ O.dom[i], space) for i in 1:4]

        @autoopt @tensor T[DNa DONa; DNb DONb] :=
            ψ[Dpa; DNa DE DS DW] *
            O[DOp Dpa; DONa DOE DOS DOW] *
            conj(O[DOp Dpb; DONb DOE DOS DOW]) *
            conj(ψ[Dpb; DNb DE DS DW])
        U, Σ, V = tsvd(T; trunc=truncspace(space))
        U = U * sqrt(Σ)
        V = sqrt(Σ) * V
        for dir in 1:4
            Ws[dir][][:, :, :] = U[][:, :, :]
        end
        return Ws
    elseif initial_guess == "eigen"
        Ws = [TensorMap(zeros, ψ.dom[i] ⊗ O.dom[i], space) for i in 1:4]

        @autoopt @tensor T[DNa DONa; DNb DONb] :=
            ψ[Dpa; DNa DE DS DW] *
            O[DOp Dpa; DONa DOE DOS DOW] *
            conj(O[DOp Dpb; DONb DOE DOS DOW]) *
            conj(ψ[Dpb; DNb DE DS DW])

        _, V = eigen(T)
        W2_new = TensorMap(zeros, ComplexF64, ψ.dom[2] ⊗ O.dom[2], space)
        W2_new[][:, :, :] = V[][:, :, 1:dim(space)]

        for dir in 1:4
            Ws[dir][][:, :, :] = V[][:, :, 1:dim(space)]
        end
        return Ws
    elseif initial_guess == "SVD environment"
        ψnew = InfinitePEPS(apply_and_fuse(ψ, O))

        χenv = 12
        env0 = CTMRGEnv(ψnew, ComplexSpace(χenv));
        env = leading_boundary(env0, ψnew, ctm_alg);

        @autoopt @tensor E[DNa; DSa] := env.corners[1, 1, 1][χ6; χ1] *
        env.edges[1, 1, 1][χ1; DNa Db χ2] *
        env.corners[2, 1, 1][χ2; χ3] *
        env.corners[3, 1, 1][χ3; χ4] *
        env.edges[3, 1, 1][χ4; DSa Db χ5] *
        env.corners[4, 1, 1][χ5; χ6]

        U, Σ, V = tsvd(E, trunc = truncdim(dim(ψ.dom[1])))
        Ws = [TensorMap(zeros, ComplexF64, ψ.dom[i] ⊗ O.dom[i], space) for i in 1:4]
        for dir in 1:4
            Ws[dir][][:, :, :] = reshape(V[][:, :, :], (dim(ψ.dom[i]), dim(O.dom[i]), dim(space)))
        end
        return Ws
    else
        @error "Type of initial guess `$(initial_guess)` not defined"
    end
end

function approximate(ψ::AbstractTensorMap{S,1,4}, O::AbstractTensorMap{S,2,4}, Ws) where {S}
    @tensor A[-1; -2 -3 -4 -5] :=
        ψ[1; 2 4 6 8] *
        O[-1 1; 3 5 7 9] *
        Ws[1][2 3; -2] *
        Ws[2][4 5; -3] *
        Ws[3][6 7; -4] *
        Ws[4][8 9; -5]
    return A
end

function approximate(ψ::AbstractTensorMap{S,1,4}, Ws) where {S}
    @tensor A[-1; -2 -3 -4 -5] :=
        ψ[-1; 1 2 3 4] *
        Ws[1][1; -2] *
        Ws[2][2; -3] *
        Ws[3][3; -4] *
        Ws[4][4; -5]
    return A
end

function approximate(ψ::AbstractTensorMap{S,2,4}, O::AbstractTensorMap{S,2,4}, Ws) where {S}
    @tensor A[-1 -2; -3 -4 -5 -6] :=
        ψ[1 -2; 2 4 6 8] *
        O[-1 1; 3 5 7 9] *
        Ws[1][2 3; -3] *
        Ws[2][4 5; -4] *
        Ws[3][6 7; -5] *
        Ws[4][8 9; -6]
    return A
end

function update_isometry(
    ψ::AbstractTensorMap{S,1,4}, O::AbstractTensorMap{S,2,4}, Ws, χenv; space=ψ.dom[2]
) where {S}
    A = approximate(ψ, O, Ws)

    ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
    )

    A2 = InfinitePEPS(A)
    env0 = CTMRGEnv(A2, ℂ^χenv)

    env = leading_boundary(env0, A2, ctm_alg)

    @autoopt @tensor E[DLaE1 DLaE2; DRaW1 DRaW2] :=
        env.corners[1, 1, 1][χ8; χ1] *
        env.edges[1, 1, 1][χ1; DLaN3 DLbN χN] *
        env.edges[1, 1, 1][χN; DRaN3 DRbN χ2] *
        env.corners[2, 1, 1][χ2; χ3] *
        env.edges[2, 1, 1][χ3; DRaE3 DRbE χ4] *
        env.corners[3, 1, 1][χ4; χ5] *
        env.edges[3, 1, 1][χ5; DRaS3 DRbS χS] *
        env.edges[3, 1, 1][χS; DLaS3 DLbS χ6] *
        env.corners[4, 1, 1][χ6; χ7] *
        env.edges[4, 1, 1][χ7; DLaW3 DLbW χ8] *
        conj(A[DLpb; DLbN Dbconnected DLbS DLbW]) *
        conj(A[DRpb; DRbN DRbE DRbS Dbconnected]) *
        ψ[DLpa; DLaN1 DLaE1 DLaS1 DLaW1] *
        ψ[DRpa; DRaN1 DRaE1 DRaS1 DRaW1] *
        O[DLpb DLpa; DLaN2 DLaE2 DLaS2 DLaW2] *
        O[DRpb DRpa; DRaN2 DRaE2 DRaS2 DRaW2] *
        Ws[1][DLaN1 DLaN2; DLaN3] *
        Ws[3][DLaS1 DLaS2; DLaS3] *
        Ws[4][DLaW1 DLaW2; DLaW3] *
        Ws[1][DRaN1 DRaN2; DRaN3] *
        Ws[2][DRaE1 DRaE2; DRaE3] *
        Ws[3][DRaS1 DRaS2; DRaS3]
    Diag, V = eigen(E)
    @tensor Etest[-1 -2; -3 -4] := V[-1 -2; 1] * Diag[1; 2] * inv(V)[2; -3 -4]
    @assert norm(E - Etest) / norm(E) < 1e-10 "eigenvalue decomposition is not exact: relative norm difference is $(norm(E-Etest)/norm(E))"

    @tensor unittest1[-1 -2; -3 -4] := V[-1 -2; 1] * inv(V)[1; -3 -4]
    @tensor unittest2[-1; -2] := inv(V)[-1; 1 2] * V[1 2; -2]

    W2_new = isdual(ψ.dom[2]) ? TensorMap(zeros, ComplexF64, ψ.dom[2] ⊗ O.dom[2], space') : TensorMap(zeros, ComplexF64, ψ.dom[2] ⊗ O.dom[2], space)
    W2_new[][:, :, :] = V[][:, :, 1:dim(space)]
    W4_new = isdual(ψ.dom[4]) ? TensorMap(zeros, ComplexF64, ψ.dom[4] ⊗ O.dom[4], space') : TensorMap(zeros, ComplexF64, ψ.dom[4] ⊗ O.dom[4], space)
    W4_new[][:, :, :] = permute(inv(V), ((2, 3), (1,)))[][:, :, 1:dim(space)]

    return [Ws[1], W2_new, Ws[3], W4_new], A
end

function update_isometry(
    ψ::AbstractTensorMap{S,1,4}, Ws, χenv; space = ψ.dom[2]
) where {S}
    A = approximate(ψ, Ws)
    
    ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
    )
    A2 = InfinitePEPS(A)
    env0 = CTMRGEnv(A2, ℂ^χenv)

    env = leading_boundary(env0, A2, ctm_alg)

    @autoopt @tensor E[DLaE1; DRaW1] :=
        env.corners[1, 1, 1][χ8; χ1] *
        env.edges[1, 1, 1][χ1; DLaN3 DLbN χN] *
        env.edges[1, 1, 1][χN; DRaN3 DRbN χ2] *
        env.corners[2, 1, 1][χ2; χ3] *
        env.edges[2, 1, 1][χ3; DRaE3 DRbE χ4] *
        env.corners[3, 1, 1][χ4; χ5] *
        env.edges[3, 1, 1][χ5; DRaS3 DRbS χS] *
        env.edges[3, 1, 1][χS; DLaS3 DLbS χ6] *
        env.corners[4, 1, 1][χ6; χ7] *
        env.edges[4, 1, 1][χ7; DLaW3 DLbW χ8] *
        conj(A[DLp; DLbN Dbconnected DLbS DLbW]) *
        conj(A[DRp; DRbN DRbE DRbS Dbconnected]) *
        ψ[DLp; DLaN1 DLaE1 DLaS1 DLaW1] *
        ψ[DRp; DRaN1 DRaE1 DRaS1 DRaW1] *
        Ws[1][DLaN1; DLaN3] *
        Ws[3][DLaS1; DLaS3] *
        Ws[4][DLaW1; DLaW3] *
        Ws[1][DRaN1; DRaN3] *
        Ws[2][DRaE1; DRaE3] *
        Ws[3][DRaS1; DRaS3]

    Diag, V = eigen(E)
    @tensor Etest[-1; -2] := V[-1; 1] * Diag[1; 2] * inv(V)[2; -2]
    @assert norm(E - Etest) / norm(E) < 1e-10 "eigenvalue decomposition is not exact: relative norm difference is $(norm(E-Etest)/norm(E))"

    @tensor unittest1[-1; -2] := V[-1; 1] * inv(V)[1; -2]
    @tensor unittest2[-1; -2] := inv(V)[-1; 1] * V[1; -2]

    W2_new = isdual(ψ.dom[2]) ? TensorMap(zeros, ComplexF64, ψ.dom[2], space') : TensorMap(zeros, ComplexF64, ψ.dom[2], space)
    W2_new[][:, :] = V[][:, 1:dim(space)]
    W4_new = isdual(ψ.dom[4]) ? TensorMap(zeros, ComplexF64, ψ.dom[4], space') : TensorMap(zeros, ComplexF64, ψ.dom[4], space)
    W4_new[][:, :] = permute(inv(V), ((2,), (1,)))[][:, 1:dim(space)]

    return [Ws[1], W2_new, Ws[3], W4_new], A
end

function apply_and_fuse(
    ψ::AbstractTensorMap{S,1,4},
    O::AbstractTensorMap{S,2,4};    
) where {S}
    Is = [i > 2 ? isometry(ψ.dom[i] ⊗ O.dom[i], fuse(ψ.dom[i], O.dom[i])') : isometry(ψ.dom[i] ⊗ O.dom[i], fuse(ψ.dom[i], O.dom[i])) for i = 1:4]
    @tensor ψnew[-1; -2 -3 -4 -5] := ψ[1; 2 4 6 8] * O[-1 1; 3 5 7 9] * Is[1][2 3; -2] * Is[2][4 5; -3] * Is[3][6 7; -4] * Is[4][8 9; -5]
    return ψnew
end

function apply(
    ψ::AbstractTensorMap{S,1,4},
    O::AbstractTensorMap{S,2,4},
    W0;
    maxiter=50,
    spaces=[ψ.dom[1]],
    χenv=12,
    tol=1e-10,
    verbosity=1,
    initial_guess="random",
) where {S}
    if (verbosity > 0)
        @info "Approximating from $(ψ.dom[1]) ⊗ $(O.dom[1]) to $(spaces)"
    end
    first_approx_space = popfirst!(spaces)
    println("first space = $(first_approx_space), the rest = $(spaces)")
    Ws = W0 #initialize_isometry(ψ, O; initial_guess=initial_guess, space = first_approx_space)
    A = rotr90(approximate(ψ, O, Ws))
    ϵ = 0
    ϵ_Ws = 0
    for i in 1:maxiter
        Ws_old = copy(Ws)
        A_old = copy(A)
        for _ in 1:4
            Ws, A = update_isometry(ψ, O, Ws, χenv; space = first_approx_space)
            ψ = rotl90(ψ)
            O = rotl90(O)
            Ws = circshift(Ws, -1)
        end
        ϵ = norm(A - A_old) / norm(A_old)
        ϵ_Ws = ([norm(Ws[i] - Ws_old[i]) / norm(Ws_old[i]) for i = 1:4])
        if ϵ < tol
            @info "Converged after $i iterations: norm difference in A is $ϵ"
            ψnew = rotl90(A)
            ψintermediate = ψnew * norm(ψ)/norm(ψnew)
            for space = spaces
                ψintermediate = approximate_iteratively(ψintermediate, space; maxiter=maxiter, χenv = χenv, tol = tol, verbosity = verbosity)
            end
            return ψintermediate
        end
        if (verbosity > 0)
            @info "Step $i of $maxiter: norm difference in A is $ϵ"
            @info "Step $i of $maxiter: norm difference in Ws is $ϵ_Ws"
        end
    end
    @info "Not converged after $maxiter iterations: norm difference in A is $ϵ"
    ψnew = rotl90(A)
    ψintermediate = ψnew * norm(ψ)/norm(ψnew)
    for space = spaces
        ψintermediate = approximate_iteratively(ψintermediate, space; maxiter=maxiter, χenv = χenv, tol = tol, verbosity = verbosity)
    end
    return ψintermediate
end

function approximate_iteratively(
    ψ::AbstractTensorMap{S,1,4},
    space::ElementarySpace;
    maxiter=50,
    χenv=12,
    tol=1e-5,
    verbosity=1,
    initial_guess="random",
) where {S}
    @error "In approx iter"
    if (verbosity > 0)
        @info "Approximating from $(ψ.dom[1]) to $(space)"
    end
    Ws = [isdual(ψ.dom[i]) ? TensorMap(randn, ψ.dom[i], space') : TensorMap(randn, ψ.dom[i], space) for i in 1:4]
    A = rotr90(approximate(ψ, Ws))
    ϵ = 0
    ϵ_Ws = 0
    for i in 1:maxiter
        Ws_old = copy(Ws)
        A_old = copy(A)
        for _ in 1:4
            Ws, A = update_isometry(ψ, Ws, χenv; space = space)
            ψ = rotl90(ψ)
            Ws = circshift(Ws, -1)
        end
        ϵ = norm(A - A_old) / norm(A_old)
        ϵ_Ws = ([norm(Ws[i] - Ws_old[i]) / norm(Ws_old[i]) for i = 1:4])
        if ϵ < tol
            @info "Converged after $i iterations: norm difference in A is $ϵ"
            @info "Step $i of $maxiter: norm difference in Ws is $ϵ_Ws"
            ψnew = rotl90(A)
            return ψnew * norm(ψ)/norm(ψnew)
        end
        if (verbosity > 0)
            @info "Step $i of $maxiter: norm difference in A is $ϵ"
        end
    end
    @info "Not converged after $maxiter iterations: norm difference in A is $ϵ"
    ψnew = rotl90(A)
    return ψnew * norm(ψ)/norm(ψnew)
end