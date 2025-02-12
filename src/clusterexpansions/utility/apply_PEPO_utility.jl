function make_translationally_invariant(A)
    Anew = TensorMap(zeros, A.codom, A.dom)
    for _ = 1:4
        A = rotl90(A)
        Anew += A
    end
    return Anew / 4
end

function make_translationally_invariant_fermionic(A)
    return flip_arrows(make_translationally_invariant(flip_arrows(A)))
end

function flip_arrows(A::AbstractTensorMap{E,S,1,4} 
    ) where {E,S<:ElementarySpace}
    I₃ = isometry(A.dom[3], (A.dom[3])')
    I₄ = isometry(A.dom[4], (A.dom[4])')
    @tensor A_flipped[-1; -2 -3 -4 -5] := A[-1; -2 -3 1 2] * I₃[1; -4] * I₄[2; -5]
    return A_flipped
end

function flip_arrows(A::AbstractTensorMap{E,S,2,4} 
    ) where {E,S<:ElementarySpace}
    I₃ = isometry(A.dom[3], (A.dom[3])')
    I₄ = isometry(A.dom[4], (A.dom[4])')
    @tensor A_flipped[-1 -2; -3 -4 -5 -6] := A[-1 -2; -3 -4 1 2] * I₃[1; -5] * I₄[2; -6]
    return A_flipped
end

function get_gradient(A, O, W)
    @tensor g[-1; -2 -3 -4 -5 -6] := A[1; 2 4 6 -5] * O[-1 1; 3 5 7 -6] * W[2 3; -2] * W[4 5; -3] * W[6 7; -4]
    return g
end

function get_diff(A, O, W)
    @tensor D[-1; -2 -3 -4 -5] := A[1; 2 4 6 8] * O[-1 1; 3 5 7 9] * W[2 3; -2] * W[4 5; -3] * W[6 7; -4] * W[8 9; -5]
    return D - A
end

function apply_isometry(A, O, W)
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[1; 2 4 6 8] * O[-1 1; 3 5 7 9] * W[2 3; -2] * W[4 5; -3] * W[6 7; -4] * W[8 9; -5]
    return A_trunc
end

function get_W_nudge(A, O, W; c = 0)
    D = get_diff(A, O, W)
    g = get_gradient(A, O, W)
    # W_nudge = ncon([D, g], [[1 2 3 4 -1 -2], [1 2 3 4 -3]], [false, true])
    W_nudge = ncon([D, g], [[1 2 3 4 -3], [1 2 3 4 -1 -2]], [false, true])

    f = norm(D)^2 + c*norm(W)^2
    ∂f = permute(W_nudge, ((1,2),(3,))) + c*W

    return f, ∂f
end

function get_step_size(A, O, W, W_nudge, step_size, linesearch_options::Int)
    αs = step_size .* [10.0^(i) for i = -linesearch_options:linesearch_options]
    errors = [norm(get_diff(A, O, W-α*W_nudge)) for α in αs]
    return αs[argmin(errors)]
end

function find_truncation(A_base, O_base; verbosity = 2, c = 0)
    # A = (A_base) / norm(A_base)
    # O = (O_base) / norm(O_base)
    @assert norm(A_base - rotl90_fermionic(A_base)) / norm(A_base) < 1e-10 "State is not rotationally invariant. Error = $(norm(A_base - rotl90_fermionic(A_base)) / norm(A_base))"
    @assert norm(O_base - rotl90_fermionic(O_base)) / norm(O_base) < 1e-5 "Operator is not rotationally invariant. Error = $(norm(O_base - rotl90_fermionic(O_base)) / norm(O_base))"
    if norm(O_base - rotl90_fermionic(O_base)) / norm(O_base) > 1e-10
        @warn "Operator is not rotationally invariant. Error = $(norm(O_base - rotl90_fermionic(O_base)) / norm(O_base))"
    end
    A = flip_arrows(A_base) / norm(A_base)
    O = flip_arrows(O_base) / norm(O_base)

    # @assert norm(A - rotl90(A)) / norm(A) < 1e-10 "State is not rotationally invariant. Error = $(norm(A - rotl90(A)) / norm(A))"
    # @assert norm(O - rotl90(O)) / norm(O) < 1e-10 "Operator is not rotationally invariant. Error = $(norm(O - rotl90(O)) / norm(O))"

    Aspace = A.dom[1]
    Ospace = O.dom[1]

    W = TensorMap(randn, Aspace ⊗ Ospace, Aspace)
    W = W / norm(W)

    # Define an inner product to deal with complex numbers
    my_inner(x, y1, y2) = real(dot(y1, y2))

    cfun = x -> get_W_nudge(A, O, x; c = c)

    W, fx, gx, numfg, normgradhistory = optimize(cfun, W, LBFGS(; verbosity=verbosity, maxiter = 5000); inner=my_inner);
    W_other = W / norm(W)
    W = W / (norm(O_base))^(1/4)

    U, Σ, V = tsvd(W)
    W = U*V

    Ws = [isdual(A_base.dom[i]) ? TensorMap(zeros, ComplexF64, A_base.dom[i] ⊗ O_base.dom[i], (A_base.dom[1])') : TensorMap(zeros, ComplexF64, A_base.dom[i] ⊗ O_base.dom[i], A_base.dom[1]) for i in 1:4]
    for dir = 1:4
        Ws[dir][] = W[]
    end
    return Ws, flip_arrows(apply_isometry(flip_arrows(A_base), flip_arrows(O_base), W)), sqrt(fx)
end

function find_truncation_GD(A, O; step_size = 1e-3, ϵ = 1e-10, max_iter = 10000, line_search = false, linesearch_options = 1, verbosity = 2)
    verbosity = 3
    A = flip_arrows(A)
    O = flip_arrows(O)

    Aspace = A.dom[1]
    Ospace = O.dom[1]

    println("spaces = $Aspace, $Ospace")
    base_step_size = step_size
    W = TensorMap(randn, Aspace ⊗ Ospace, Aspace)
    errors = [Inf]
    c = 0
    for i = 1:max_iter
        W_nudge, error = get_W_nudge(A, O, W; c = c)
        W_nudge = W_nudge * (norm(W) / norm(W_nudge))
        step_size = base_step_size*error
        if line_search
            step_size = get_step_size(A, A_nudge, exp_H, step_size, linesearch_options)
        end
        W = W - step_size*W_nudge
        if error < ϵ
            if verbosity >= 2
                @info "Converged after $(i) iterations - error = $(error)"
            end
            return W, errors
        end
        if verbosity >= 3
            @info "Iteration $(i) of loop solver: error = $(error) - step size = $(step_size)"
        end
        push!(errors, error)
        if errors[end] > errors[end-1]
            if verbosity >= 1
                @info "Error starting increasing after iteration $(i): Early stopping at error = $(error)"
            end
            return W, errors
        end
    end
    error = norm(get_diff(A, O, W))/norm(A)
    if verbosity >= 1
        @warn "Not converged after $(max_iter) iterations - error = $(error)"
    end
    return W, errors
end