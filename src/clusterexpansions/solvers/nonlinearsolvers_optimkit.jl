function contract_tensors_symmetric(A)
    loop = ncon([A, A, A, A], [[-1 -5 4 1], [-2 -6 1 2], [-3 -7 2 3], [-4 -8 3 4]])
    return permute(loop, ((1,2,3,4),(5,6,7,8)))
end

function contract_tensors_symmetric_fused(A)
    loop = ncon([A, A, A, A], [[1 -1 2], [2 -2 3], [3 -3 4], [4 -4 1]])
    return permute(loop, ((1,2),(3,4)))
end

function get_gradient(A)
    return ncon([A, A, A], [[-3 -6 -2 1], [-4 -7 1 2], [-5 -8 2 -1]])
end

function get_gradient_fused(A)
    return ncon([A, A, A], [[-2 -3 1], [1 -4 2], [2 -5 -1]])
end

function my_add!(Y, X, a)
    Y .+= X .* a
    return InfiniteMPS([Y]).AL[1]
end

function construct_PEPO_loop(A, pspace, vspace, trivspace, levels_to_update; symmetry = nothing)
    T = scalartype(A)
    if symmetry == "C4"
        A_SW = zeros(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')        
        A_SW[][:,:,:,:,1,1] = A[]
        A_SE = rotl90_fermionic(A_SW)
        A_NE = rotl90_fermionic(A_SE)
        A_NW = rotl90_fermionic(A_NE)
        As = [A_NW, A_NE, A_SE, A_SW]
    elseif isnothing(symmetry)
        A_NW = zeros(T, pspace ⊗ pspace', trivspace ⊗ vspace ⊗ vspace' ⊗ trivspace')
        A_NE = zeros(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ vspace' ⊗ vspace')
        A_SE = zeros(T, pspace ⊗ pspace', vspace ⊗ trivspace ⊗ trivspace' ⊗ vspace')
        A_SW = zeros(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')
        A_NW[][:,:,1,:,:,1] = A[]
        A_NE[][:,:,1,1,:,:] = A[]
        A_SE[][:,:,:,1,1,:] = A[]
        A_SW[][:,:,:,:,1,1] = A[]
        As = [A_NW, A_NE, A_SE, A_SW]
    else
        error("Symmetry $(symmetry) not implemented")
    end    
    dict = Dict((0, -1, -1, 0) => 1, (0, 0, -1, -1) => 2, (-1, 0, 0, -1) => 3, (-1, -1, 0, 0) => 4)
    values = [dict[key] for key in levels_to_update]
    As = [As[values[1]], As[values[2]], As[values[3]], As[values[4]]]
    return As
end

function get_A_nudge(A, exp_H; c = 0)
    D = contract_tensors_symmetric(A) - exp_H
    g = get_gradient(A)

    x1 = ncon([D, g], [[-1 1 2 3 -2 4 5 6], [-3 -4 1 2 3 4 5 6]], [false true])
    x2 = ncon([D, g], [[1 -1 2 3 4 -2 5 6], [-3 -4 1 2 3 4 5 6]], [false true])
    x3 = ncon([D, g], [[1 2 -1 3 4 5 -2 6], [-3 -4 1 2 3 4 5 6]], [false true])
    x4 = ncon([D, g], [[1 2 3 -1 4 5 6 -2], [-3 -4 1 2 3 4 5 6]], [false true])

    f = (norm(D)/norm(exp_H))^2 + c*norm(A)^2
    ∂f = permute(x1+x2+x3+x4, ((1,2),(3,4))) + c*A
    return f, ∂f
end

function get_A_nudge_leftgauge(A, exp_H; c = 0)
    D = contract_tensors_symmetric_fused(A) - exp_H
    g = get_gradient_fused(A)
    x1 = ncon([D, g], [[-2 1 2 3], [-1 -3 1 2 3]], [false true])

    f = norm(D)^2 + c*norm(A)^2
    grad = permute(x1, ((1,2),(3,))) + c*A

    V = leftnull(A; alg = QRpos())
    @tensor ∂f[-1 -2; -3] := grad[1 2; -3] * conj(V[1 2; 3]) * V[-1 -2; 3]

    @tensor check[-1; -2] := A[1 2; -1] * conj(∂f[1 2; -2])
    @assert norm(check)/(norm(A)*norm(∂f)) < 1e-14 "Gradient is not in the nullspace of A"

    return f, ∂f
end

# unpack the gradient
function fg_base(ℒ, ψ)
    f, g = Zygote.withgradient(ℒ, ψ)
    return f, g[1]
end

function solve_4_loop_optim(RHS, spaces, levels_to_update; verbosity = 1, symmetry = nothing, x0 = nothing, gradtol = 1e-9)
    gradtol = 1e-1
    vspace = spaces(-1)
    trivspace = spaces(0)
    pspace = codomain(RHS)[1]

    # RHS = RHS / norm(RHS)
    D = dim(vspace)
    T = scalartype(RHS)
    
    if isnothing(x0)
        # A = randn(T, vspace ⊗ fuse(pspace, pspace'), vspace)
        A = randn(T, pspace ⊗ pspace', vspace ⊗ vspace')
        # A = InfiniteMPS([A]).AL[1]
    else
        A = x0
    end
    # A = A / norm(A) * (norm(exp_H))^(1/4) * sqrt(10)

    c = 0

    # Define an inner product to deal with complex numbers
    my_inner(x, y1, y2) = real(dot(y1, y2))

    # ℒ = A -> norm(contract_tensors_symmetric(A) - exp_H)
    # fg = ψ -> fg_base(ℒ, ψ)
    cfun = x -> get_A_nudge(x, RHS; c = c)

    A, fx, gx, numfg, normgradhistory = optimize(cfun, A, LBFGS(3; verbosity=verbosity, gradtol = gradtol, maxiter=5); inner=my_inner);

    # Anew = zeros(T, vspace ⊗ pspace ⊗ pspace', vspace)
    # Anew[] = reshape(A[], (D,2,2,D))
    # A = permute(Anew, ((2,3),(1,4)))
    error = norm(contract_tensors_symmetric(A)-RHS) / norm(RHS)
    As = construct_PEPO_loop(A, pspace, vspace, trivspace, levels_to_update; symmetry = symmetry)

    return As, error, nothing, A
end

function solve_4_loop_optim_leftgauge(RHS, vspace, levels_to_update; verbosity = 1, symmetry = nothing, x0 = nothing, gradtol = 1e-9)
    # RHS = RHS / norm(RHS)
    D = dim(vspace)
    T = scalartype(RHS)
    RHS = permute(RHS, ((1,5,2,6),(3,7,4,8)))
    exp_H = zeros(T, fuse(codomain(RHS)[1], codomain(RHS)[2]) ⊗ fuse(codomain(RHS)[3], codomain(RHS)[4]), fuse(domain(RHS)[1], domain(RHS)[2])' ⊗ fuse(domain(RHS)[3], domain(RHS)[4])')
    exp_H[] = reshape(RHS[], (4,4,4,4))

    pspace = ℂ^2
    trivspace = ℂ^1
    A = randn(T, vspace ⊗ fuse(pspace, pspace'), vspace)
    A = InfiniteMPS([A]).AL[1]

    c = 0

    # Define an inner product to deal with complex numbers
    my_inner(x, y1, y2) = real(dot(y1, y2))

    # ℒ = A -> norm(contract_tensors_symmetric(A) - exp_H)
    # fg = ψ -> fg_base(ℒ, ψ)
    cfun = x -> get_A_nudge_leftgauge(x, exp_H; c = c)

    A, fx, gx, numfg, normgradhistory = optimize(cfun, A, LBFGS(; verbosity=verbosity, gradtol = gradtol); inner=my_inner);

    error = norm(contract_tensors_symmetric_fused(A)-exp_H) / norm(exp_H)
    Anew = zeros(T, vspace ⊗ pspace ⊗ pspace', vspace)
    Anew[] = reshape(A[], (D,2,2,D))
    A = permute(Anew, ((2,3),(1,4)))

    As = construct_PEPO_loop(A, pspace, vspace, trivspace, levels_to_update; symmetry = symmetry)
    return As, error, nothing, A, normgradhistory
end
