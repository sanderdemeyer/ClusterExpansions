function contract_tensors_symmetric(A)
    flipper = isometry(domain(A)[1], codomain(A)[1])
    println(norm(A))
    @tensor loop[-1 -2 -3 -4; -5 -6 -7 -8] := A[1 -4; 2 -8] * flipper[2; 3] * A[3 -3; 4 -7] * flipper[4; 5] *  A[5 -2; 6 -6] * flipper[6; 7] * A[7 -1; 8 -5] * flipper[8; 1]
    @tensor loop[-1 -2 -3 -4; -5 -6 -7 -8] := twist(A,1)[1 -4; 2 -8] * flipper[2; 3] * A[3 -3; 4 -7] * flipper[4; 5] *  A[5 -2; 6 -6] * flipper[6; 7] * A[7 -1; 8 -5] * flipper[8; 1]
    return loop
    A2 = flip(A, [3])
    A3 = twist(flip(A, [1 3]), 1)
    A4 = flip(A, 1)
    # A2 = copy(A)
    # A3 = copy(A)
    # A4 = copy(A)

    @tensor loop[-1 -2 -3 -4; -5 -6 -7 -8] := A[3 -4 4 -8] * A2[4 -3 1 -7] * A3[1 -2 2 -6] * A4[2 -1 3 -5]
    return loop
    # loop = ncon([A, A2, A3, A4], [[3 -4 4 -8], [4 -3 1 -7], [1 -2 2 -6], [2 -1 3 -5]])

    # A2 = flip(A, 4)
    # A3 = twist(flip(A, [3 4]), 1)
    # A4 = flip(A, 3)
    # loop = ncon([A, A2, A3, A4], [[-4 -8 4 1], [-3 -7 1 2], [-2 -6 2 3], [-1 -5 3 4]])
    return permute(loop, ((1,2,3,4),(5,6,7,8)))
end

function contract_tensors_symmetric_fused(A)
    loop = ncon([A, A, A, A], [[1 -1 2], [2 -2 3], [3 -3 4], [4 -4 1]])
    return permute(loop, ((1,2),(3,4)))
end

function get_gradient(A)
    A2 = flip(A, 3)
    A3 = twist(flip(A, [1 3]), 1)
    A4 = flip(A, 1)
    # g = ncon([A4 A2 A3], [[-7 -1 1 -4], [1 -2 2 -5], [2 -3 -8 -6]])
    g = ncon([A2, A3, A4], [[-8 -3 1 -6], [1 -2 2 -5], [2 -1 -7 -4]])

    # A2 = flip(A, 4)
    # A3 = twist(flip(A, [3 4]), 1)
    # A4 = flip(A, 3)
    # g = ncon([A4 A2 A3], [[-1 -4 -7 1], [-2 -5 1 2], [-3 -6 2 -8]])
    return permute(g, ((1,2,3,4,5,6),(7,8)))

    # return ncon([A, A, A], [[-3 -6 -2 1], [-4 -7 1 2], [-5 -8 2 -1]])
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
        A = permute(A, ((2,4),(1,3)))
        Isom1 = isometry(domain(A)[1], domain(A)[1] ⊗ trivspace')
        Isom2 = isometry(domain(A)[2], trivspace ⊗ domain(A)[2])
        @tensor A_NW[-1 -2; -3 -4 -5 -6] := A[-1 -2; 1 2] * Isom1[1; -5 -6] * Isom2[2; -3 -4]
        # A_SW = zeros(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')        
        # A_SW[][:,:,:,:,1,1] = A[]
        A_SW = rotl90_fermionic(A_NW)
        A_SE = rotl90_fermionic(A_SW)
        A_NE = rotl90_fermionic(A_SE)
        As = [A_NW, A_NE, A_SE, A_SW]
        println("Summary A_SW = $(summary(A_SW))")
        println("Summary A_NW = $(summary(A_NW))")
        println("Summary A_SE = $(summary(A_SE))")
        println("Summary A_NE = $(summary(A_NE))")
    elseif isnothing(symmetry)
        @error "TBA"
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
    return [A_NW, A_NE, A_SE, A_SW], As
end

function get_A_nudge(A, exp_H; c = 0)
    D = contract_tensors_symmetric(A) - exp_H
    D_permuted = permute(D, ((1,5,2,6,3,7,4,8),()))
    g = get_gradient(A)
    g_permuted = permute(g, ((1,4,2,5,3,6),(7,8)))
    # g_permuted = twist(g_permuted, 8)
    # D_permuted = twist(D_permuted, [2 4 6 8])

    x = ncon([D_permuted, g_permuted], [[1 2 3 4 5 6 -2 -4], [1 2 3 4 5 6 -1 -3]], [false true])
    ∂f = permute(x, ((1,2),(3,4)))

    x_conj = ncon([g_permuted, D_permuted], [[1 2 3 4 5 6 -1 -3], [1 2 3 4 5 6 -2 -4]], [true false])
    # x = twist(x, 3)

    ∂f = permute(x_conj, ((1,2),(3,4)))

    # x1 = ncon([D, g], [[-1 1 2 3 -2 4 5 6], [-3 -4 1 2 3 4 5 6]], [false true])
    # x2 = ncon([D, g], [[1 -1 2 3 4 -2 5 6], [-3 -4 1 2 3 4 5 6]], [false true])
    # x3 = ncon([D, g], [[1 2 -1 3 4 5 -2 6], [-3 -4 1 2 3 4 5 6]], [false true])
    # x4 = ncon([D, g], [[1 2 3 -1 4 5 6 -2], [-3 -4 1 2 3 4 5 6]], [false true])

    fbefore = norm(D)^2# /norm(exp_H))^2
    f = ncon([D, D], [[1 2 3 4 5 6 7 8], [1 2 3 4 5 6 7 8]], [false true])
    # ∂f = permute(x1+x2+x3+x4, ((1,2),(3,4))) + c*A



    # A2 = flip(A, 3)
    # A3 = twist(flip(A, [1 3]), 1)
    # A4 = flip(A, 1)

    A2 = copy(A)
    A3 = copy(A)
    A4 = copy(A)
    loop = ncon([A, A2, A3, A4], [[3 -4 4 -8], [4 -3 1 -7], [1 -2 2 -6], [2 -1 3 -5]])

    # f = ncon([A, A2, A3, A4, loop], [[3 8 4 12], [4 7 1 11], [1 6 2 10], [2 5 3 9], [5 6 7 8 9 10 11 12]], [false false false false true])
    # ∂f1 = ncon([A2, A3, A4, loop], [[-3 7 1 11], [1 6 2 10], [2 5 -1 9], [5 6 7 -2 9 10 11 -4]], [false false false true])
    # ∂f2 = ncon([A, A3, A4, loop], [[3 8 -1 12], [-3 6 2 10], [2 5 3 9], [5 6 -2 8 9 10 -4 12]], [false false false true])
    # ∂f3 = ncon([A, A2, A4, loop], [[3 8 4 12], [4 7 -1 11], [-3 5 3 9], [5 -2 7 8 9 -4 11 12]], [false false false true])
    # ∂f4 = ncon([A, A2, A3, loop], [[-3 8 4 12], [4 7 1 11], [1 6 -1 10], [-2 6 7 8 -4 10 11 12]], [false false false true])

    # ∂f = permute(∂f1+∂f2+∂f3+∂f4, ((3,4),(1,2)))

    loop_twisted = twist(loop, [5 6 7 8])
    f = ncon([loop loop_twisted], [[1 2 3 4 5 6 7 8], [1 2 3 4 5 6 7 8]], [false true])
    println("f = $f")

    return real(f)
    f = ncon([A, A2, A3, A4, loop], [[3 8 4 12], [4 7 1 11], [1 6 2 10], [2 5 3 9], [5 6 7 8 9 10 11 12]], [true true true true false])
    # ∂f1 = ncon([A2, A3, A4, loop], [[-3 7 1 11], [1 6 2 10], [2 5 -1 9], [5 6 7 -2 9 10 11 -4]], [true true true false])
    # ∂f2 = ncon([A, A3, A4, loop], [[3 8 -1 12], [-3 6 2 10], [2 5 3 9], [5 6 -2 8 9 10 -4 12]], [true true true false])
    # ∂f3 = ncon([A, A2, A4, loop], [[3 8 4 12], [4 7 -1 11], [-3 5 3 9], [5 -2 7 8 9 -4 11 12]], [true true true false])
    # ∂f4 = ncon([A, A2, A3, loop], [[-3 8 4 12], [4 7 1 11], [1 6 -1 10], [-2 6 7 8 -4 10 11 12]], [true true true false])

    # ∂f = permute(∂f1+∂f2+∂f3+∂f4, ((1,2),(3,4)))

    # f = ncon([A, A2, A3, A4, loop], [[3 8 4 12], [4 7 1 11], [1 6 2 10], [2 5 3 9], [5 6 7 8 9 10 11 12]], [true true true true false])
    # ∂f = ncon([A2, A3, A4, loop], [[-3 7 1 11], [1 6 2 10], [2 5 -1 9], [5 6 7 -2 9 10 11 -4]], [true true true false])
    # ∂f = permute(∂f, ((1,2),(3,4)))
    # println("∂f = $(summary(∂f))")

    return real(f), ∂f
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

function check_loop(As, RHS, spaces)
    trivspace = spaces(0)
    # conj = Bool[0 1 0 0 0 1 1 1]
    conjugated = [2 6 7 8]
    tens = [(c ∈ conjugated) ? Tensor(ones, trivspace') : Tensor(ones, trivspace) for c = 1:8]
    # @tensor RHS_check[-1 -2 -3 -4; -5 -6 -7 -8] := As[1][-4 -8; 5 1 4 6] * As[2][-3 -7; 7 8 2 1] * As[3][-2 -6; 2 9 10 3] * As[4][-1 -5; 4 3 11 12] * tens[1][5] * tens[2][6] * tens[3][7] * tens[4][8] * tens[5][9] * tens[6][10] * tens[7][11] * tens[8][12]
    @tensor RHS_check[-1 -2 -3 -4; -5 -6 -7 -8] := As[1][-4 -8; 1 9 12 2] * As[2][-3 -7; 3 4 10 9] * As[3][-2 -6; 10 5 6 11] * As[4][-1 -5; 12 11 7 8] * tens[1][1] * tens[2][2] * tens[3][3] * tens[4][4] * tens[5][5] * tens[6][6] * tens[7][7] * tens[8][8]
    return norm(RHS_check - RHS) / norm(RHS)
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
        # A = randn(T, pspace ⊗ pspace', vspace' ⊗ vspace)
        A = randn(T, vspace ⊗ pspace, vspace' ⊗ pspace)
        A *= norm(RHS)^(1/4) / norm(A)

        A_NW = randn(T, pspace ⊗ pspace', trivspace ⊗ vspace ⊗ vspace' ⊗ trivspace')
        A_NE = randn(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ vspace' ⊗ vspace')
        A_SE = randn(T, pspace ⊗ pspace', vspace ⊗ trivspace ⊗ trivspace' ⊗ vspace')
        A_SW = randn(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')
        A_NW *= norm(RHS)^(1/4) / norm(A_NW)
        A_NE *= norm(RHS)^(1/4) / norm(A_NE)
        A_SE *= norm(RHS)^(1/4) / norm(A_SE)
        A_SW *= norm(RHS)^(1/4) / norm(A_SW)
        As = [A_NW, A_NE, A_SE, A_SW]
    else
        A = x0
    end
    # A = A / norm(A) * (norm(exp_H))^(1/4) * sqrt(10)

    c = 0

    # Define an inner product to deal with complex numbers
    my_inner(x, y1, y2) = real(dot(y1, y2))

    # ℒ = A -> norm(contract_tensors_symmetric(A) - exp_H)
    # fg = ψ -> fg_base(ℒ, ψ)

    # RHS = RHS / norm(RHS)
    # exp_H = TensorMap(convert(Array{ComplexF64}, RHS.data), codomain(RHS), domain(RHS))

    # cfun = x -> get_A_nudge(x, exp_H; c = 0.0)

    println("Norm of RHS = $(norm(RHS))")
    # result = optimtest(cfun, A; inner = my_inner)
    # println("result = $result")
    # alphas, fs, dfs1, dfs2 = result
    # println("alphas = $alphas")
    # println("fs = $fs")
    # println("dfs1 = $(sort(real.(dfs1)))")
    # println("dfs2 = $(sort(real.(dfs2)))")

    RHS_rot = permute(RHS, ((2,3,4,1),(6,7,8,5)))
    println("Rot inv before: $(norm(permute(RHS, ((2,3,4,1),(6,7,8,5)))-RHS)/norm(RHS))")
    RHS = (RHS + permute(RHS, ((2,3,4,1),(6,7,8,5))) + permute(RHS, ((3,4,1,2),(7,8,5,6))) + permute(RHS, ((4,1,2,3),(8,5,6,7)))) / 4
    println("Rot inv after: $(norm(permute(RHS, ((2,3,4,1),(6,7,8,5)))-RHS)/norm(RHS))")

    # A, fx, gx, numfg, normgradhistory = optimize(cfun, A, GradientDescent(; verbosity=verbosity, gradtol = gradtol, maxiter=4); inner=my_inner);

    opt_alg = LBFGS(; maxiter=500, gradtol=1e-9, verbosity)

    custom_costfun = ψ -> check_loop(ψ, RHS, spaces)
    # optimize free energy per site
    As_final, f, = optimize(
        As,
        opt_alg;
        inner=PEPSKit.real_inner,
    ) do psi
        E, gs = withgradient(psi) do ψ
            return custom_costfun(ψ)
            # return norm(contract_tensors_symmetric(ψ) - RHS)/norm(RHS)
        end
        g = only(gs)
        return E, g
    end
    dict = Dict((0, -1, -1, 0) => 1, (0, 0, -1, -1) => 2, (-1, 0, 0, -1) => 3, (-1, -1, 0, 0) => 4)
    values = [dict[key] for key in levels_to_update]
    As = [As[values[1]], As[values[2]], As[values[3]], As[values[4]]]
    return As, f, nothing, As[1]

    println("After Zygote optim, ψ_final = $(summary(As_final)) and  f = $f")

    @warn "Norm not yet included"
    # Anew = zeros(T, vspace ⊗ pspace ⊗ pspace', vspace)
    # Anew[] = reshape(A[], (D,2,2,D))
    # A = permute(Anew, ((2,3),(1,4)))
    error_orig = norm(contract_tensors_symmetric(A_final)-RHS)^2
    error = norm(contract_tensors_symmetric(A_final)-RHS) / norm(RHS)
    println("Error on loop is $(error), with original f = $(error_orig)")
    As_test, As = construct_PEPO_loop(A_final, pspace, vspace, trivspace, levels_to_update; symmetry = symmetry)

    conj = Bool[0 1 0 0 0 1 1 1]
    tens = [c ? Tensor(ones, trivspace') : Tensor(ones, trivspace) for c in conj]
    for dir = 1:4
        println("dir = $dir, As[$dir] = $(summary(As_test[dir]))")
    end

    # for perm = reverse(collect(Combinatorics.permutations([1,2,3,4])))
    #     (a1,a2,a3,a4,a5,a6,a7,a8) = (-perm[1], -perm[1]-4, -perm[2], -perm[2]-4, -perm[3], -perm[3]-4, -perm[4], -perm[4]-4)
    #     # @tensor RHS_check[a1 a3 a5 a7; a2 a4 a6 a8] := As_test[1][a1 a2; 5 1 4 6] * As_test[2][a3 a4; 7 8 2 1] * As_test[3][a5 a6; 2 9 10 3] * As_test[4][a7 a8; 4 3 11 12] * tens[1][5] * tens[2][6] * tens[3][7] * tens[4][8] * tens[5][9] * tens[6][10] * tens[7][11] * tens[8][12]
    #     # @tensor RHS_check[-1 -2 -3 -4; -5 -6 -7 -8] := As_test[1][a1 a2; 5 1 4 6] * As_test[2][a3 a4; 7 8 2 1] * As_test[3][a5 a6; 2 9 10 3] * As_test[4][a7 a8; 4 3 11 12] * tens[1][5] * tens[2][6] * tens[3][7] * tens[4][8] * tens[5][9] * tens[6][10] * tens[7][11] * tens[8][12]
    #     RHS_check = ncon([As_test[1] As_test[2] As_test[3] As_test[4] tens...], [[a1 a2 5 1 4 6], [a3 a4 7 8 2 1], [a5 a6 2 9 10 3], [a7 a8 4 3 11 12], [[i] for i = 5:12]...])
    #     RHS_check = permute(RHS_check, ((1,2,3,4),(5,6,7,8)))
    #     println("For perm = $(perm), error on loop is $(norm(RHS-RHS_check)/norm(RHS))")
    # end
    @tensor RHS_check[-1 -2 -3 -4; -5 -6 -7 -8] := As_test[1][-4 -8; 5 1 4 6] * As_test[2][-3 -7; 7 8 2 1] * As_test[3][-2 -6; 2 9 10 3] * As_test[4][-1 -5; 4 3 11 12] * tens[1][5] * tens[2][6] * tens[3][7] * tens[4][8] * tens[5][9] * tens[6][10] * tens[7][11] * tens[8][12]
    println("Error on loop is $(norm(RHS-RHS_check)/norm(RHS))")
    @tensor RHS_check[-1 -2 -3 -4; -5 -6 -7 -8] := As_test[1][-2 -6; 5 1 4 6] * As_test[2][-1 -5; 7 8 2 1] * As_test[3][-3 -7; 2 9 10 3] * As_test[4][-4 -8; 4 3 11 12] * tens[1][5] * tens[2][6] * tens[3][7] * tens[4][8] * tens[5][9] * tens[6][10] * tens[7][11] * tens[8][12]
    println("Error on loop is $(norm(RHS-RHS_check)/norm(RHS))")
    println("Norm of RHS = $(norm(RHS))")
    println("Error on loop is $(norm(RHS-RHS_check)/norm(RHS))")

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
