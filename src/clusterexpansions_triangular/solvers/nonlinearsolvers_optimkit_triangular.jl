function check_3_loop(lattice::Triangular, Δ, As, RHS, spaces)
    trivspace = spaces(0)
    if Δ
        conjugated = [4 7 8 10 11 12]
        tens = [(c ∈ conjugated) ? Tensor(ones, trivspace') : Tensor(ones, trivspace) for c = 1:12]
    
        @tensor RHS_check[-1 -2 -3; -4 -5 -6] := As[1][-3 -6; 1 2 3 13 15 4] * As[2][-2 -5; 13 5 6 7 8 14] * As[3][-1 -4; 9 15 14 10 11 12] * 
        tens[1][1] * tens[2][2] * tens[3][3] * tens[4][4] * tens[5][5] * tens[6][6] * 
        tens[7][7] * tens[8][8] * tens[9][9] * tens[10][10] * tens[11][11] * tens[12][12]    
    else
        conjugated = [3 4 8 10 11 12]
        tens = [(c ∈ conjugated) ? Tensor(ones, trivspace') : Tensor(ones, trivspace) for c = 1:12]
        @tensor RHS_check[-1 -2 -3; -4 -5 -6] := As[1][-3 -6; 1 2 13 15 3 4] * As[2][-2 -5; 5 6 7 8 14 13] * As[3][-1 -4; 15 14 9 10 11 12] * 
        tens[1][1] * tens[2][2] * tens[3][3] * tens[4][4] * tens[5][5] * tens[6][6] * 
        tens[7][7] * tens[8][8] * tens[9][9] * tens[10][10] * tens[11][11] * tens[12][12]    
    end
    return norm(RHS_check - RHS) / norm(RHS)
end

function construct_PEPO_3_loop(A_N) # A_NW
    # project A on space that will eventually be hermitian
    A_perm = permute(A_N, ((1,3,4,5),(2,8,7,6)))
    A_sym = 0.5 * (A_perm + A_perm')
    A_N_C6v = permute(A_sym, ((1,5),(2,3,4,8,7,6)))

    A_SW = rotl60_fermionic(rotl60_fermionic(A_N_C6v)) # A_S
    A_SE = rotl60_fermionic(rotl60_fermionic(A_SW)) # A_NE
    return [A_N, A_SE, A_SW] # [A_NW, A_NE, A_S]
end 

function solve_3_loop_optim(lattice, RHS, spaces, levels_to_update; verbosity = 1, symmetry = nothing, gradtol = 1e-9)
    println("Solving 3-loop cluster with optimization..., levels to update = $(levels_to_update)")
    Δ = (-1, 0, 0, 0, 0, -1) ∈ levels_to_update # only alternative is ∇
    T = scalartype(RHS)
    vspace = spaces(-1)
    trivspace = spaces(0)
    pspace = codomain(RHS)[1]
    
    opt_alg = LBFGS(; maxiter=2000, gradtol=gradtol, verbosity)

    if isnothing(symmetry)
        if Δ
            A_N = randn(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace ⊗ vspace' ⊗ vspace' ⊗ trivspace')
            A_SE = randn(T, pspace ⊗ pspace', vspace ⊗ trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace' ⊗ vspace')
            A_SW = randn(T, pspace ⊗ pspace', trivspace ⊗ vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace' ⊗ trivspace')
            A_N *= norm(RHS)^(1/4) / norm(A_N)
            A_SE *= norm(RHS)^(1/4) / norm(A_SE)
            A_SW *= norm(RHS)^(1/4) / norm(A_SW)
            As = [A_N, A_SE, A_SW]
        else
            A_NW = randn(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ vspace ⊗ vspace' ⊗ trivspace' ⊗ trivspace')
            A_NE = randn(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace ⊗ trivspace' ⊗ vspace' ⊗ vspace')
            A_S = randn(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace' ⊗ trivspace')
            A_NW *= norm(RHS)^(1/4) / norm(A_NW)
            A_NE *= norm(RHS)^(1/4) / norm(A_NE)
            A_S *= norm(RHS)^(1/4) / norm(A_S)
            As = [A_NW, A_NE, A_S]
        end
        custom_costfun = ψ -> check_3_loop(lattice, Δ, ψ, RHS, spaces)

        # optimize free energy per site
        As_final, f, = optimize(
            As,
            opt_alg;
            inner=PEPSKit.real_inner,
        ) do psi
            E, gs = withgradient(psi) do ψ
                return custom_costfun(ψ)
            end
            g = only(gs)
            return E, g
        end
    elseif symmetry == "C6"
        if Δ
            A = randn(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace ⊗ vspace' ⊗ vspace' ⊗ trivspace') # this is A_N
        else
            A = randn(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ vspace ⊗ vspace' ⊗ trivspace' ⊗ trivspace') # this is A_NW
        end
        A *= norm(RHS)^(1/4) / norm(A)

        custom_costfun = ψ -> check_3_loop(lattice, Δ, construct_PEPO_3_loop(ψ), RHS, spaces)

        # optimize free energy per site
        A_final, f, = optimize(
            A,
            opt_alg;
            inner=PEPSKit.real_inner,
        ) do psi
            E, gs = withgradient(psi) do ψ
                return custom_costfun(ψ)
            end
            g = only(gs)
            return E, g
        end
        As_final = construct_PEPO_3_loop(A_final)
    else
        @error "Symmetry $(symmetry) not implemented"
    end

    if Δ
        dict = Dict((0, 0, 0, -1, -1, 0) => 1, (-1, 0, 0, 0, 0, -1) => 2, (0, -1, -1, 0, 0, 0) => 3)
    else
        dict = Dict((0, 0, -1, -1, 0, 0) => 1, (0, 0, 0, 0, -1, -1) => 2, (-1, -1, 0, 0, 0, 0) => 3)
    end
    values = [dict[key] for key in levels_to_update]
    As = [As_final[values[1]], As_final[values[2]], As_final[values[3]]]
    return As, f
end
