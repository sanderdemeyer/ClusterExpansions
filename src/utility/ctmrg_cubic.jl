abstract type CTMRG3D_alg end

"""
1 O
6 A
12 E
8 C
tot = 27
"""

# corners: UP, NORTH, EAST. All arrows outgoing
# edges: UP, NORTH, DOWN, EAST. All arrows ingoing
# As: UP, NORTH, EAST, SOUTH, WEST. All arrows outgoing
# tensor O: UP, NORTH, EAST, DOWN, SOUTH, WEST. All arrows ingoing

function leading_boundary(env::CTMRG3D_alg)
    S_old = id(envspace(env))
    ϵs = []
    for i = 1:env.maxiter
        S_new = ctmrgstep(env)
        push!(ϵs, norm(S_new - S_old))
        if ϵs[end] < env.tol
            if env.verbosity > 1
                @info "CTMRG3D converged after $i iterations with ϵ = $(ϵs[end])"
            end
            return env, ϵs
        end
        if env.verbosity > 2
            @info "CTMRG3D step $i, ϵ = $(ϵs[end])"
        end
        S_old = copy(S_new)
    end
    if env.verbosity > 0
        @warn "CTMRG3D not converged after $(env.maxiter) iterations. ϵ = $(ϵs[end])"
    end
    return env, ϵs
end

function MPSKit.expectation_value(PEPO::AbstractTensorMap{A,S,2,6}, obss, bpepsspace, envspace; ctm_alg = CTMRG3D, maxiter = 150, tol = 1e-8, verbosity = 0) where {A,S}
    @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
    O = flip(permute(O, ((),(1,2,3,4,5,6))), [4 5 6])
    
    trunc_bpeps = truncdim(dim(bpepsspace))
    trunc_env = truncdim(dim(envspace))
    env = ctm_alg(O; trunc_env, trunc_bpeps, Vbp = bpepsspace, Vv = envspace, maxiter, tol, verbosity)
    env, ϵs = leading_boundary(env);
    println("A = $(env.A.space)")
    println("E = $(env.E.space)")
    println("C = $(env.C.space)")
    expvals = []
    for obs = obss
        if obs == :spectrum
            TM = transfer_matrix(env)
            x0 = rand(scalartype(TM), domain(TM), ℂ^1)
            eigvals, eigvecs = eigsolve(x -> TM * x, x0, 5)
            m = marek_gap(eigvals)
        else
            @tensor M[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 2; -1 -2 -3 -4 -5 -6] * obs[2; 1]
            M = flip(permute(M, ((),(1,2,3,4,5,6))), [4 5 6])
            m = contract_onesite(env, M) / contract_onesite(env, O)
        end
        push!(expvals, m)
    end
    return expvals
end
