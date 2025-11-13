using ClusterExpansions
using TensorKit
using JLD2

function rotl60_pf(T::TensorMap{A, S, 3, 3}) where {A, S}
    return permute(T, ((4, 1, 2), (5, 6, 3)))
end

function rotl60_pf(T::TensorMap{A, S, 0, 6}) where {A, S}
    return permute(T, ((), (2, 3, 4, 5, 6, 1)))
end

function evolution_operator_triangular(ce_alg::ClusterExpansion, β::Number; T_conv = ComplexF64, canoc_alg::Union{Nothing,Canonicalization} = nothing)
    if β == 0.0
        pspace = domain(ce_alg.onesite_op)[1]
        vspace = ce_alg.spaces(0)
        t = id(T_conv, pspace ⊗ vspace ⊗ vspace ⊗ vspace)
        return permute(t, ((1,5),(6,7,8,2,3,4)))
    end
    lattice = ClusterExpansions.Triangular()
    _, O_clust_full = clusterexpansion(lattice, ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
    O_clust_full = convert(TensorMap, O_clust_full)
    O_canoc = canonicalize(O_clust_full, canoc_alg)
    O = zeros(T_conv, codomain(O_canoc), domain(O_canoc))
    for (f_full, f_conv) in zip(blocks(O_canoc), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O
end

J = 1.0
g = 0.0
z = 0.0

χenv = 14

ce_alg = ising_operators(-J, g, z; p = 3, symmetry = "C6", verbosity = 3, loop_space = ℂ^4, solving_loops = true, T = BigFloat)
M = TensorMap(ComplexF64[1.0 0.0; 0.0 -1.0], ℂ^2, ℂ^2)

β = 0.1
lattice = ClusterExpansions.Triangular()
O_clust, O_clust_full = clusterexpansion(lattice, ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)

for i1 = 1:3
    for i2 = 1:3
        for i3 = 1:3
            for i4 = 1:3
                for i5 = 1:3
                    for i6 = 1:3
                        key = (1,1,i1,i2,i3,i4,i5,i6)
                        dual_key = (1,1,i4,i5,i6,i1,i2,i3)
                        @tensor T[-6 -5 -4; -3 -2 -1] := twist(O_clust_full[key...], 2)[1 1; -1 -2 -3 -4 -5 -6]
                        @tensor T_dual[-6 -5 -4; -3 -2 -1] := twist(O_clust_full[dual_key...], 2)[1 1; -1 -2 -3 -4 -5 -6]
                        normdiff = norm(T - T_dual')
                        if normdiff > 1e-70
                            println("for key = $(key), norm diff = $(normdiff)")
                        end
                        # @assert O_clust_full[i1,i2,i3;j1,j2,j3] ≈ O_clust[i1,i2,i3;j1,j2,j3] "Entries do not match at ($(i1,i2,i3;j1,j2,j3))"
                    end
                end
            end
        end
    end
end
# key = (1,1,1,1,1,1,3,3)
# dual_key = (1,1,1,3,3,1,1,1)
# @tensor T[-6 -5 -4; -3 -2 -1] := twist(O_clust_full[key...], 2)[1 1; -1 -2 -3 -4 -5 -6]
# @tensor T_dual[-6 -5 -4; -3 -2 -1] := twist(O_clust_full[dual_key...], 2)[1 1; -1 -2 -3 -4 -5 -6]
# normdiff = norm(T - T_dual')

Onew = convert(TensorMap, O_clust_full);
@tensor Tnew[-6 -5 -4; -1 -2 -3] := twist(Onew, 2)[1 1; -1 -2 -3 -4 -5 -6];
println(" hermitian. Error = $(norm(Tnew - Tnew'))")

norm(convert(Array, Tnew) - convert(Array, Tnew'))
for i1 = 1:7
    for i2 = 1:7
        for i3 = 1:7
            for i4 = 1:7
                for i5 = 1:7
                    for i6 = 1:7
                        ϵ = norm(convert(Array, Tnew)[i1,i2,i3,i4,i5,i6] - convert(Array, Tnew')[i1,i2,i3,i4,i5,i6])
                        if ϵ > 1e-70
                            println("key = ($([i1,i2,i3,i4,i5,i6])), error = $ϵ")
                            println("values: $(convert(Array, Tnew)[i1,i2,i3,i4,i5,i6]) and $(convert(Array, Tnew')[i1,i2,i3,i4,i5,i6])")
                        end
                    end
                end
            end
        end
    end
end
                    

println(done)
β = 0.1
O = evolution_operator_triangular(ce_alg, β);

@tensor T_flipped[-6 -5 -4; -1 -2 -3] := twist(O, 2)[1 1; -1 -2 -3 -4 -5 -6]
T_unflipped = permute(flip(T_flipped, (1, 2, 3); inv = true), ((), (4, 5, 6, 3, 2, 1)))
println(" hermitian. Error = $(norm(T_flipped - T_flipped'))")
println(" C6 symmetric. Error = $(norm(T_unflipped - rotl60_pf(T_unflipped)))")


println(done)
βs = LinRange(0.1, 0.1, 1)
ms = []
Os = []
for β = βs
    println("β = $(β): ")
    O = evolution_operator_triangular(ce_alg, β)
    push!(Os, O)
    # println(typeof(O))
    # push!(ms, expectation_value_triangular(O, M, χenv))
end

file = jldopen("triangular_expval_test_minus_p_4_wo_loops.jld2", "w")
file["βs"] = βs
file["Os"] = Os
close(file)
