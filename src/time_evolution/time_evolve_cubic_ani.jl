function construct_tensors_ani(tensors, TS)
    @assert allequal([scalartype(tens) for tens in tensors])
    @assert allequal([domain(tens) for tens in tensors])
    
    VS = domain(tensors[1])
    F1 = isomorphism(VS, TS ⊗ VS[1] ⊗ VS[2] ⊗ TS' ⊗ VS[3] ⊗ VS[4])
    F2 = isomorphism(VS, VS[1] ⊗ TS ⊗ VS[2] ⊗ VS[3] ⊗ TS' ⊗ VS[4])
    F3 = isomorphism(VS, VS[1] ⊗ VS[2] ⊗ TS ⊗ VS[3] ⊗ VS[4] ⊗ TS')

    tens1 = tensors[1] * F1
    tens2 = tensors[2] * F2
    tens3 = tensors[3] * F3
    
    return [tens1, tens2, tens3]
end

function construct_cubic_ani_CE(O_2Ds)
    keys_3D = []
    tensors_3D = []

    @assert allequal([domain(O_2D[(0,0,0,0)])[1] for O_2D in O_2Ds])

    TS = domain(O_2Ds[1][(0,0,0,0)])[1]
    for (key,tens_1) in O_2Ds[1]
        tens_2 = O_2Ds[2][key]
        tens_3 = O_2Ds[3][key]

        push!(keys_3D, construct_levels(key)...)
        tensors = (tens_1,tens_2,tens_3)
        println("ok")
        push!(tensors_3D, construct_tensors_ani(tensors, TS)...)
    end
    return Dict(zip(keys_3D, tensors_3D))
end

function evolution_operator_cubic_anisotropic(ce_algs::NTuple{3,ClusterExpansion}, β::Number; T_conv = ComplexF64, canoc_alg::Union{Nothing,Canonicalization} = nothing)
    @assert allequal([ce_alg.onesite_op for ce_alg in ce_algs])
    # @assert allequal([ce_alg.spaces(0) for ce_alg in ce_algs])
    @assert allequal([ce_alg.T for ce_alg in ce_algs])
    @assert allequal([ce_alg.p for ce_alg in ce_algs])
    for i = 0:ce_algs[1].p
        @assert allequal([ce_alg.spaces(i) for ce_alg in ce_algs]) # in principle only necessary for i = 0
    end
    pspace = domain(ce_algs[1].onesite_op)[1]
    if β == 0.0
        vspace = ce_algs[1].spaces(0)
        t = id(T_conv, pspace ⊗ vspace ⊗ vspace ⊗ vspace)
        return permute(t, ((1,5),(6,7,8,2,3,4)))
    end
    PEPO_2Ds = [
                clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; 
                nn_term = ce_alg.nn_term, 
                spaces = ce_alg.spaces, 
                verbosity = ce_alg.verbosity, 
                symmetry = ce_alg.symmetry, 
                solving_loops = ce_alg.solving_loops, 
                svd = ce_alg.svd)[1] 
                for ce_alg in ce_algs
                ]
    PEPO_3D = construct_cubic_ani_CE(PEPO_2Ds)
    return PEPO_3D
    O_clust_full = get_PEPO_cubic(ce_algs[1].T, pspace, PEPO_3D, ce_algs[1].spaces)

    O_clust_full_tm = convert(TensorMap, O_clust_full)
    O_canoc = canonicalize(O_clust_full_tm, canoc_alg)
    O = zeros(T_conv, codomain(O_canoc), domain(O_canoc))
    for (f_full, f_conv) in zip(blocks(O_canoc), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O # Don't normalize, otherwise Atsushi will be mad.
end
