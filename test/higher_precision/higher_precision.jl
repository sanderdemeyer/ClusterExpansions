using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Test
using Random
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare

using LinearAlgebra: exp! as la_exp
using LinearAlgebra
using GenericLinearAlgebra: svd as generic_svd
using GenericLinearAlgebra: eigen as generic_eigen
using GenericLinearAlgebra: Diagonal
using GenericLinearAlgebra

setprecision(128)

# function _compute_svddata!(t::TensorMap, alg::Union{SVD,SDD})
#     InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:tsvd!)
#     I = sectortype(t)
#     dims = SectorDict{I,Int}()
#     generator = Base.Iterators.map(blocks(t)) do (c, b)
#         U, Σ, V = svd!(b, alg)
#         dims[c] = length(Σ)
#         return c => (U, Σ, V)
#     end
#     SVDdata = SectorDict(generator)
#     return SVDdata, dims
# end

# function svd!(A::StridedMatrix{T}, alg::Union{SVD,SDD}) where {T<:BigFloat}
#     # fix another type instability in LAPACK wrappers
#     println("Here in new svd")
#     TT = Tuple{Matrix{T},Vector{real(T)},Matrix{T}}
#     U, S, V = alg isa SVD ? LAPACK.gesvd!('S', 'S', A)::TT : LAPACK.gesdd!('S', A)::TT
#     return U, S, V
# end

B = TensorMap(randn, Complex{BigFloat}, ℂ^2 ⊗ ℂ^3, ℂ^2)
A = TensorMap(convert(Vector{ComplexF64},B.data), ℂ^2 ⊗ ℂ^3, ℂ^2)

@tensor AA[-1 -2; -3 -4] := A[-1 -2; 1] * conj(A[-3 -4; 1])
@tensor BB[-1 -2; -3 -4] := B[-1 -2; 1] * conj(B[-3 -4; 1])

@assert norm(BB - AA) < 1e-14

UA, ΣA, VA = tsvd(AA)
UB, ΣB, VB = tsvd(BB)

AA_rec = UA*ΣA*VA
@tensor AA_rec2[-1 -2; -3 -4] := UA[-1 -2; 1] * ΣA[1; 2] * VA[2; -3 -4]

BB_rec = UB*ΣB*VB
@tensor BB_rec2[-1 -2; -3 -4] := UB[-1 -2; 1] * ΣB[1; 2] * VB[2; -3 -4]

BBarray = convert(Array, reshape(BB[], 6, 6))
UB2, ΣB2_el, VB2 = generic_svd(BBarray)
ΣB2 = zeros(scalartype(BBarray), 6, 6)
for i = 1:6
    ΣB2[i,i] = ΣB2_el[i]
end
BB2_rec = UB2 * ΣB2 * VB2'

@assert (norm(AA_rec - AA) < 1e-14)
@assert (norm(AA_rec2 - AA) < 1e-14)

@assert (norm(BB_rec - BB) < 1e-36)
@assert (norm(BB_rec2 - BB) < 1e-36)

@assert (norm(BB2_rec - BBarray) < 1e-36)

@assert (norm(AA-BB) < 1e-14)
@assert (norm(ΣA-ΣB) < 1e-14)

@assert norm(ΣB -ΣA) < 1e-14 "error between B and A = $(norm(ΣB -ΣA))"

println("Using doubles, machine precision is around $(norm(AA_rec - AA))")
println("Using BigFloats, machine precision is around $(norm(BB_rec - BB))")


# for (c, b) in blocks(TA)
#     vals, vecs = generic_eigen(b)
#     copy!(b, vecs * LinearAlgebra.diagm(exp.(vals)) * vecs')
# end
# for (c, b) in blocks(TB)
#     vals, vecs = generic_eigen(b)
#     copy!(b, vecs * LinearAlgebra.diagm(exp.(vals)) * vecs')
# end



TB = ones(Complex{BigFloat}, ℂ^2, ℂ^2)
TA = TensorMap(convert(Vector{ComplexF64},TB.data), ℂ^2, ℂ^2)
TBunit = TensorMap(Complex{BigFloat}[1.0 0.0; 0.0 2.0], ℂ^2, ℂ^2)
TBunit_exp = TensorMap(Complex{BigFloat}[exp(BigFloat(1.0)) 0.0; 0.0 exp(BigFloat(2.0))], ℂ^2, ℂ^2)

function own_exponential(t::TensorMap)
    for (c, b) in blocks(t)
        println("Type of b = $(typeof(b))")
        vals, vecs = generic_eigen(b)
        exponential = vecs * LinearAlgebra.diagm(exp.(vals)) * vecs'
        copy!(b, exponential)
    end
    return t
end

exp_TA = exp(TA)
exp_TB = exp(TB)
TBunit_exp2 = exp(TBunit)

@assert (norm(exp_TA-exp_TB) < 1e-14)
@assert (norm(TBunit_exp2-TBunit_exp) < 1e-36)

TB = randn(Complex{BigFloat}, ℂ^2, ℂ^2)
TB = (TB + TB') / 2
TA = TensorMap(convert(Vector{ComplexF64},TB.data), ℂ^2, ℂ^2)

# for (c, b) in blocks(TB)
#     println("Type of b = $(typeof(b))")
#     vals, vecs = GenericLinearAlgebra.eigen(b)
#     exponential = vecs * LinearAlgebra.diagm(exp.(vals)) * vecs'
#     copy!(b, exponential)
# end


# eigvalA, eigvecA = eig(TA)
# eigvalB, eigvecB = eig(TB)

TB = randn(Complex{BigFloat}, ℂ^2 ⊗ ℂ^3, ℂ^2 ⊗ ℂ^3)
TA = TensorMap(convert(Vector{ComplexF64},TB.data), ℂ^2 ⊗ ℂ^3, ℂ^2 ⊗ ℂ^3)

eigvalA, eigvecA = generic_eigen(TA)
eigvalB, eigvecB = generic_eigen(TB)

