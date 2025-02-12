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

using GenericLinearAlgebra: svd as generic_svd

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


function custom_add(x::Float64, y::Float64)
    return x + y
end

function custom_add(x::BigFloat, y::BigFloat)
    return x + y
end

x = Float64(1.0)
y = Float64(1e-20)


x = BigFloat(1.0)
y = BigFloat(1e-36)


n0 = BigFloat(1e-30)
n1 = BigFloat(1.0)
n2 = BigFloat(2.0)

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

@assert norm(ΣB -ΣA) < 3*1e-15 "error between B and A = $(norm(ΣB -ΣA))"

println("Using doubles, machine precision is around $(norm(AA_rec - AA))")
println("Using BigFloats, machine precision is around $(norm(BB_rec - BB))")

# function apply_A(x, ::Val{false})
#     @tensor xnew[-1; -2] := C[-1; 1] * x[1; -2]
#     return xnew
# end

# function apply_A(x, ::Val{true})
#     @tensor xnew[-1; -2] := conj(C[1; -1]) * x[1; -2]
#     return xnew
# end

# C = TensorMap(randn, Complex{Float64}, ℂ^2, ℂ^2)
# D = TensorMap(randn, Complex{Float64}, ℂ^2, ℂ^2)
# x0 = TensorMap(randn, Complex{Float64}, ℂ^2, ℂ^2)

# x, info = lssolve(apply_A, D, LSMR(verbosity = 1, maxiter = 1000))

# println("Now with BigFloat")
# C = TensorMap(randn, Complex{BigFloat}, ℂ^2, ℂ^2)
# D = TensorMap(randn, Complex{BigFloat}, ℂ^2, ℂ^2)
# x0 = TensorMap(randn, Complex{BigFloat}, ℂ^2, ℂ^2)

# T0 = zeros(Complex{BigFloat}, 2, 2)
# Trand = randn(Complex{BigFloat}, 2, 2)
# Tone = ones(Complex{BigFloat}, 2, 2)
# println(Trand)
# println(T0 + Trand + Tone*1e-30)