using ClusterExpansions
using PEPSKit
using TensorKit
import TensorKit: flip

pspace = ℂ^2
trivspace = ℂ^1

A = TensorMap(randn, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace')
println(typeof(A))
A_flipped = flip(A, 1)
println(summary(A_flipped))