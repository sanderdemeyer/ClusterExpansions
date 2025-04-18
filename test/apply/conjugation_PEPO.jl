using TensorKit
using Test

O = randn(ComplexF64, pspace ⊗ pspace', vspace ⊗ vspace ⊗ vspace' ⊗ vspace');
A = randn(ComplexF64, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
O_perm = permute(O, ((1,5,6),(2,3,4)));
O_conj_unperm = convert(TensorMap, O_perm');
O_conj = permute(O_conj_unperm, ((1,4),(2,3,5,6)));

@tensor A_contracted[-1; -2 -3 -4 -5 -6 -7 -8 -9] := A[1; -2 -4 -6 -8] * O[-1 1; -3 -5 -7 -9];
result = @tensor A_contracted[1; 2 3 4 5 6 7 8 9] * conj(A_contracted[1; 2 3 4 5 6 7 8 9]);

result2 = @tensor A[1; 4 5 6 7] * O[2 1; 8 9 10 11] * O_conj[3 2; 8 9 10 11] * conj(A[3; 4 5 6 7])

@test result ≈ result2 atol=1e-12