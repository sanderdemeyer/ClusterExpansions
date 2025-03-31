using Test
using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare, InfinitePartitionFunction, LocalOperator, vertices
using JLD2

symmetry = nothing
critical = false

setprecision(128)
T = Complex{BigFloat}

p = 3
β = 1e-1

t = T(1.0)
# V = T(-2.5)
V = T(0.0)
μ = 2*V

kinetic_operator = -t * (c_plusmin(T) + c_minplus(T))
number_operator = c_number(T)
@tensor number_twosite[-1 -2; -3 -4] := number_operator[-1; -3] * number_operator[-2; -4]
onesite_op = rmul!(number_operator, -μ)
twosite_op = rmul!(kinetic_operator, -t) + rmul!(number_twosite, V)

I = fℤ₂
function spaces_dict(V, I, i)
    if i == 0
        return Vect[I](0 => 1)
    elseif i < 0
        return Vect[I](0 => 5, 1 => 5)
    else
        if false
            return Vect[I](0 => 2^(i-1), 1 => 2^(i-1))
        else
            return Vect[I](0 => 2^(2*i-1), 1 => 2^(2*i-1))
        end
    end
end

spaces = i -> spaces_dict(V, I, i)

checking = false
if checking
    for checking_number = 0:63
        O, O_clust = clusterexpansion(T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry, verbosity = 0);
    end
else
    O, O_clust = clusterexpansion(T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = symmetry, verbosity = 1);
end
# for checking_number1 = 0:255
#     for checking_number2 = 0:63
#         for checking_number3 = 3:-1:0
#             println("checking number = $(checking_number1), $(checking_number2), $(checking_number3)")
#             O, O_clust = clusterexpansion([checking_number1 checking_number2 checking_number3], T, p, β, twosite_op, onesite_op; levels_convention = "tree_depth", spaces = spaces, symmetry = "C4", verbosity = 0);
#         end
#     end
# end
println("Done")
