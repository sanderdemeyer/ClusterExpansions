module ClusterExpansions

using TensorKit, KrylovKit
using BlockTensorKit: ⊕, SumSpace
using MPSKit, MPSKitModels
using PEPSKit
using Graphs
using Zygote, OptimKit

include("clusterexpansions/utility/loop_filtering.jl")

include("clusterexpansions/utility/symmetries.jl")
include("clusterexpansions/utility/truncations_utility.jl")
include("clusterexpansions/utility/truncations.jl")

include("clusterexpansions/generate_trees.jl")
include("clusterexpansions/generate_loops.jl")
include("clusterexpansions/clusters.jl")
include("clusterexpansions/contractions.jl")
include("clusterexpansions/PEPOs.jl")

include("clusterexpansions/solvers/linearsolvers.jl")
include("clusterexpansions/solvers/nonlinearsolvers_optimkit.jl")
include("clusterexpansions/solve_clusters.jl")

export entanglement_filtering
export exponentiate_hamiltonian, contract_PEPO

export Cluster
export get_nontrivial_terms, get_levels
export solve_4_loop, solve_N_loop
export init_PEPO, get_PEPO
export solve_cluster, get_all_indices, clusterexpansion
export truncate_tensor, truncate_hor, truncate_ver
export flip_arrows, make_translationally_invariant, make_translationally_invariant_fermionic
export find_truncation, apply

export fidelity, apply_isometry
export apply_PEPO_exact
export ExactEnvTruncation, ApproximateEnvTruncation, IntermediateEnvTruncation
export approximate_state, find_isometry, truncation_environment

end # module ClusterExpansions
