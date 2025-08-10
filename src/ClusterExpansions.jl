module ClusterExpansions

using TensorKit, TensorKitTensors
using KrylovKit
using BlockTensorKit: ⊕, SumSpace
using MPSKit
using PEPSKit
using Graphs
using Zygote, OptimKit

include("clusterexpansions/utility/loop_filtering.jl")
include("clusterexpansions/utility/symmetries.jl")

include("utility/canonical_form.jl")
include("utility/observables.jl")

include("time_evolution/truncations_utility.jl")
include("time_evolution/truncations.jl")
include("time_evolution/vomps.jl")

include("models/models.jl")

include("time_evolution/time_evolve.jl")
include("models/time_evolve_models.jl")

include("models/data_generation.jl")

include("clusterexpansions/generate_trees.jl")
include("clusterexpansions/generate_loops.jl")
include("clusterexpansions/clusters.jl")
include("clusterexpansions/contractions.jl")
include("clusterexpansions/PEPOs.jl")

include("clusterexpansions/solvers/linearsolvers.jl")
include("clusterexpansions/solvers/nonlinearsolvers_optimkit.jl")
include("clusterexpansions/solve_clusters.jl")

export EntanglementFiltering, filter_loop, truncate_loop
export rotl90_fermionic, rotl180_fermionic
export symmetrize

export Canonicalization, canonicalize
export PEPOObservable, PEPO_observables, localoperator_model, calculate_observables

export fidelity, apply_isometry
export ExactEnvTruncation, ApproximateEnvTruncation, IntermediateEnvTruncation, NoEnvTruncation
export VOPEPO_CTMRG, VOPEPO_VUMPS
export approximate_state, find_isometry, apply_PEPO_exact

export ClusterExpansion, evolution_operator
export ising_operators, ising_operators_Z2
export spinless_fermion_operators, spinless_fermion_model
export heisenberg_operators, J1J2_operators
export tJ_operators, hubbard_operators

export evolution_operator
export StaticTimeEvolution, TimeDependentTimeEvolution
export UniformTimeEvolution, SquaringTimeEvolution
export UniformGroundStateTimeEvolution, SquaringGroundStateTimeEvolution
export time_evolve, get_time_array, time_scan
export time_evolve_model, time_scan_model

export data_generation_SF_CE, data_generation_ising_CE
export data_generation_SF_SU, data_generation_ising_SU

export clusterexpansion

end # module ClusterExpansions
