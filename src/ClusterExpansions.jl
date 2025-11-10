module ClusterExpansions

using TensorKit, TensorKitTensors
using KrylovKit
using BlockTensorKit: ⊕, SumSpace
using MPSKit
using PEPSKit
using Graphs
using Zygote, OptimKit

include("lattices.jl")

include("clusterexpansions/utility/loop_filtering.jl")
include("clusterexpansions/utility/symmetries.jl")

include("utility/canonical_form.jl")
include("utility/observables.jl")

include("time_evolution/utility.jl")
include("time_evolution/global_truncation.jl")
include("time_evolution/local_truncation.jl")
include("time_evolution/vomps_utility.jl")
include("time_evolution/vomps.jl")

include("models/models.jl")
include("models/models_Trotter.jl")

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

# Triangular lattice specific
include("clusterexpansions_triangular/utility/symmetries_triangular.jl")

include("clusterexpansions_triangular/generate_loops_triangular.jl")
include("clusterexpansions_triangular/clusters_triangular.jl")
include("clusterexpansions_triangular/contractions_triangular.jl")
include("clusterexpansions_triangular/PEPOs_triangular.jl")

include("clusterexpansions_triangular/solvers/linearsolvers_triangular.jl")
include("clusterexpansions_triangular/solvers/nonlinearsolvers_optimkit_triangular.jl")
include("clusterexpansions_triangular/solve_clusters_triangular.jl")

export EntanglementFiltering, filter_loop, truncate_loop
export rotl90_fermionic, rotl180_fermionic
export symmetrize

export Canonicalization, canonicalize
export PEPOObservable, PEPO_observables, localoperator_model, calculate_observables

export fidelity, apply_PEPO_exact
export ApproximateEnvTruncation, NoEnvTruncation
export VOPEPO_CTMRG, VOPEPO_VUMPS
export approximate_state

export ClusterExpansion, evolution_operator
export ising_operators
export spinless_fermion_operators, spinless_fermion_model
export heisenberg_operators, J1J2_operators
export tJ_operators, hubbard_operators

export TrotterDecomposition, ising_operators_Trotter

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
