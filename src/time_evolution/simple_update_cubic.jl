#
# Simple update (SU) for infinite PEPS on the 3-dimensional cubic lattice.
#
# The conventions follow the ones already used for the cubic-lattice PEPOs in this package:
# the six virtual legs of a site tensor are ordered as
#
#       (UP, NORTH, EAST, DOWN, SOUTH, WEST)
#
# with the first three spaces non-dual and the last three dual, i.e.
#
#       A : P ← V_U ⊗ V_N ⊗ V_E ⊗ V_D ⊗ V_S ⊗ V_W
#
# Pictorially (the numbers are the leg indices of `A`):
#
#            2   3
#            ↑  ↗
#            | /
#      7 ←---A---→ 4          1 = physical
#           /|                (2,5) = (UP, DOWN)     along the z-axis
#          ↙ ↓                (3,6) = (NORTH, SOUTH) along the y-axis
#         6  5                (4,7) = (EAST, WEST)   along the x-axis
#
# Sites of the unit cell are labelled by a `CartesianIndex{3}` `(x, y, z)`; the positive
# direction along the x/y/z axis is EAST/NORTH/UP respectively.
#

# leg `d + 1` of a site tensor carries direction `d`
const DIR_UP, DIR_NORTH, DIR_EAST, DIR_DOWN, DIR_SOUTH, DIR_WEST = 1, 2, 3, 4, 5, 6

"""Axis (1 = x, 2 = y, 3 = z) along which direction `dir` points."""
_axis3D(dir::Int) = (3, 2, 1, 3, 2, 1)[dir]

"""Direction pointing in the positive/negative sense of `axis` (1 = x, 2 = y, 3 = z)."""
_posdir3D(axis::Int) = (DIR_EAST, DIR_NORTH, DIR_UP)[axis]
_negdir3D(axis::Int) = (DIR_WEST, DIR_SOUTH, DIR_DOWN)[axis]

_isposdir3D(dir::Int) = dir <= 3
_opposite3D(dir::Int) = mod1(dir + 3, 6)

"""Shift `site` by `δ` steps along `axis`, periodic in a unit cell of size `sz`."""
function _shift_site(site::CartesianIndex{3}, axis::Int, δ::Int, sz::NTuple{3,Int})
    return CartesianIndex(ntuple(i -> i == axis ? mod1(site[i] + δ, sz[i]) : site[i], 3))
end

_elementaryspace(V::ElementarySpace) = V
_elementaryspace(d::Int) = ℂ^d

const PEPSTensor3D{S} = AbstractTensorMap{<:Any,S,1,6}
const PEPSWeight3D{T,S} = AbstractTensorMap{T,S,1,1}

#
## Bond weights
#

"""
    struct SUWeight3D{E<:PEPSWeight3D}

Schmidt bond weights of a cubic-lattice PEPS, stored as `data[axis, x, y, z]`.
`data[a, x, y, z]` lives on the bond connecting site `(x, y, z)` to its neighbour in the
positive direction along axis `a` (1 = x/EAST, 2 = y/NORTH, 3 = z/UP).

Every weight is a real, non-negative `DiagonalTensorMap` on a non-dual space.
"""
struct SUWeight3D{E<:PEPSWeight3D}
    data::Array{E,4}
    SUWeight3D{E}(data::Array{E,4}) where {E} = new{E}(data)
end

function SUWeight3D(data::Array{E,4}) where {E<:PEPSWeight3D}
    scalartype(data) <: Real || error("Weight elements must be real numbers.")
    size(data, 1) == 3 || error("The first axis of the weight array must have size 3.")
    for wt in data
        isa(wt, DiagonalTensorMap) ||
            error("Each weight matrix should be a DiagonalTensorMap")
        domain(wt, 1) == codomain(wt, 1) ||
            error("Domain and codomain of each weight matrix must be the same.")
        !isdual(codomain(wt, 1)) ||
            error("Domain and codomain of each weight matrix cannot be a dual space.")
        all(wt.data .>= 0) || error("Weight elements must be non-negative.")
    end
    return SUWeight3D{E}(data)
end

Base.size(W::SUWeight3D) = size(W.data)
Base.size(W::SUWeight3D, i) = size(W.data, i)
Base.length(W::SUWeight3D) = length(W.data)
Base.eltype(::Type{SUWeight3D{E}}) where {E} = E
Base.eltype(W::SUWeight3D) = eltype(typeof(W))
TensorKit.VectorInterface.scalartype(::Type{T}) where {T<:SUWeight3D} = scalartype(eltype(T))

Base.getindex(W::SUWeight3D, args...) = Base.getindex(W.data, args...)
Base.setindex!(W::SUWeight3D, args...) = (Base.setindex!(W.data, args...); W)
Base.axes(W::SUWeight3D, args...) = axes(W.data, args...)
Base.iterate(W::SUWeight3D, args...) = iterate(W.data, args...)

"""Size `(Nx, Ny, Nz)` of the unit cell that `W` is defined on."""
lattice_size(W::SUWeight3D) = size(W.data)[2:end]

Base.:(==)(wts1::SUWeight3D, wts2::SUWeight3D) = wts1.data == wts2.data
function Base.isapprox(wts1::SUWeight3D, wts2::SUWeight3D; kwargs...)
    return all(isapprox(wt1, wt2; kwargs...) for (wt1, wt2) in zip(wts1, wts2))
end

PEPSKit._singular_value_distance(tup::NTuple{2}) = PEPSKit._singular_value_distance(tup...)

"""
    compare_weights(wts1::SUWeight3D, wts2::SUWeight3D)

Mean singular-value distance between two sets of bond weights, used as SU convergence
measure. Weights living on different (truncated) spaces are compared on their common
subspace.
"""
function compare_weights(wts1::SUWeight3D, wts2::SUWeight3D)
    @assert size(wts1) == size(wts2)
    return sum(PEPSKit._singular_value_distance, zip(wts1.data, wts2.data)) / length(wts1)
end

function Base.show(io::IO, ::MIME"text/plain", wts::SUWeight3D)
    println(io, typeof(wts))
    for idx in CartesianIndices(wts.data)
        println(io, Tuple(idx), ":")
        for (k, b) in blocks(wts.data[idx])
            println(io, k, " = ", diag(b))
        end
    end
    return nothing
end

#
## States
#

"""
    struct InfiniteWeightPEPS3D{T<:PEPSTensor3D,E<:PEPSWeight3D}

Infinite PEPS on the 3-dimensional cubic lattice consisting of vertex tensors
`vertices[x, y, z]` and bond weights `weights[axis, x, y, z]`.

The unit cell must have size at least 2 along every axis -- for a smaller unit cell a bond
would connect a tensor to itself, which simple update cannot handle. The canonical choice
for e.g. a Néel-ordered state is a `(2, 2, 2)` unit cell.

## Constructors

    InfiniteWeightPEPS3D(vertices::Array{T,3}, weights::SUWeight3D)
    InfiniteWeightPEPS3D([f=randn, T=ComplexF64,] Pspaces, Vxspaces, Vyspaces, Vzspaces)
    InfiniteWeightPEPS3D([f=randn, T=ComplexF64,] Pspace, Vspace; unitcell=(2, 2, 2))

The three `V*spaces` arrays hold, for every site, the (non-dual) virtual space of the bond
leaving that site in the positive x/y/z direction.
"""
struct InfiniteWeightPEPS3D{T<:PEPSTensor3D,E<:PEPSWeight3D}
    vertices::Array{T,3}
    weights::SUWeight3D{E}

    function InfiniteWeightPEPS3D(
        vertices::Array{T,3}, weights::SUWeight3D{E}
    ) where {T<:PEPSTensor3D,E<:PEPSWeight3D}
        @assert size(vertices) == lattice_size(weights)
        sz = size(vertices)
        all(n -> n >= 2, sz) || throw(
            ArgumentError(
                "The unit cell must have size >= 2 along every axis, got $(sz). " *
                "A smaller unit cell would make a bond connect a tensor to itself.",
            ),
        )
        for site in CartesianIndices(vertices), axis in 1:3
            wt = weights[axis, site]
            nb = _shift_site(site, axis, 1, sz)
            pos, neg = _posdir3D(axis), _negdir3D(axis)
            space(wt, 1)' == space(vertices[site], 1 + pos) || throw(
                SpaceMismatch(
                    "Positive-direction space of bond weight [$axis, $(Tuple(site))] does not match.",
                ),
            )
            space(wt, 2)' == space(vertices[nb], 1 + neg) || throw(
                SpaceMismatch(
                    "Negative-direction space of bond weight [$axis, $(Tuple(site))] does not match.",
                ),
            )
        end
        return new{T,E}(vertices, weights)
    end
end

function InfiniteWeightPEPS3D(
    Pspaces::A, Vxspaces::A, Vyspaces::A, Vzspaces::A
) where {A<:AbstractArray{<:Union{Int,ElementarySpace},3}}
    return InfiniteWeightPEPS3D(randn, ComplexF64, Pspaces, Vxspaces, Vyspaces, Vzspaces)
end
function InfiniteWeightPEPS3D(
    f, T, Pspaces::A, Vxspaces::A, Vyspaces::A, Vzspaces::A
) where {A<:AbstractArray{<:Union{Int,ElementarySpace},3}}
    @assert size(Pspaces) == size(Vxspaces) == size(Vyspaces) == size(Vzspaces)
    sz = size(Pspaces)
    Vspaces = (Vxspaces, Vyspaces, Vzspaces)
    @assert all(V -> !isdual(_elementaryspace(V)), Vxspaces)
    @assert all(V -> !isdual(_elementaryspace(V)), Vyspaces)
    @assert all(V -> !isdual(_elementaryspace(V)), Vzspaces)

    vertices = map(CartesianIndices(Pspaces)) do site
        # positive directions carry their own space, negative ones the neighbour's dual
        Vs = ntuple(6) do dir
            axis = _axis3D(dir)
            if _isposdir3D(dir)
                _elementaryspace(Vspaces[axis][site])
            else
                _elementaryspace(Vspaces[axis][_shift_site(site, axis, -1, sz)])'
            end
        end
        return f(T, _elementaryspace(Pspaces[site]) ← ⊗(Vs...))
    end
    weights = map(Iterators.product(1:3, ntuple(i -> 1:sz[i], 3)...)) do (axis, x, y, z)
        V = _elementaryspace(Vspaces[axis][x, y, z])
        return DiagonalTensorMap(ones(reduceddim(V)), V)
    end
    return InfiniteWeightPEPS3D(collect(vertices), SUWeight3D(collect(weights)))
end

function InfiniteWeightPEPS3D(
    Pspace::S, Vspace::S; unitcell::NTuple{3,Int}=(2, 2, 2)
) where {S<:ElementarySpace}
    return InfiniteWeightPEPS3D(randn, ComplexF64, Pspace, Vspace; unitcell)
end
function InfiniteWeightPEPS3D(
    f, T, Pspace::S, Vspace::S; unitcell::NTuple{3,Int}=(2, 2, 2)
) where {S<:ElementarySpace}
    return InfiniteWeightPEPS3D(
        f,
        T,
        fill(Pspace, unitcell),
        fill(Vspace, unitcell),
        fill(Vspace, unitcell),
        fill(Vspace, unitcell),
    )
end

"""
    product_peps3D([f=randn, T,] states::AbstractArray{<:AbstractTensorMap,3},
                   Vxspaces, Vyspaces, Vzspaces; noise=1e-1)
    product_peps3D([f=randn, T,] state::AbstractTensorMap, Vspace;
                   unitcell=(2, 2, 2), noise=1e-1)

Product state on the cubic lattice, embedded in the given virtual spaces and perturbed by a
random tensor of relative size `noise`. Every `state` is a physical vector, i.e. a tensor
map with one codomain index and no domain indices.

This is the recommended starting point for [`simpleupdate3D`](@ref). A *random* cubic-lattice
PEPS carries close to maximal junk entanglement on all six virtual legs, which simple update
at fixed bond dimension cannot shed -- it then stalls in a badly converged state. Starting
from a weakly perturbed product state avoids this. The perturbation is required: with
`noise = 0` all bond weights are exactly rank one and the state stays frozen at `D = 1`.
"""
function product_peps3D(states::AbstractArray{<:AbstractTensorMap,3}, args...; kwargs...)
    return product_peps3D(randn, scalartype(first(states)), states, args...; kwargs...)
end
function product_peps3D(
    f,
    T,
    states::AbstractArray{<:AbstractTensorMap,3},
    Vxspaces::AA,
    Vyspaces::AA,
    Vzspaces::AA;
    noise::Real=1e-1,
) where {AA<:AbstractArray{<:Union{Int,ElementarySpace},3}}
    Pspaces = map(s -> space(s, 1), states)
    peps = InfiniteWeightPEPS3D(f, T, Pspaces, Vxspaces, Vyspaces, Vzspaces)
    for site in CartesianIndices(states)
        A = peps.vertices[site]
        prod = _product_tensor3D(T, states[site], domain(A))
        peps.vertices[site] = prod / norm(prod) + (noise / norm(A)) * A
    end
    return peps
end
function product_peps3D(state::AbstractTensorMap, Vspace; kwargs...)
    return product_peps3D(randn, scalartype(state), state, Vspace; kwargs...)
end
function product_peps3D(
    f, T, state::AbstractTensorMap, Vspace; unitcell::NTuple{3,Int}=(2, 2, 2), kwargs...
)
    return product_peps3D(
        f,
        T,
        fill(state, unitcell),
        fill(Vspace, unitcell),
        fill(Vspace, unitcell),
        fill(Vspace, unitcell);
        kwargs...,
    )
end

"""Embed the physical vector `state` into a PEPS tensor with virtual spaces `doms`."""
function _product_tensor3D(T, state::AbstractTensorMap, doms::ProductSpace{<:Any,6})
    numout(state) == 1 && numin(state) == 0 ||
        throw(ArgumentError("A product state must be a vector: one codomain, no domain index."))
    # a unit vector in the dual of each virtual space, so that the @tensor below produces
    # exactly the domain spaces `doms`
    vs = ntuple(6) do d
        V = dual(doms[d])
        return removeunit(isometry(T, V, oneunit(V)), 2)
    end
    @tensor A[-1; -2 -3 -4 -5 -6 -7] :=
        state[-1] * vs[1][-2] * vs[2][-3] * vs[3][-4] * vs[4][-5] * vs[5][-6] * vs[6][-7]
    return A
end

Base.size(peps::InfiniteWeightPEPS3D) = size(peps.vertices)
Base.size(peps::InfiniteWeightPEPS3D, i) = size(peps.vertices, i)
function TensorKit.VectorInterface.scalartype(::Type{<:InfiniteWeightPEPS3D{T}}) where {T}
    return scalartype(T)
end

function Base.:(==)(peps1::InfiniteWeightPEPS3D, peps2::InfiniteWeightPEPS3D)
    return peps1.vertices == peps2.vertices && peps1.weights == peps2.weights
end
function Base.isapprox(peps1::InfiniteWeightPEPS3D, peps2::InfiniteWeightPEPS3D; kwargs...)
    all(isapprox(v1, v2; kwargs...) for (v1, v2) in zip(peps1.vertices, peps2.vertices)) ||
        return false
    return isapprox(peps1.weights, peps2.weights; kwargs...)
end

function Base.show(io::IO, ::MIME"text/plain", peps::InfiniteWeightPEPS3D)
    println(io, "$(join(size(peps), "×")) InfiniteWeightPEPS3D:")
    for site in CartesianIndices(peps.vertices)
        println(io, "  $(Tuple(site)): ", space(peps.vertices[site]))
    end
    return nothing
end

"""
    struct InfinitePEPS3D{T<:PEPSTensor3D}

Cubic-lattice PEPS without bond weights, obtained from an [`InfiniteWeightPEPS3D`](@ref) by
absorbing all weights into the vertex tensors. This is the object to feed into a 3D
contraction algorithm.
"""
struct InfinitePEPS3D{T<:PEPSTensor3D}
    A::Array{T,3}
end

Base.size(peps::InfinitePEPS3D, args...) = size(peps.A, args...)
Base.getindex(peps::InfinitePEPS3D, args...) = getindex(peps.A, args...)
Base.setindex!(peps::InfinitePEPS3D, args...) = (setindex!(peps.A, args...); peps)
Base.axes(peps::InfinitePEPS3D, args...) = axes(peps.A, args...)

function InfinitePEPS3D(peps::InfiniteWeightPEPS3D)
    alldirs = ntuple(identity, 6)
    # square roots on both sides of a bond, so that each bond weight is absorbed exactly once
    alltrue = ntuple(Returns(true), 6)
    A = map(CartesianIndices(peps.vertices)) do site
        return _absorb_weights3D(
            peps.vertices[site], peps.weights, site, alldirs, alltrue, false
        )
    end
    return InfinitePEPS3D(collect(A))
end

"""
    InfiniteWeightPEPS3D(peps::InfinitePEPS3D)

Attach trivial (identity) bond weights to a cubic-lattice PEPS.
"""
function InfiniteWeightPEPS3D(peps::InfinitePEPS3D)
    weights = map(Iterators.product(1:3, axes(peps.A)...)) do (axis, x, y, z)
        V = domain(peps[x, y, z], _posdir3D(axis))
        @assert !isdual(V)
        return DiagonalTensorMap(ones(reduceddim(V)), V)
    end
    return InfiniteWeightPEPS3D(copy(peps.A), SUWeight3D(collect(weights)))
end

#
## Absorbing bond weights
#

"""
    _absorb_weights3D(t, weights, site, dirs, sqrtwts, invwt)

Absorb the bond weights adjacent to `site` on the directions `dirs` into the vertex tensor
`t`. `sqrtwts[i]` selects the square root of the corresponding weight and `invwt` inverts
all of them (i.e. removes the weights instead of absorbing them).
"""
function _absorb_weights3D(
    t::PEPSTensor3D,
    weights::SUWeight3D,
    site::CartesianIndex{3},
    dirs::NTuple{N,Int},
    sqrtwts::NTuple{N,Bool},
    invwt::Bool,
) where {N}
    sz = lattice_size(weights)
    @assert all(1 .<= Tuple(site) .<= sz)
    @assert 1 <= N <= 6
    @assert allunique(dirs)
    tensors = Vector{AbstractTensorMap}()
    indices = Vector{Vector{Int}}()
    indices_t = collect(-1:-1:-7)
    for (dir, sqrtwt) in zip(dirs, sqrtwts)
        @assert 1 <= dir <= 6
        dirp1 = dir + 1
        indices_t[dirp1] *= -1
        axis = _axis3D(dir)
        wt = if _isposdir3D(dir)
            weights[axis, site]
        else
            weights[axis, _shift_site(site, axis, -1, sz)]
        end
        @assert !isdual(space(wt, 1)) && isdual(space(wt, 2))
        if !sqrtwt && !invwt
            push!(tensors, wt)
        else
            pow = (sqrtwt ? 1//2 : 1) * (invwt ? -1 : 1)
            push!(tensors, PEPSKit.sdiag_pow(wt, pow))
        end
        # a weight sitting on a positive direction is contracted with its dual (2nd) index
        push!(indices, _isposdir3D(dir) ? [dirp1, -dirp1] : [-dirp1, dirp1])
    end
    push!(tensors, t)
    push!(indices, indices_t)
    return permute(ncon(tensors, indices), ((1,), ntuple(i -> i + 1, 6)))
end

"""
    absorb_weight(t::PEPSTensor3D, site::CartesianIndex{3}, dir::Int, weights::SUWeight3D;
                  sqrtwt::Bool=false, invwt::Bool=false)

Absorb (or, with `invwt = true`, remove) the bond weight on direction `dir` of the vertex
tensor `t` located at `site`. `dir` runs over `1:6` in the order
`(UP, NORTH, EAST, DOWN, SOUTH, WEST)`.
"""
function absorb_weight(
    t::PEPSTensor3D,
    site::CartesianIndex{3},
    dir::Int,
    weights::SUWeight3D;
    sqrtwt::Bool=false,
    invwt::Bool=false,
)
    return _absorb_weights3D(t, weights, site, (dir,), (sqrtwt,), invwt)
end

#
## Bond update
#

"""
    _qr_bond3D(A, B, dir)

Split off the parts of the neighbouring tensors `A` and `B` that are touched by a gate
acting on the bond in direction `dir` (as seen from `A`):

```
    A ─→ B   ====>   X ─→ a ─→ b ─→ Y
```

`a` carries the indices `(X-bond, physical, shared bond)` and `b` the indices
`(shared bond, physical, Y-bond)`.
"""
function _qr_bond3D(A::PEPSTensor3D, B::PEPSTensor3D, dir::Int)
    dirop = _opposite3D(dir)
    lA, lB = dir + 1, dirop + 1
    @assert isdual(space(A, lA)) # A ← B
    restA = Tuple(filter(!=(lA), 2:7))
    restB = Tuple(filter(!=(lB), 2:7))
    X, a = left_orth(permute(A, (restA, (1, lA))))
    Y, b = left_orth(permute(B, (restB, (1, lB))))
    @assert !isdual(space(a, 1))
    @assert !isdual(space(b, 1))
    b = permute(b, ((3, 2), (1,)))
    return X, a, b, Y
end

"""
    _qr_bond_undo3D(X, a, b, Y, dir)

Inverse of [`_qr_bond3D`](@ref): glue the (updated) reduced tensors back onto their
orthogonal parts. The virtual leg produced by the QR is re-inserted at the position of the
bond leg, so that the result again has the canonical `P ← U N E D S W` leg order.
"""
function _qr_bond_undo3D(X, a, b, Y, dir::Int)
    dirop = _opposite3D(dir)
    idxX = [[-(d + 1) for d in 1:6 if d != dir]; 1]
    idxa = [1, -1, -(dir + 1)]
    idxY = [[-(d + 1) for d in 1:6 if d != dirop]; 1]
    idxb = [-(dirop + 1), -1, 1]
    dom = ntuple(i -> i + 1, 6)
    A = permute(ncon([X, a], [idxX, idxa]), ((1,), dom))
    B = permute(ncon([b, Y], [idxb, idxY]), ((1,), dom))
    return A, B
end

"""
    _apply_gate3D(a, b, gate, trscheme)

Apply the two-site `gate` to the reduced tensors `a`, `b` and truncate the resulting bond
according to `trscheme`.

```
    -1 ← a ← 3 ← b ← -4
         ↓        ↓
         1        2
         ↓        ↓
         |──gate──|
         ↓        ↓
        -2       -3
```
"""
function _apply_gate3D(a, b, gate::AbstractTensorMap{<:Any,S,2,2}, trscheme) where {S}
    @tensor a2b2[-1 -2; -3 -4] := gate[-2 -3; 1 2] * a[-1 1 3] * b[3 2 -4]
    trunc = if trscheme isa FixedSpaceTruncation
        V = space(b, 1)
        truncspace(isdual(V) ? V' : V)
    else
        trscheme
    end
    return svd_trunc!(a2b2; trunc)#, alg=TensorKit.SVD())
end

#
## Hamiltonians
#

"""
    abstract type AbstractCubicHamiltonian

Supertype for the nearest-neighbour bond data fed to [`simpleupdate3D`](@ref): anything
that supports `size`, `getindex` (both over `[axis, x, y, z]`) and [`get_expham`](@ref).
Implementations: [`CubicHamiltonian`](@ref) (ground-state simple update) and
[`ThermalCubicHamiltonian`](@ref) (finite-temperature simple update).
"""
abstract type AbstractCubicHamiltonian end

"""
    struct CubicHamiltonian{T<:AbstractTensorMap} <: AbstractCubicHamiltonian

Nearest-neighbour Hamiltonian on the cubic lattice. `terms[axis, x, y, z]` is the two-site
operator acting on the bond between `(x, y, z)` and its neighbour in the positive direction
along `axis`; on-site terms are folded into the bond terms.
"""
struct CubicHamiltonian{T<:AbstractTensorMap} <: AbstractCubicHamiltonian
    terms::Array{T,4}
end

Base.size(H::CubicHamiltonian) = size(H.terms)[2:end]
Base.getindex(H::CubicHamiltonian, args...) = getindex(H.terms, args...)
PEPSKit.physicalspace(H::CubicHamiltonian) = space(first(H.terms), 1)

"""
    CubicHamiltonian(twosite, onesite=nothing; unitcell=(2, 2, 2))

Build a uniform nearest-neighbour cubic-lattice Hamiltonian.

`twosite` is either a single two-site operator `P ⊗ P ← P ⊗ P` used on all bonds, or a
3-tuple of such operators giving the interaction along the x-, y- and z-axis separately.
An on-site operator `onesite` is distributed evenly over the six bonds meeting at each site,
so that [`simpleupdate3D`](@ref) only ever has to apply two-site gates.
"""
function CubicHamiltonian(twosite, onesite=nothing; unitcell::NTuple{3,Int}=(2, 2, 2))
    twosites = twosite isa Tuple ? twosite : ntuple(Returns(twosite), 3)
    length(twosites) == 3 ||
        throw(ArgumentError("`twosite` must be a single operator or a 3-tuple."))
    all(n -> n >= 2, unitcell) || throw(
        ArgumentError("The unit cell must have size >= 2 along every axis, got $(unitcell)."),
    )
    bonds = map(twosites) do h
        numin(h) == numout(h) == 2 ||
            throw(ArgumentError("`twosite` must be a two-site operator."))
        onesite === nothing && return h
        unit = id(scalartype(onesite), space(h, 1))
        # every site is shared by the 6 bonds of the cubic lattice
        return h + (onesite ⊗ unit + unit ⊗ onesite) / 6
    end
    terms = map(Iterators.product(1:3, ntuple(i -> 1:unitcell[i], 3)...)) do idx
        return bonds[first(idx)]
    end
    return CubicHamiltonian(collect(terms))
end

"""
    get_expham(dt::Number, H::CubicHamiltonian)

Exponentiate every bond term of `H` into the imaginary-time gate `exp(-dt * h)`.
"""
get_expham(dt::Number, H::CubicHamiltonian) = CubicHamiltonian(map(h -> exp(-dt * h), H.terms))

#
## Finite-temperature simple update
#
# The density operator ρ = e^{-βH} is represented directly (no ancilla/purification) by
# fusing, at every site, the ket and bra copies of the physical leg into a single leg of
# dimension d². This is exactly the same trick as representing an operator by its
# "vectorization"; see `DensityMatrixSpace` below. Since the fused leg is still a single
# leg, `InfiniteWeightPEPS3D`/`SimpleUpdate3D` do not need to know that it secretly
# represents an operator rather than a wavefunction amplitude -- only the initial state and
# the two-site gate differ from the ground-state case.
#

"""
    struct DensityMatrixSpace{S<:ElementarySpace,TU<:AbstractTensorMap}

Bookkeeping for representing a density operator on physical space `P` as a single
(vectorized) PEPS leg, as used for finite-temperature simple update (see
[`ThermalCubicHamiltonian`](@ref) and [`infinite_temperature_peps3D`](@ref)).

The physical leg of such a "density-operator PEPS" is the *fused* ket-bra space
`fused = fuse(P, P')`, related to the doubled ket-bra space `P ⊗ P'` through the
isomorphism `U : P ⊗ P' ← fused`. A physical operator `O : P ← P` is embedded into the
fused leg via [`vectorize`](@ref); the reverse map is [`unvectorize`](@ref).
"""
struct DensityMatrixSpace{S<:ElementarySpace,TU<:AbstractTensorMap}
    P::S
    fused::S
    U::TU
end

function DensityMatrixSpace(T::Type{<:Number}, P::ElementarySpace)
    fused = fuse(P, P')
    return DensityMatrixSpace(P, fused, isomorphism(T, P ⊗ P', fused))
end
DensityMatrixSpace(P::ElementarySpace) = DensityMatrixSpace(ComplexF64, P)

"""
    vectorize(dm::DensityMatrixSpace, O::AbstractTensorMap{<:Any,S,1,1})

Embed a physical operator `O : P ← P` into the fused ket-bra leg of `dm`, giving a
single-leg PEPS "vector". For a Hermitian `O`, this same vector also serves as the
"cap" that computes `Tr[Oρ]` when contracted against a density-operator PEPS leg (with
`O = id(P)` giving the trace/normalization cap).
"""
function vectorize(dm::DensityMatrixSpace, O::AbstractTensorMap)
    return dm.U' * permute(O, ((1, 2), ()))
end

"""
    unvectorize(dm::DensityMatrixSpace, v::AbstractTensorMap{<:Any,S,1,0})

Inverse of [`vectorize`](@ref): recover the physical operator `P ← P` embedded in the
fused ket-bra vector `v`.
"""
function unvectorize(dm::DensityMatrixSpace, v::AbstractTensorMap)
    return permute(dm.U * v, ((1,), (2,)))
end

"""
    _bra_conj(t::AbstractTensorMap)

Entrywise complex conjugate of `t`, with codomain and domain replaced by their duals
(index *positions* are unchanged, unlike `t'`, which also swaps codomain and domain).
This implements the "conjugate operator acting on the bra copy" `Ḡ` used to build
[`thermal_gate`](@ref): if `ρ ↦ GρG^†`, then in vectorized form `vec(ρ) ↦ (G ⊗ Ḡ) vec(ρ)`.
"""
function _bra_conj(t::AbstractTensorMap)
    W = similar(t, dual(codomain(t)) ← dual(domain(t)))
    for (c, b) in blocks(t)
        copy!(block(W, dual(c)), conj(b))
    end
    return W
end

"""
    thermal_gate(dt::Number, h::AbstractTensorMap{<:Any,S,2,2}, dm::DensityMatrixSpace)

Two-site gate implementing `ρ ↦ exp(-dt/2 * h) ρ exp(-dt/2 * h)` on the fused ket-bra legs
of two neighbouring sites, given the physical (non-doubled) bond Hamiltonian term
`h : P ⊗ P ← P ⊗ P`. This is the finite-temperature analogue of the ket-only gate
`exp(-dt * h)` used for ground-state simple update: since the sandwiching gate appears on
both sides of ρ, each bond only needs to be evolved by `dt/2` to advance `ρ` by `dt`.
"""
function thermal_gate(
    dt::Number, h::AbstractTensorMap{<:Any,S,2,2}, dm::DensityMatrixSpace{S}
) where {S}
    g = exp((-dt / 2) * h)
    gg = permute(g ⊗ _bra_conj(g), ((1, 3, 2, 4), (5, 7, 6, 8)))
    Ubig = dm.U ⊗ dm.U
    return Ubig' * gg * Ubig
end

"""
    struct ThermalCubicHamiltonian{T<:AbstractTensorMap,S<:ElementarySpace} <: AbstractCubicHamiltonian

Cubic-lattice Hamiltonian for finite-temperature simple update on the *vectorized* density
operator. Same bond-term bookkeeping as [`CubicHamiltonian`](@ref) (`terms` holds the plain,
non-doubled physical bond operators), but [`get_expham`](@ref) builds the doubled/fused
[`thermal_gate`](@ref) instead of the ket-only `exp(-dt*h)`.
"""
struct ThermalCubicHamiltonian{T<:AbstractTensorMap,S<:ElementarySpace} <:
       AbstractCubicHamiltonian
    terms::Array{T,4}
    dm::DensityMatrixSpace{S}
end

Base.size(H::ThermalCubicHamiltonian) = size(H.terms)[2:end]
Base.getindex(H::ThermalCubicHamiltonian, args...) = getindex(H.terms, args...)
PEPSKit.physicalspace(H::ThermalCubicHamiltonian) = H.dm.fused

"""
    ThermalCubicHamiltonian(dm::DensityMatrixSpace, twosite, onesite=nothing; unitcell=(2, 2, 2))

Build a [`ThermalCubicHamiltonian`](@ref) the same way [`CubicHamiltonian`](@ref) is built,
tagging it with the `dm` bookkeeping needed to construct the finite-temperature gates.
"""
function ThermalCubicHamiltonian(
    dm::DensityMatrixSpace, twosite, onesite=nothing; unitcell::NTuple{3,Int}=(2, 2, 2)
)
    return ThermalCubicHamiltonian(CubicHamiltonian(twosite, onesite; unitcell).terms, dm)
end

"""
    get_expham(dt::Number, H::ThermalCubicHamiltonian)

Build the finite-temperature [`thermal_gate`](@ref) for every bond term of `H`.
"""
function get_expham(dt::Number, H::ThermalCubicHamiltonian)
    return CubicHamiltonian(map(h -> thermal_gate(dt, h, H.dm), H.terms))
end

"""
    infinite_temperature_peps3D(dm::DensityMatrixSpace, Vspace; unitcell=(2, 2, 2), noise=1e-1)

Product state representing the infinite-temperature density operator `ρ ∝ 𝟙`: the standard
starting point (`β = 0`) for finite-temperature simple update with a
[`ThermalCubicHamiltonian`](@ref). As for [`product_peps3D`](@ref), `noise` perturbs the
state away from the exactly rank-1 identity so that the bonds can develop entanglement
during the evolution.
"""
function infinite_temperature_peps3D(
    dm::DensityMatrixSpace, Vspace; unitcell::NTuple{3,Int}=(2, 2, 2), noise::Real=1e-1
)
    return infinite_temperature_peps3D(dm, Vspace, Vspace, Vspace; unitcell, noise)
end
function infinite_temperature_peps3D(
    dm::DensityMatrixSpace,
    Vxspace,
    Vyspace,
    Vzspace;
    unitcell::NTuple{3,Int}=(2, 2, 2),
    noise::Real=1e-1,
)
    seed = vectorize(dm, id(scalartype(dm.U), dm.P))
    states = fill(seed, unitcell)
    Vx = Vxspace isa AbstractArray ? Vxspace : fill(Vxspace, unitcell)
    Vy = Vyspace isa AbstractArray ? Vyspace : fill(Vyspace, unitcell)
    Vz = Vzspace isa AbstractArray ? Vzspace : fill(Vzspace, unitcell)
    return product_peps3D(states, Vx, Vy, Vz; noise)
end

"""
    cubic_pepo(peps::InfiniteWeightPEPS3D, dm::DensityMatrixSpace)

Convert a converged density-operator PEPS (fused ket-bra physical leg) into the cubic-lattice
PEPO convention used elsewhere in this package (see e.g. `evolution_operator_cubic` and
`ctmrg_cubic.jl`): a `{T,S,2,6}` tensor per site with physical legs `(ket, bra')` -- a plain
copy of `P` and its dual -- and the same six virtual legs `(U, N, E, D, S, W)`.

This lets a finite-temperature state obtained from [`simpleupdate3D`](@ref) (with a
[`ThermalCubicHamiltonian`](@ref)) be fed directly into the existing CTMRG3D-based
`MPSKit.expectation_value` machinery to compute `Tr[Oρ]/Tr[ρ]` via a genuine boundary
contraction. For the cheaper bond-weight (belief-propagation) estimate of the observables of
the purified state, see [`onesite_rdm`](@ref)/[`bond_rdm`](@ref) with a `DensityMatrixSpace`.
"""
function cubic_pepo(peps::InfiniteWeightPEPS3D, dm::DensityMatrixSpace)
    p3 = InfinitePEPS3D(peps)
    return map(A -> _cubic_pepo_tensor(dm, A), p3.A)
end

function _cubic_pepo_tensor(dm::DensityMatrixSpace, A::PEPSTensor3D)
    @tensor O[p1 p2; u n e d s w] := dm.U[p1 p2; f] * A[f; u n e d s w]
    return O
end

function cubic_pepo_purified(peps::InfiniteWeightPEPS3D, dm::DensityMatrixSpace)
    p3 = InfinitePEPS3D(peps)
    return map(A -> _cubic_pepo_purified_tensor(dm, A), p3.A)
end

function _cubic_pepo_purified_tensor(dm::DensityMatrixSpace, A::PEPSTensor3D)
    Vs = ntuple(k -> k <= 3 ? domain(A)[k] : domain(A)[k]', 6)
    F = map(V -> isometry(fuse(V, V'), V ⊗ V'), Vs)
    Fu, Fn, Fe = adjoint(F[1]), adjoint(F[2]), adjoint(F[3])
    Fd, Fs, Fw = F[4], F[5], F[6]
    @tensor O[p1 p2; u n e d s w] :=
        dm.U[p1 a; f1] * A[f1; u1 n1 e1 d1 s1 w1] *
        conj(dm.U[p2 a; f2]) * conj(A[f2; u2 n2 e2 d2 s2 w2]) *
        Fu[u1 u2; u] * Fn[n1 n2; n] * Fe[e1 e2; e] *
        Fd[d; d1 d2] * Fs[s; s1 s2] * Fw[w; w1 w2]
    return O
end



#
## Simple update
#

"""
    struct SimpleUpdate3D

Algorithm struct for simple update of a cubic-lattice PEPS with bond weights. A run is
converged once the mean singular-value change of the bond weights drops below `tol`.

## Fields

- `dt::Float64` : imaginary time step.
- `tol::Float64` : convergence tolerance on the bond weights.
- `maxiter::Int` : maximum number of sweeps.
- `trscheme::MatrixAlgebraKit.TruncationStrategy` : bond truncation after each gate application.
- `normalize_vertices::Bool` : rescale every vertex tensor after each bond update. This has
  no physical effect but avoids over/underflow during long evolutions.
"""
struct SimpleUpdate3D
    dt::Float64
    tol::Float64
    maxiter::Int
    trscheme::MatrixAlgebraKit.TruncationStrategy
    normalize_vertices::Bool
end
function SimpleUpdate3D(
    dt::Number,
    tol::Number,
    maxiter::Int,
    trscheme::MatrixAlgebraKit.TruncationStrategy;
    normalize_vertices::Bool=true,
)
    return SimpleUpdate3D(Float64(dt), Float64(tol), maxiter, trscheme, normalize_vertices)
end

"""
    _su_bond!(site, axis, gate, peps, alg)

Simple update of the bond connecting `site` to its neighbour in the positive direction along
`axis`. The two vertex tensors and the bond weight of `peps` are updated in place; the
truncation error of the bond SVD is returned.
"""
function _su_bond!(
    site::CartesianIndex{3},
    axis::Int,
    gate::AbstractTensorMap{<:Any,S,2,2},
    peps::InfiniteWeightPEPS3D,
    alg::SimpleUpdate3D,
) where {S<:ElementarySpace}
    sz = size(peps)
    @assert all(1 .<= Tuple(site) .<= sz)
    dir = _posdir3D(axis)
    dirop = _opposite3D(dir)
    site2 = _shift_site(site, axis, 1, sz)

    # absorb the environment weights; the shared bond gets a square root on either side
    alldirs = ntuple(identity, 6)
    A = _absorb_weights3D(
        peps.vertices[site], peps.weights, site, alldirs, ntuple(==(dir), 6), false
    )
    B = _absorb_weights3D(
        peps.vertices[site2], peps.weights, site2, alldirs, ntuple(==(dirop), 6), false
    )

    # apply the gate to the reduced tensors and truncate the bond
    X, a, b, Y = _qr_bond3D(A, B, dir)
    a, s, b, ϵ = _apply_gate3D(a, b, gate, alg.trscheme)
    A, B = _qr_bond_undo3D(X, a, b, Y, dir)

    # remove the environment weights again
    otherA = Tuple(filter(!=(dir), 1:6))
    otherB = Tuple(filter(!=(dirop), 1:6))
    allfalse = ntuple(Returns(false), 5)
    A = _absorb_weights3D(A, peps.weights, site, otherA, allfalse, true)
    B = _absorb_weights3D(B, peps.weights, site2, otherB, allfalse, true)

    if alg.normalize_vertices
        A = A / norm(A, Inf)
        B = B / norm(B, Inf)
    end
    peps.vertices[site], peps.vertices[site2] = A, B
    peps.weights[axis, site] = s / norm(s, Inf)
    return ϵ
end

"""
    _bond_order(sz::NTuple{3,Int})

All bonds of the unit cell as `(axis, site)` pairs, grouped per axis.
"""
function _bond_order(sz::NTuple{3,Int})
    return [(axis, site) for axis in 1:3 for site in CartesianIndices(sz)]
end

"""
    su_iter3D(gates::CubicHamiltonian, peps::InfiniteWeightPEPS3D, alg::SimpleUpdate3D;
              symmetric::Bool=true)

One simple-update sweep: apply the two-site `gates` to every bond of the unit cell.

With `symmetric = true` the bonds are swept forwards and then backwards, which makes the
sweep a second-order Trotter step -- provided `gates` were built with half the time step.
"""
function su_iter3D(
    gates::CubicHamiltonian,
    peps::InfiniteWeightPEPS3D,
    alg::SimpleUpdate3D;
    symmetric::Bool=true,
)
    @assert size(gates) == size(peps)
    peps = deepcopy(peps)
    order = _bond_order(size(peps))
    symmetric && (order = [order; reverse(order)])
    ϵ = 0.0
    for (axis, site) in order
        ϵ = max(ϵ, _su_bond!(site, axis, gates[axis, site], peps, alg))
    end
    return peps, ϵ
end

"""
    simpleupdate3D(peps::InfiniteWeightPEPS3D, H::CubicHamiltonian, alg::SimpleUpdate3D;
                   symmetric::Bool=true, check_interval::Int=100, verbosity::Int=1)

Evolve `peps` in imaginary time with the nearest-neighbour cubic-lattice Hamiltonian `H`
using simple update. Returns `(peps, wtdiff)`, where `wtdiff` is the mean singular-value
change of the bond weights in the last sweep.

With `symmetric = true` every sweep is a second-order Trotter step: the gates are built with
`alg.dt / 2` and the bonds are swept forwards and backwards.

Initialize `peps` with [`product_peps3D`](@ref) rather than with random tensors: a random
cubic-lattice PEPS carries close to maximal junk entanglement on all six virtual legs, which
simple update at fixed bond dimension cannot shed, and the evolution then stalls far above
the ground-state energy.
"""
function simpleupdate3D(
    peps::InfiniteWeightPEPS3D,
    H::AbstractCubicHamiltonian,
    alg::SimpleUpdate3D;
    symmetric::Bool=true,
    check_interval::Int=100,
    verbosity::Int=1,
)
    @assert size(H) == size(peps)
    time_start = time()
    gates = get_expham(symmetric ? alg.dt / 2 : alg.dt, H)
    wtdiff = 1.0
    wts0 = deepcopy(peps.weights)
    for count in 1:(alg.maxiter)
        time0 = time()
        peps, _ = su_iter3D(gates, peps, alg; symmetric)
        wtdiff = compare_weights(peps.weights, wts0)
        converged = wtdiff < alg.tol
        cancel = count == alg.maxiter
        wts0 = deepcopy(peps.weights)
        time1 = time()
        show_info =
            verbosity > 0 &&
            (count == 1 || count % check_interval == 0 || converged || cancel)
        if show_info
            label = converged ? "conv" : (cancel ? "cancel" : "iter")
            elapsed = time1 - ((converged || cancel) ? time_start : time0)
            message =
                "SU3D $label $count: dt = $(alg.dt), weight diff = $(wtdiff), " *
                "time = $(round(elapsed; digits=3)) sec"
            (cancel && !converged) ? (@warn message) : (@info message)
        end
        converged && break
    end
    return peps, wtdiff
end

"""
    simpleupdate3D(peps, H, dts::AbstractVector; tol=1e-8, maxiter=1000, trscheme, kwargs...)

Run consecutive simple-update stages with decreasing time steps `dts`, which is the usual
way of converging a simple-update ground state.
"""
function simpleupdate3D(
    peps::InfiniteWeightPEPS3D,
    H::AbstractCubicHamiltonian,
    dts::AbstractVector{<:Number};
    tol::Number=1e-8,
    maxiter::Int=1000,
    trscheme::MatrixAlgebraKit.TruncationStrategy=FixedSpaceTruncation(),
    normalize_vertices::Bool=true,
    kwargs...,
)
    wtdiff = 1.0
    for dt in dts
        alg = SimpleUpdate3D(dt, tol, maxiter, trscheme; normalize_vertices)
        peps, wtdiff = simpleupdate3D(peps, H, alg; kwargs...)
    end
    return peps, wtdiff
end

"""
    canonicalize3D(peps::InfiniteWeightPEPS3D; tol=1e-12, maxiter=10000, verbosity=0)

Bring `peps` into simple-update canonical form by sweeping identity gates over all bonds
until the bond weights stop changing (up to `tol`). The represented state is unchanged as
long as no truncation occurs; only the gauge, and hence the bond weights, are updated. The
fixed point is the belief-propagation fixed point of the norm network, i.e. the environment
that the simple-update observables below rely on.

During an imaginary-time evolution the weights are only re-canonicalized locally after each
gate, so they lag behind the evolving state. Call this before computing observables from the
bond weights.
"""
function canonicalize3D(
    peps::InfiniteWeightPEPS3D; tol::Real=1e-12, maxiter::Int=10000, verbosity::Int=0
)
    gates = map(Iterators.product(1:3, axes(peps.vertices)...)) do (axis, x, y, z)
        site = CartesianIndex(x, y, z)
        site2 = _shift_site(site, axis, 1, size(peps))
        P1, P2 = space(peps.vertices[site], 1), space(peps.vertices[site2], 1)
        return id(scalartype(peps), P1 ⊗ P2)
    end
    alg = SimpleUpdate3D(0.0, tol, 1, FixedSpaceTruncation())
    wts0 = deepcopy(peps.weights)
    wtdiff = Inf
    for count in 1:maxiter
        peps, _ = su_iter3D(CubicHamiltonian(collect(gates)), peps, alg)
        wtdiff = compare_weights(peps.weights, wts0)
        wts0 = deepcopy(peps.weights)
        if wtdiff < tol
            verbosity > 0 && @info "canonicalize3D conv $count: weight diff = $wtdiff"
            return peps
        end
    end
    @warn "canonicalize3D cancel $maxiter: weight diff = $wtdiff"
    return peps
end

#
## Simple-update level observables
#
# These use the bond weights as a stand-in for the environment, which is the same
# approximation that the update itself makes. They are cheap and good enough to monitor
# convergence; for accurate numbers, contract the `InfinitePEPS3D` with a proper 3D
# contraction scheme (e.g. the CTMRG3D algorithms in this package).
#

"""
    onesite_rdm(peps::InfiniteWeightPEPS3D, site::CartesianIndex{3})

Single-site reduced density matrix in the simple-update approximation, normalized to unit
trace.
"""
function onesite_rdm(peps::InfiniteWeightPEPS3D, site::CartesianIndex{3})
    A = _absorb_weights3D(
        peps.vertices[site],
        peps.weights,
        site,
        ntuple(identity, 6),
        ntuple(Returns(false), 6),
        false,
    )
    @tensor ρ[-1; -2] := A[-1; 1 2 3 4 5 6] * conj(A[-2; 1 2 3 4 5 6])
    return ρ / tr(ρ)
end

"""
    bond_rdm(peps::InfiniteWeightPEPS3D, site::CartesianIndex{3}, axis::Int)

Two-site reduced density matrix of the bond between `site` and its neighbour in the positive
direction along `axis`, in the simple-update approximation. Normalized to unit trace.
"""
function bond_rdm(peps::InfiniteWeightPEPS3D, site::CartesianIndex{3}, axis::Int)
    sz = size(peps)
    dir = _posdir3D(axis)
    dirop = _opposite3D(dir)
    site2 = _shift_site(site, axis, 1, sz)
    alldirs = ntuple(identity, 6)
    A = _absorb_weights3D(
        peps.vertices[site], peps.weights, site, alldirs, ntuple(==(dir), 6), false
    )
    B = _absorb_weights3D(
        peps.vertices[site2], peps.weights, site2, alldirs, ntuple(==(dirop), 6), false
    )
    # the environment legs are traced between ket and bra; only the shared bond links A to B.
    # the labels are chosen such that ncon contracts A-A†, then B-B†, then the two halves.
    idxA, idxAc = Vector{Int}(undef, 7), Vector{Int}(undef, 7)
    idxB, idxBc = Vector{Int}(undef, 7), Vector{Int}(undef, 7)
    idxA[1], idxAc[1], idxB[1], idxBc[1] = -1, -3, -2, -4
    envA, envB = 0, 5
    for d in 1:6
        if d == dir
            idxA[1 + d], idxAc[1 + d] = 11, 12
        else
            envA += 1
            idxA[1 + d] = idxAc[1 + d] = envA
        end
        if d == dirop
            idxB[1 + d], idxBc[1 + d] = 11, 12
        else
            envB += 1
            idxB[1 + d] = idxBc[1 + d] = envB
        end
    end
    ρ = ncon([A, A, B, B], [idxA, idxAc, idxB, idxBc], [false, true, false, true])
    ρ = permute(ρ, ((1, 2), (3, 4)))
    return ρ / tr(ρ)
end

"""
    expectation_value(peps::InfiniteWeightPEPS3D, O::AbstractTensorMap)

Simple-update estimate of the expectation value of a one- or two-site operator `O`, averaged
over the unit cell (and, for two-site operators, over the three bond directions).
"""
function MPSKit.expectation_value(peps::InfiniteWeightPEPS3D, O::AbstractTensorMap)
    if numin(O) == 1
        vals = [tr(onesite_rdm(peps, site) * O) for site in CartesianIndices(peps.vertices)]
        return sum(vals) / length(vals)
    elseif numin(O) == 2
        vals = [
            tr(bond_rdm(peps, site, axis) * O) for
            site in CartesianIndices(peps.vertices), axis in 1:3
        ]
        return sum(vals) / length(vals)
    else
        throw(ArgumentError("Only one- and two-site operators are supported."))
    end
end

"""
    energy_per_site(peps::InfiniteWeightPEPS3D, H::CubicHamiltonian)

Simple-update estimate of the energy per site: the sum of all bond energies in the unit cell
divided by the number of sites. On-site terms folded into the bond terms by
[`CubicHamiltonian`](@ref) are thereby counted correctly.
"""
function energy_per_site(peps::InfiniteWeightPEPS3D, H::CubicHamiltonian)
    @assert size(H) == size(peps)
    sites = CartesianIndices(peps.vertices)
    E = sum(tr(bond_rdm(peps, site, axis) * H[axis, site]) for site in sites, axis in 1:3)
    return E / length(sites)
end

#
## Finite-temperature observables in the simple-update approximation
#
# A density-operator PEPS evolved with a `ThermalCubicHamiltonian` up to imaginary time `τ`
# represents ρ(τ) = e^{-τH}. Reading its fused ket-bra leg as (physical, ancilla), it is a
# purification of the thermal state at inverse temperature `β = 2τ`:
#
#     ⟨ρ(τ)| O ⊗ 𝟙 |ρ(τ)⟩ / ⟨ρ(τ)|ρ(τ)⟩ = Tr[O e^{-2τH}] / Tr[e^{-2τH}].
#
# The norm network ⟨ρ(τ)|ρ(τ)⟩ is an ordinary double-layer PEPS network, so the bond weights
# provide the same (Bethe-lattice/belief-propagation) environment as for a ground state. The
# functions below compute the reduced density matrices of the purification on the fused leg
# and trace out the ancilla. This is the same state that `cubic_pepo_purified` hands to the
# CTMRG3D machinery, but without any boundary contraction.
#

"""Trace out the ancilla (bra) part of every fused ket-bra leg of the fused-leg operator `R`."""
function _trace_ancilla(dm::DensityMatrixSpace, R::AbstractTensorMap{<:Any,S,1,1}) where {S}
    R4 = dm.U * R * dm.U'
    @tensor ρ[-1; -2] := R4[-1 1; -2 1]
    return ρ
end
function _trace_ancilla(dm::DensityMatrixSpace, R::AbstractTensorMap{<:Any,S,2,2}) where {S}
    UU = dm.U ⊗ dm.U
    R8 = UU * R * UU'
    @tensor ρ[-1 -2; -3 -4] := R8[-1 1 -2 2; -3 1 -4 2]
    return ρ
end

"""
    onesite_rdm(peps::InfiniteWeightPEPS3D, dm::DensityMatrixSpace, site::CartesianIndex{3})

Single-site physical reduced density matrix of the thermal state at `β = 2τ`, for a
density-operator PEPS evolved up to imaginary time `τ` (see [`ThermalCubicHamiltonian`](@ref)),
in the simple-update approximation. Normalized to unit trace.
"""
function onesite_rdm(peps::InfiniteWeightPEPS3D, dm::DensityMatrixSpace, site::CartesianIndex{3})
    ρ = _trace_ancilla(dm, onesite_rdm(peps, site))
    return ρ / tr(ρ)
end

"""
    bond_rdm(peps::InfiniteWeightPEPS3D, dm::DensityMatrixSpace, site::CartesianIndex{3}, axis::Int)

Two-site physical reduced density matrix of the thermal state at `β = 2τ` on the bond between
`site` and its neighbour in the positive direction along `axis`, in the simple-update
approximation. See [`onesite_rdm`](@ref) for the conventions. Normalized to unit trace.
"""
function bond_rdm(
    peps::InfiniteWeightPEPS3D, dm::DensityMatrixSpace, site::CartesianIndex{3}, axis::Int
)
    ρ = _trace_ancilla(dm, bond_rdm(peps, site, axis))
    return ρ / tr(ρ)
end

"""
    expectation_value(peps::InfiniteWeightPEPS3D, dm::DensityMatrixSpace, O::AbstractTensorMap)

Simple-update estimate of the thermal expectation value `Tr[O e^{-βH}] / Tr[e^{-βH}]` at
`β = 2τ` of a one- or two-site physical operator `O`, for a density-operator PEPS evolved up
to imaginary time `τ`. Averaged over the unit cell (and, for two-site operators, over the
three bond directions).
"""
function MPSKit.expectation_value(
    peps::InfiniteWeightPEPS3D, dm::DensityMatrixSpace, O::AbstractTensorMap
)
    sites = CartesianIndices(peps.vertices)
    if numin(O) == 1
        vals = [tr(onesite_rdm(peps, dm, site) * O) for site in sites]
    elseif numin(O) == 2
        vals = [tr(bond_rdm(peps, dm, site, axis) * O) for site in sites, axis in 1:3]
    else
        throw(ArgumentError("Only one- and two-site operators are supported."))
    end
    return sum(vals) / length(vals)
end

"""
    energy_per_site(peps::InfiniteWeightPEPS3D, H::ThermalCubicHamiltonian)

Simple-update estimate of the thermal energy per site at `β = 2τ`, for a density-operator
PEPS evolved with `H` up to imaginary time `τ`.
"""
function energy_per_site(peps::InfiniteWeightPEPS3D, H::ThermalCubicHamiltonian)
    @assert size(H) == size(peps)
    sites = CartesianIndices(peps.vertices)
    E = sum(
        tr(bond_rdm(peps, H.dm, site, axis) * H[axis, site]) for site in sites, axis in 1:3
    )
    return E / length(sites)
end
