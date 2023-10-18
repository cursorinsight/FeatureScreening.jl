###-----------------------------------------------------------------------------
### Copyright (C) FeatureScreening.jl
###
### SPDX-License-Identifier: MIT License
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureSets: AbstractFeatureSet
using DecisionTree: Ensemble as RandomForest, Node, Leaf, build_forest
using Random: AbstractRNG, GLOBAL_RNG
using StatsBase: sample, weights

import Base: ∘, show, size

###=============================================================================
### Implementation
###=============================================================================

const DEFAULT_CONFIG_FOR_FEATURE_IMPORTANCE =
    (n_subfeatures        = -1,
     n_trees              = 100,
     partial_sampling     = 0.7,
     max_depth            = -1,
     min_samples_leaf     = 4,
     min_samples_split    = 4,
     min_purity_increase  = 0.1)

"""
    feature_importance(fs::AbstractFeatureSet; config = (;))

Rank feature importances based on their frequency in a random forest. Return a
vector of pairs, mapping feature names to importances.

Steps:
1. build a random forest;
2. iterate over the forest's trees to count the occurrences of every feature;
3. sort features by their importance (frequency).
"""
function feature_importance(feature_set::AbstractFeatureSet{_L, N};
                            config = (;),
                            kwargs...
                           )::Vector{Pair{N, Int}} where {_L, N}
    config::NamedTuple = merge(DEFAULT_CONFIG_FOR_FEATURE_IMPORTANCE, config)
    forest::RandomForest = build_forest(feature_set; config, kwargs...)
    return [names(feature_set)[i] => importance
            for (i, importance) in feature_importance(forest)]
end

"""
    feature_importance(forest::RandomForest)

Rank feature importances based on their frequency in a random `forest`. Return a
vector of pairs, mapping feature indices to importances.

Steps:
1. iterate over the forest's trees to count the occurrences of every feature;
2. sort features by their importance (frequency).
"""
function feature_importance(forest::RandomForest)::Vector{Pair{Int, Int}}
    occurrences::Dict{Int, Int} =
        fold(accumulate_id!, forest; init = Dict{Int, Int}())
    importances::Vector{Pair{Int, Int}} =
        sort!(collect(occurrences); by = last, rev = true)
    return importances
end

function fold(f, forest::RandomForest; init)
    return foldl(forest.trees; init) do occurrences, tree
        return fold(f, tree; init = occurrences)
    end
end

function fold(f, node::Node; init)
    init = f(init, node)
    init = fold(f, node.left; init)
    init = fold(f, node.right; init)
    return init
end

function fold(f, leaf::Leaf; init)
    return f(init, leaf)
end

function accumulate_id!(occurrences::Dict{Int, Int}, node::Node)::Dict{Int, Int}
    occurrences[node.featid] = get(occurrences, node.featid, 0) + 1
    return occurrences
end

function accumulate_id!(occurrences::Dict{Int, Int}, ::Leaf)::Dict{Int, Int}
    return occurrences
end

###-----------------------------------------------------------------------------
### Selection modes
###-----------------------------------------------------------------------------

abstract type SelectionMode end

name(mode::SelectionMode) = string(nameof(mode))

size(mode::SelectionMode) = mode.size

isstrict(mode::SelectionMode) = mode.strict

function select(collection, mode::SelectionMode)
    return select(GLOBAL_RNG, collection, mode)
end

function show(io::IO, mode::SelectionMode)
    print(io,
          name(mode),
          '(',
          size(mode),
          isstrict(mode) ? "" : "; strict = false",
          ')')
end

##------------------------------------------------------------------------------
## SelectTop
##------------------------------------------------------------------------------

"""
    SelectTop(size::Int)

Deterministically select the top `size` part of the collection.

---

    SelectTop(ratio::Real)

Deterministically select the top `ratio` portion of the collection.
"""
struct SelectTop{T <: Real} <: SelectionMode
    size::T
    strict::Bool

    function SelectTop(size::T; strict = true) where {T <: Real}
        return new{T}(size, strict)
    end
end

name(::SelectTop) = "SelectTop"

function select(::AbstractRNG,
                collection::AbstractVector{T},
                mode::SelectTop
               )::Vector{T} where {T}
    count = get_count(collection, mode.size; strict = mode.strict)
    return collection[begin:count]
end

##------------------------------------------------------------------------------
## SelectRandom
##------------------------------------------------------------------------------

"""
    SelectRandom(size::Int [, weights_fn])

Select a `size` part randomly from the collection. Selection probabilities are
proportional to weights.

---

    SelectRandom(ratio::Real [, weights_fn])

Select a `ratio` portion randomly from the collection. Selection probabilities
are proportional to weights.
"""
struct SelectRandom{F <: Function, T <: Real} <: SelectionMode
    weights::F
    size::T
    strict::Bool
    replace::Bool

    function SelectRandom(weights::F,
                           size::T;
                           strict::Bool = true,
                           replace::Bool = false) where {F <: Function,
                                                         T <: Real}
        return new{F, T}(weights, size, strict, replace)
    end
end

function SelectRandom(size::Real; kwargs...)
    return SelectRandom(unit_weights, size; kwargs...)
end

name(::SelectRandom{F}) where {F} = "SelectRandom{$F}"

function select(rng::AbstractRNG,
                collection::AbstractVector{T},
                mode::SelectRandom
               )::Vector{T} where {T}
    count = get_count(collection, mode.size; strict = mode.strict)
    return sample(rng,
                  collection,
                  weights(mode.weights(collection)),
                  count;
                  replace = mode.replace,
                  ordered = true)
end

unit_weights(v::AbstractVector) = fill(1, length(v))

##------------------------------------------------------------------------------
## SelectByImportance
##------------------------------------------------------------------------------

"""
A specialized, weighted `SelectRandom` using importances as weights.

`SelectByImportance` takes a vector of `feature => importance` pairs, and uses
the importance values as weights to perform a random selection of the pairs,
without replacement.
"""
function SelectByImportance(size::Real; strict::Bool = true)
    return SelectRandom(importances, size; replace = false, strict)
end

"""
    importances(feature_importances::AbstractVector{<: }Pair)::Vector

Return the vector of importances from vector of feature importance pairs.
"""
function importances(feature_importances::AbstractVector{<: Pair})::Vector
    return importance.(feature_importances)
end

name(::SelectRandom{typeof(importances)}) = "SelectByImportance"

##------------------------------------------------------------------------------
## ComposedSelectionMode
##------------------------------------------------------------------------------

struct ComposedSelectionMode{A <: SelectionMode,
                             B <: SelectionMode} <: SelectionMode
    a::A
    b::B
end

function select(rng::AbstractRNG,
                collection::AbstractVector,
                mode::ComposedSelectionMode)
    return select(rng, select(rng, collection, mode.b), mode.a)
end

∘(a::SelectionMode, b::SelectionMode) = ComposedSelectionMode(a, b)

function show(io::IO, mode::ComposedSelectionMode)
    print(io, "$(mode.a) ∘ $(mode.b)")
end

##------------------------------------------------------------------------------
## Get count
##------------------------------------------------------------------------------

"""
    get_count(collection::AbstractVector,
              count::Integer;
              strict::Bool = true
             )::Integer

# Description
Internal function for getting count. In strict mode, there is an explicit check
whether `count` is within bounds. Otherwise, it is clamped to be in range.
"""
function get_count(collection::AbstractVector,
                   count::Integer;
                   strict::Bool = true
                  )::Integer
    if strict
        @assert 0 <= count <= length(collection)
    else
        count = clamp(count, 0, length(collection))
    end

    return count
end

"""
    get_count(collection::AbstractVector,
              ratio::Real;
              strict::Bool = true
             )::Int

# Description
Internal function for getting percentage by the given `ratio` parameter. In
strict mode, there is an explicit check whether `ratio` is within bounds.
Otherwise, it is clamped to be in range.
"""
function get_count(collection::AbstractVector,
                   ratio::Real;
                   strict::Bool = true
                  )::Int
    if strict
        @assert 0.0 <= ratio <= 1.0
    else
        ratio = clamp(ratio, 0.0, 1.0)
    end
    return floor(Int, length(collection) * ratio)
end

##------------------------------------------------------------------------------
## Getters
##------------------------------------------------------------------------------

"""
    label(feature_importance::Pair{L, I})::L

Return the label from feature importance pair.
"""
function label(feature_importance::Pair{L, I})::L where {L, I}
    (label::L, _) = feature_importance
    return label
end

"""
    importance(feature_importance::Pair{L, I})::I

Return the importance from feature importance pair.
"""
function importance(feature_importance::Pair{L, I})::I where {L, I}
    (_, importance::I) = feature_importance
    return importance
end
