###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening.Utilities:
    build_forest

using DecisionTree:
    Ensemble as RandomForest,
    Node,
    Leaf

using Random:
    AbstractRNG,
    GLOBAL_RNG

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

# TODO https://github.com/cursorinsight/FeatureScreening.jl/issues/18
"""
    feature_importance(features...; config = (;))

- `features` are the `build_forest` arguments
- `config` is the configuration of the `build_forest` call

Rank feature importances based on their frequency in a random forest.

Algorithm:
  1. Build a random forest.
  2. Iterate over all the trees of this forest to count occurrences of each
     features.
  3. Sort features by their importance (frequency).
"""
function feature_importance(features...;
                            config::NamedTuple = (;),
                            kwargs...
                           )::Vector{Pair{Int, Int}}
    config::NamedTuple =
        (; DEFAULT_CONFIG_FOR_FEATURE_IMPORTANCE..., config...)

    # 1. step
    forest::RandomForest = build_forest(features...; config, kwargs...)

    # 2-3. step
    importances::Vector{Pair{Int, Int}} = feature_importance(forest)

    return importances
end

function fold(f, node::Node; init)
    init = f(init, node)
    init = fold(f, node.left; init = init)
    init = fold(f, node.right; init = init)
    return init
end

function fold(f, leaf::Leaf; init)
    return f(init, leaf)
end

"""
    feature_importance(forest::RandomForest)

This function contains only the 2. and 3. steps.
"""
function feature_importance(forest::RandomForest)::Vector{Pair{Int, Int}}
    # 2. step
    occurrences::Dict{Int, Int} =
        foldl(forest.trees;
              init = Dict{Int, Int}()) do occurrences, tree
            return fold(accumulate_id!, tree; init = occurrences)
        end

    # 3. step
    importances::Vector{Pair{Int, Int}} =
        sort!(collect(occurrences); by = last, rev = true)

    return importances
end

function accumulate_id!(occurrences::Dict{Int, Int}, node::Node)::Dict{Int, Int}
    occurrences[node.featid] = get!(occurrences, node.featid, 0) + 1;
    return occurrences
end

function accumulate_id!(occurrences::Dict{Int, Int}, ::Leaf)::Dict{Int, Int}
    return occurrences
end

###-----------------------------------------------------------------------------
### Selection
###-----------------------------------------------------------------------------

# TODO https://github.com/cursorinsight/FeatureScreening.jl/issues/20
abstract type Selector end

"""
    select(collection::AbstractVector, selector::Selector; strict::Bool = true)

# Description
Select by a specific selector from to given collection.

# Parameters
- `collection`
- `selector`
- `strict`: flag to check the validity of the selection process, if `false` then
  try to fix, otherwise let it crush
"""
function select end

##------------------------------------------------------------------------------
## Top selector
##------------------------------------------------------------------------------

"""
    Top(size::Int)

Select the top `size` part of the collection.

    Top(ratio::Real)

Select the top `ratio` portion of the collection.
"""
struct Top{T <: Real} <: Selector
    size::T
end

function select(collection::AbstractVector{T},
                selector::Top;
                strict::Bool = true
               )::Vector{T} where {T}
    count::Integer = get_count(collection, selector.size; strict)
    return collection[begin:count]
end

##------------------------------------------------------------------------------
## Random selector
##------------------------------------------------------------------------------

"""
    Random(size::Int)

Select the `size` part randomly from the collection.

    Random(ratio::Real[, rng::AbstractRNG])

Select the `ratio` portion randomly from the collection, using `rng`.
"""
struct Random{T <: Real, R <: AbstractRNG} <: Selector
    size::T
    rng::R
end

Random(size::Real) = Random(size, GLOBAL_RNG)

function select(collection::AbstractVector{T},
                selector::Random;
                strict::Bool = true
               )::Vector{T} where {T}
    count::Integer = get_count(collection, selector.size; strict)
    return rand(selector.rng, collection, count)
end

##------------------------------------------------------------------------------
## Index based selector
##------------------------------------------------------------------------------

"""
    IndexBased(indices::AbstractVector{T})

Select simply by the given `indices`.
"""
struct IndexBased{T} <: Selector
    indices::AbstractVector{T}
end

function select(collection::AbstractVector{T},
                selector::IndexBased{I};
                strict::Bool = true
               )::AbstractVector{T} where {T, I}
    if strict
        @assert selector.indices ⊆ keys(collection)
        return collection[selector.indices]
    else
        return collection[selector.indices ∩ keys(collection)]
    end
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
