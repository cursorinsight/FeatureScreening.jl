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
    feature_importance(features...; config = ())

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
abstract type SelectorMethod end

function select end

struct Top{T <: Real} <: SelectorMethod
    size::T
end

function select(feature_importances::Vector{<: Pair{L}},
                top::Top;
                strict::Bool = true
               )::Vector{L} where {L}
    selected_top::Integer = get_count(feature_importances, top.size; strict)
    return label.(feature_importances[begin:selected_top])
end

struct Random{T <: Real} <: SelectorMethod
    size::T
end

function select(feature_importances::Vector{<: Pair{L}},
                random::Random;
                strict::Bool = true
               )::Vector{L} where {L}
    selected_random::Integer =
        get_count(feature_importances, random.size; strict)
    random_feature_importances::Vector =
        rand(feature_importances, selected_random)
    sort!(random_feature_importances; rev = true, by = importance)
    return label.(random_feature_importances)
end

struct IndexBased <: SelectorMethod
    indices::AbstractVector{<: Integer}
end

function select(feature_importances::Vector{<: Pair{L}},
                selector::IndexBased;
                strict::Bool = true
               )::Vector{L} where {L}
    indices::AbstractVector{<: Integer} =
        if strict
            @assert selector.indices ⊆ keys(feature_importances)
            selector.indices
        else
            filter(selector.indices) do idx
                return idx in keys(feature_importances)
            end
        end
    return label.(feature_importances[indices])
end

##------------------------------------------------------------------------------
## Get count
##------------------------------------------------------------------------------

function get_count(itr::AbstractVector,
                   count::Integer;
                   strict::Bool = true
                  )::Integer
    if strict
        @assert 0 <= count <= length(itr)
    else
        count = clamp(count, 0, length(itr))
    end
    return count
end

function get_count(itr::AbstractVector,
                   ratio::Real;
                   strict::Bool = true
                  )::Int
    if strict
        @assert 0.0 <= ratio <= 1.0
    else
        ratio = clamp(ratio, 0.0, 1.0)
    end
    return floor(Int, length(itr) * ratio)
end

##------------------------------------------------------------------------------
## Getters
##------------------------------------------------------------------------------

function label(feature_importance::Pair{L, I})::L where {L, I}
    (label::L, _) = feature_importance
    return label
end

function importance(feature_importance::Pair{L, I})::I where {L, I}
    (_, importance::I) = feature_importance
    return importance
end
