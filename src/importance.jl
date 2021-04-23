###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using DecisionTree:
    build_forest,
    RandomForestRegressor,
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

# TODO feature indices
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

using DecisionTree: Node, Leaf

function fold(f, node::Node; init)
    init = f(init, node)
    init = fold(f, node.left; init = init)
    init = fold(f, node.right; init = init)
    return init
end

function fold(f, leaf::Leaf; init)
    return f(init, leaf)
end

# TODO feature indices
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
    if !haskey(occurrences, node.featid)
        occurrences[node.featid] = 0
    end
    occurrences[node.featid] += 1
    return occurrences
end

function accumulate_id!(occurrences::Dict{Int, Int}, ::Leaf)::Dict{Int, Int}
    return occurrences
end
