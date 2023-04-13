###-----------------------------------------------------------------------------
### Copyright (C) FeatureScreening.jl
###
### SPDX-License-Identifier: MIT License
###-----------------------------------------------------------------------------

module Utilities

###=============================================================================
### Imports
###=============================================================================

using MacroTools: combinedef, rmlines, splitdef

# `DecisionTree` wrappers
using DecisionTree: Ensemble as RandomForest
using DecisionTree: build_forest, nfoldCV_forest

# Rest
using Random: AbstractRNG, GLOBAL_RNG, MersenneTwister

###=============================================================================
### API
###=============================================================================

macro unimplemented(function_definition::Expr)
    def = splitdef(function_definition)
    @assert(rmlines(def[:body]) == Expr(:block), "Function definition of " *
        "@unimplemented $(def[:name]) has a non-empty body!")
    def[:body] = :(error("Unimplemented method: ",
                         $(string(function_definition.args[1]))))
    return esc(combinedef(def))
end

###-----------------------------------------------------------------------------
### `DecisionTree` wrappers
###-----------------------------------------------------------------------------

const DEFAULT_BUILD_FOREST_CONFIG =
    (n_subfeatures          = -1,
     n_trees                = 10,
     partial_sampling       = 0.7,
     max_depth              = -1,
     min_samples_leaf       = 1,
     min_samples_split      = 2,
     min_purity_increase    = 0.0)

function __build_forest(labels::AbstractVector{L},
                        features::AbstractMatrix{F};
                        config::NamedTuple = (;),
                        kwargs...
                       )::RandomForest{F, L} where {L, F}
    config::NamedTuple = (; DEFAULT_BUILD_FOREST_CONFIG..., config...)
    return build_forest(labels,
                        features,
                        config.n_subfeatures,
                        config.n_trees,
                        config.partial_sampling,
                        config.max_depth,
                        config.min_samples_leaf,
                        config.min_samples_split,
                        config.min_purity_increase;
                        kwargs...)
end

const DEFAULT_NFOLDCV_FOREST_CONFIG =
    (n_folds                = 4,
     n_subfeatures          = -1,
     n_trees                = 10,
     partial_sampling       = 0.7,
     max_depth              = -1,
     min_samples_leaf       = 1,
     min_samples_split      = 2,
     min_purity_increase    = 0.0)

function __nfoldCV_forest(labels::AbstractVector,
                          features::AbstractMatrix;
                          config::NamedTuple = (;),
                          kwargs...)
    config::NamedTuple = (; DEFAULT_NFOLDCV_FOREST_CONFIG..., config...)
    return nfoldCV_forest(labels,
                          features,
                          config.n_folds,
                          config.n_subfeatures,
                          config.n_trees,
                          config.partial_sampling,
                          config.max_depth,
                          config.min_samples_leaf,
                          config.min_samples_split,
                          config.min_purity_increase;
                          kwargs...)
end

###-----------------------------------------------------------------------------
### Rest
###-----------------------------------------------------------------------------

const Maybe{T} = Union{Nothing, T}

function make_rng(rng::AbstractRNG = GLOBAL_RNG)::AbstractRNG
    return rng
end

function make_rng(seed::Integer)::AbstractRNG
    return MersenneTwister(seed)
end

end # module
