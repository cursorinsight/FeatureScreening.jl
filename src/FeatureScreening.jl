###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

module FeatureScreening

###=============================================================================
### Exports
###=============================================================================

# Types
export FeatureSet

# Types API
export names, load, save

###=============================================================================
### Imports
###=============================================================================

# Utilities
include("Utilities.jl")
using FeatureScreening.Utilities: ExpStep, partition

include("importance.jl")

# Feature set related
include("Types.jl")
using FeatureScreening.Types: FeatureSet
using FeatureScreening.Types: names, load, save

# API dependencies
using Base.Iterators: Enumerate
using FeatureScreening.Utilities: nfoldCV_forest
using Statistics: mean

###=============================================================================
### API
###=============================================================================

const DEFAULT_SCREEN_CONFIG =
    (n_subfeatures          = -1,
     n_trees                = 1000,
     partial_sampling       = 0.9,
     max_depth              = -1,
     min_samples_leaf       = 10,
     min_samples_split      = 10,
     min_purity_increase    = 0.0)

function screen(feature_set...; kwargs...)
    return screen(FeatureSet(feature_set...); kwargs...)
end

function screen(feature_set::FeatureSet;
                reduced_size::Integer       = size(feature_set, 2) ÷ 5,
                step_size::Integer          = size(feature_set, 2) ÷ 10,
                starters::AbstractVector    = [],
                config::NamedTuple          = DEFAULT_SCREEN_CONFIG,
                before::Function            = skip,
                after::Function             = skip
               )::FeatureSet
    parts = partition(names(feature_set), step_size; rest = true)
    init = feature_set[:, starters]

    return foldl(enumerate(parts); init) do selected, (i, part)
        new = feature_set[:, part]
        @info "Turn #$(i)"

        # Before the computation
        before(selected, new)

        @debug "Select" from = features plus = new

        to_be_selected::FeatureSet = merge(selected, new)

        selected = select_features(to_be_selected;
                                   count = reduced_size,
                                   config = config)

        # After the computation
        after(selected)

        @debug "Selected features" selected = names(selected)

        return selected
    end
end

function select_features(features::FeatureSet{L, N, F};
                         count::Int = 5,
                         config::NamedTuple = (;)
                        )::FeatureSet{L, N, F} where {L, N, F}
    importances::Vector{Pair{N, <: Real}} =
        feature_importance(features; config = config)

    return features[:, importants(importances; count = count)]
end

function accuracy(features::FeatureSet;
                  config::NamedTuple = (;),
                  verbose = false
                 )::Vector{Float64}
    return nfoldCV_forest(features; config, verbose)
end

# TODO
function accuracy(; kwargs...)::Function
    return function (features)
        return accuracy(features; kwargs...)
    end
end

# TODO
function importants(importances::Vector{<: Pair{N}};
                    count::Int = 1
                   )::Vector{N} where {N}
    return first.(importances[1:min(end, count)])
end

# TODO remove
function skip(args...; kwargs...)::Nothing
    return nothing
end

end # module
