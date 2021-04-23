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
export AbstractFeatureSet, FeatureSet, FeatureSubset

# Types API
export feature_names

###=============================================================================
### Imports
###=============================================================================

include("importance.jl")

# Utilities
include("Utilities.jl")
using FeatureScreening.Utilities: ExpStep, partition

# Feature set related
include("Types.jl")
using FeatureScreening.Types: AbstractFeatureSet, FeatureSet, FeatureSubset
using FeatureScreening.Types: by_labels, feature_names, mtx, feature_count

# API dependencies
using Base.Iterators: Enumerate
using DecisionTree: nfoldCV_forest
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

function screen(features;
                reduced_size::Integer       = size(features, 2) % 5,
                step_size::Integer          = size(features, 2) % 10,
                starters::AbstractVector    = [],
                config::NamedTuple          = DEFAULT_SCREEN_CONFIG,
                before::Function            = skip,
                after::Function             = skip)
    iterator::Enumerate = enumerate(partition(features, step_size; rest = true))

    return foldl(iterator; init = features[starters]) do features, (i, new)
        @info "Turn #$(i)"

        # Before the computation
        before(features, new)

        @debug "Select" from = features plus = new
        merge!(features, new)

        features = select_features(features;
                                   count = reduced_size,
                                   config = config)

        # After the computation
        after(features)

        @debug "Selected features" selected = feature_names(features)

        return features
    end
end

function select_features(features::AbstractFeatureSet{L, N, F};
                         count = count,
                         config = config
                        )::AbstractFeatureSet{L, N, F} where {L, N, F}
    importances::Vector{Pair{N, <: Real}} =
        feature_importance(features; config = config)

    return features[importants(importances; count = count)]
end

function accuracy(features::AbstractFeatureSet;
                  config::NamedTuple = (;)
                 )::Float64
    accuracies::Vector{Float64} =
        nfoldCV_forest(features; config, verbose = false)
    accuracy::Float64 = mean(accuracies)
    @info "$(length(accuracies))-fold CV" accuracies mean_accuracy=accuracy
    return accuracy
end

function accuracies(features::AbstractFeatureSet{L, N, F};
                    step = ExpStep(2),
                    config::NamedTuple = (;)
                   )::Vector{Pair{Int, Float64}} where {L, N, F}
    fns::Vector{N} = feature_names(features)
    return [n => accuracy(features[fns[1:n]]; config = config)
            for n in 1:step:length(fns)]
end

function accuracies(; kwargs...)::Function
    return function (features)
        return accuracies(features; kwargs...)
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
