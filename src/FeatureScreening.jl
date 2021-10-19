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
using FeatureScreening.Utilities: ExpStep, make_rng, partition, @dump

include("importance.jl")

# Feature set related
include("Types.jl")
using FeatureScreening.Types: FeatureSet
using FeatureScreening.Types: names, load, save

# API dependencies
using Random: AbstractRNG, GLOBAL_RNG, shuffle as __shuffle
using ProgressMeter: @showprogress
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

function screen(feature_set::FeatureSet{L, N, F};
                reduced_size::Integer       = size(feature_set, 2) ÷ 5,
                step_size::Integer          = size(feature_set, 2) ÷ 10,
                config::NamedTuple          = DEFAULT_SCREEN_CONFIG,
                shuffle::Bool               = false,
                before::Function            = skip,
                after::Function             = skip,
                rng::Union{AbstractRNG, Integer} = GLOBAL_RNG
               )::FeatureSet{L, N, F} where {L, N, F}
    all::Vector{N} = names(feature_set)
    if shuffle
        __shuffle(make_rng(rng), all)
    end

    parts = partition(all, step_size; rest = true)
    selected::FeatureSet = feature_set[:, N[]]

    @showprogress "Screen" for (i, part) in enumerate(parts)
        new::FeatureSet = feature_set[:, part]

        # Before the computation
        before(selected, new)

        to_be_selected::FeatureSet = merge(selected, new)

        @dump "importances.$i.csv" importances::Vector{Pair{N, <: Real}} =
            feature_importance(to_be_selected; config, rng)

        important_names::Vector{<: N} =
            importants(importances; count = reduced_size)

        selected = to_be_selected[:, important_names]

        # After the computation
        after(selected)
    end

    return selected
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
