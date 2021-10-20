###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

module FeatureScreening

###=============================================================================
### Exports
###=============================================================================

# API
export screen

# Types
export FeatureSet

# Types API
export load, save, labels, names, features

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
using FeatureScreening.Utilities: nfoldCV_forest, skip
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

"""
    screen(feature_set...; configuration...)::FeatureSet

Screening input feature set by the given configuration. This function produces a
new `FeatureSet` despite the type of the input feature set.

**WARNING**: This function can run for a long time, elapsing time is
proportional to the number of features and depends on the internal ranking
random forest configuration.

**DISCLAIMER**: Internal ranking, importance computation is random forest based.
Potential future improvement is to add some other ranking methods.

# Steps:
0. Optionally shuffle the features,
1. slice them into (fixed size, disjoint) partitions,
2. then iterate over the partitions

    1. to compute importances from the given partition,
    2. and hold the bests,
    3. which will be assigned to the next partition (GOTO 2.1).

3. Finally returns all the remaining bests when all the partitions were
   processed.
"""
function screen(feature_set...; kwargs...)
    return screen(FeatureSet(feature_set...); kwargs...)
end

"""
    screen(feature_set::FeatureSet;
           reduced_size::Integer            = size(feature_set, 2) ÷ 5,
           step_size::Integer               = size(feature_set, 2) ÷ 10,
           config::NamedTuple               = DEFAULT_SCREEN_CONFIG,
           shuffle::Bool                    = false,
           before::Function                 = skip,
           after::Function                  = skip,
           rng::Union{AbstractRNG, Integer} = GLOBAL_RNG
          )::FeatureSet

# Parameters:
- `reduced_size`: Expected number of screened features (Right now, this is an
  upper bound).
- `step_size`: Size of each partition.
- `config`: Random forset configuration of the importance computing.
- `shuffle`: Flag to shuffle features before partition, to randomize the order
  of the features.
- `before`: Callback function, that executes before importance computation and
  selection part. Inputs are the previously selected features and the actual
  partition, output will be ignored.
- `after`: Callback function, that executes after importance computation and
  selection part. Inputs are the selected features, output will be ignored.
- `rng`: Random generator or a seed.
"""
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

end # module
