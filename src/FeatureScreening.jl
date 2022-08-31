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
export load, save, id, labels, names, features

###=============================================================================
### Imports
###=============================================================================

# Utilities
include("Utilities.jl")
using FeatureScreening.Utilities: ExpStep, make_rng
using Base.Iterators: partition
using Dumper: @dump

include("importance.jl")

# Feature set related
include("Types.jl")
using FeatureScreening.Types: AbstractFeatureSet, FeatureSet
using FeatureScreening.Types: id, labels, names, features, load, save

# Fixtures
include("Fixtures.jl")

# API dependencies
using Random: AbstractRNG, GLOBAL_RNG, shuffle as __shuffle
using ProgressMeter: Progress, next!
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

Screen input feature set with the given configuration. Create a new `FeatureSet`
irrespective the type of the input feature set.

**WARNING**: This function can run for a long time, proportional to the number
of features, largely depending on the configuration of the random forests used
for ranking the features internally.

**DISCLAIMER**: Internal ranking, importance computation is based on random
forests. Potential future improvement is to add other ranking methods.

# Steps:

0. shuffle the features (optional);
1. slice feature into (fixed size, disjoint) partitions;
2. iterate over the partitions to:

    1. compute importances for the given partition;
    2. keep the best features (with the highest importance);
    3. merge these with the next partition, and continue with step 2.1.

3. return the last set of best features when all the partitions were processed.
"""
function screen(feature_set...; kwargs...)
    return screen(FeatureSet(feature_set...); kwargs...)
end

"""
    screen(feature_set::AbstractFeatureSet;
           reduced_size::Integer            = size(feature_set, 2) ÷ 5,
           step_size::Integer               = size(feature_set, 2) ÷ 10,
           config::NamedTuple               = DEFAULT_SCREEN_CONFIG,
           shuffle::Bool                    = false,
           before::Function                 = skip,
           after::Function                  = skip,
           rng::Union{AbstractRNG, Integer} = GLOBAL_RNG
          )::AbstractFeatureSet

# Parameters:
- `reduced_size`: Expected number of screened features (Right now, this is an
  upper bound).
- `step_size`: Size of each partition.
- `config`: Random forest configuration of the importance computing.
- `shuffle`: Flag to shuffle features before partition, to randomize the order
  of the features.
- `before`: Callback function, that executes before importance computation and
  selection part. Inputs are the previously selected features and the actual
  partition, output will be ignored.
- `after`: Callback function, that executes after importance computation and
  selection part. Inputs are the selected features, output will be ignored.
- `rng`: Random generator or a seed.
"""
function screen(feature_set::AbstractFeatureSet{L, N, F};
                reduced_size::Integer       = size(feature_set, 2) ÷ 5,
                step_size::Integer          = size(feature_set, 2) ÷ 10,
                config::NamedTuple          = DEFAULT_SCREEN_CONFIG,
                shuffle::Bool               = false,
                before::Function            = skip,
                after::Function             = skip,
                show_progress::Bool         = true,
                rng::Union{AbstractRNG, Integer} = GLOBAL_RNG
               )::AbstractFeatureSet{L, N, F} where {L, N, F}

    all::AbstractVector{N} = names(feature_set)
    if shuffle
        all = __shuffle(make_rng(rng), all)
    end

    selected::AbstractFeatureSet = feature_set[:, N[]]

    parts = partition(all, step_size)
    progress = Progress(length(parts);
                        desc = "Screening...",
                        enabled = show_progress)
    for (i, part) in enumerate(parts)
        new::AbstractFeatureSet = feature_set[:, part]

        # Before the computation
        before(selected, new)

        to_be_selected::AbstractFeatureSet = merge(selected, new)

        importances::Vector{Pair{N, <: Real}} =
            feature_importance(to_be_selected; config, rng)

        @dump importances path="importances.$i.csv"

        important_names::Vector{<: N} =
            select(importances, Top(reduced_size); strict = false) .|> label

        selected = to_be_selected[:, important_names]

        # After the computation
        after(selected)

        next!(progress)
    end

    return selected
end

end # module
