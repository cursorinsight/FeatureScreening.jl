###-----------------------------------------------------------------------------
### Copyright (C) FeatureScreening.jl
###
### SPDX-License-Identifier: MIT License
###-----------------------------------------------------------------------------

module FeatureScreening

###=============================================================================
### Includes
###=============================================================================

include("Utilities.jl")
include("importance.jl")

###=============================================================================
### Imports
###=============================================================================

# Exports
using Reexport: @reexport

# Utilities
using .Utilities: Maybe, make_rng

# API dependencies
using Base.Iterators: partition
using Compat: Returns
using Dumper: @dump
using ProgressMeter: Progress, next!
using Random: AbstractRNG, GLOBAL_RNG, shuffle as __shuffle

import FeatureSets

###=============================================================================
### Exports
###=============================================================================

# API
export screen

# FeatureSets
@reexport using FeatureSets: AbstractFeatureSet, FeatureSet
@reexport using FeatureSets: id, labels, names, features, load, save

###=============================================================================
### API
###=============================================================================

# External lib FeatureSets implements functionality of legacy submodule Types.
# Keep an alias here to maintain backwards compatibility.
const Types = FeatureSets

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
           reduced_size::Maybe{Integer}         = size(feature_set, 2) ÷ 5,
           step_size::Integer                   = size(feature_set, 2) ÷ 10,
           selection_mode::Maybe{SelectionMode} = nothing,
           config::NamedTuple                   = DEFAULT_SCREEN_CONFIG,
           shuffle::Bool                        = false,
           before::Function                     = Returns(nothing),
           after::Function                      = Returns(nothing),
           show_progress::Bool                  = true,
           rng::Union{AbstractRNG, Integer}     = GLOBAL_RNG
          )::AbstractFeatureSet

# Parameters

- `reduced_size`: Expected number of screened features (an upper bound).
  Mutually exclusive with `selection_mode`.
- `step_size`: Size of each partition.
- `selection_mode`: a mode to pick selected features after importance
  computation. Mutually exclusive with `reduced_size`. See subtypes of
  `SelectionMode` For various selection modes.
- `config`: Parameters of the random forest used for importance computation in
  each round.
- `shuffle`: Whether to shuffle the features before partitioning.
- `before`: Callback function, that is executed before importance computation
  and feature selection. It is called with the previously selected features and
  the current partition, its return value is ignored.
- `after`: Callback function, that is executed after importance computation and
  feature selection. It is called with the selected features, its return value
  is ignored.
- `rng`: Random generator or seed to be used.
"""
function screen(feature_set::AbstractFeatureSet{L, N, F};
                reduced_size::Maybe{Integer}         = nothing,
                step_size::Integer                   = size(feature_set, 2) ÷ 10,
                selection_mode::Maybe{SelectionMode} = nothing,
                config::NamedTuple                   = DEFAULT_SCREEN_CONFIG,
                shuffle::Bool                        = false,
                before::Function                     = Returns(nothing),
                after::Function                      = Returns(nothing),
                show_progress::Bool                  = true,
                rng::Union{AbstractRNG, Integer}     = GLOBAL_RNG
               )::AbstractFeatureSet{L, N, F} where {L, N, F}

    @assert reduced_size === nothing || selection_mode === nothing "At most " *
        "one of `reduced_size` and `selection_mode` must be specified!"
    reduced_size = something(reduced_size, size(feature_set, 2) ÷ 5)
    selection_mode = something(selection_mode,
                               SelectTop(reduced_size; strict = false))

    all::AbstractVector{N} = names(feature_set)
    if shuffle
        all = __shuffle(make_rng(rng), all)
    end

    selected::AbstractFeatureSet = @view feature_set[:, N[]]

    parts = partition(all, step_size)
    progress = Progress(length(parts);
                        desc = "Screening...",
                        enabled = show_progress)
    for (i, part) in enumerate(parts)
        new::AbstractFeatureSet = @view feature_set[:, part]

        # Before the computation
        before(selected, new)

        to_be_selected::AbstractFeatureSet = merge(selected, new)

        importances::Vector{Pair{N, <: Real}} =
            feature_importance(to_be_selected; config, rng)

        @dump importances path="importances.$i.csv" (mime = "text/csv")

        important_names::Vector{<: N} =
            select(make_rng(rng), importances, selection_mode) .|> label

        selected = @view to_be_selected[:, important_names]

        # After the computation
        after(selected)

        next!(progress)
    end

    return selected
end

end # module
