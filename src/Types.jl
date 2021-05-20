###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

module Types

###=============================================================================
### Imports
###=============================================================================

import Base: show, getindex, ndims, size, length, iterate, deleteat!, merge!

using DecisionTree: Ensemble as RandomForest
import DecisionTree: build_forest, nfoldCV_forest

using FeatureScreening.Utilities: Maybe
using FeatureScreening: feature_importance
import FeatureScreening: feature_importance

using IterTools: subsets
using Statistics: cor
import FeatureScreening.Utilities: partition

###=============================================================================
### Abstract feature set
###=============================================================================

"""
    `abstract type AbstractFeatureSet{L, N, F} end`

    - `L` is the type of the labels
    - `N` is the type of the feature names, by default `Int` as a vector index
    - `F` is the type of the feature values, typically some numeric type
"""
abstract type AbstractFeatureSet{L, N, F} end

###-----------------------------------------------------------------------------
### Showable API
###-----------------------------------------------------------------------------

function show(io::IO, features::T) where {T <: AbstractFeatureSet}
    (height, width) = size(features)
    println(io, "$(T)<$(height) x $(width)>")
    println(io, "  + $(label_count(features)) labels")
    println(io, "  + $(sample_count(features)) samples")
    println(io, "  + $(feature_count(features)) features")
    return nothing
end

###-----------------------------------------------------------------------------
### Indexable API
###-----------------------------------------------------------------------------

function getindex(features::AbstractFeatureSet{L, N, F},
                  feature_names::AbstractVector{>: N}
                 )::FeatureSubset{L, N, F} where {L, N, F}
    return FeatureSubset{L, N, F}(features, feature_names)
end

function getindex(features::AbstractFeatureSet{L, N, F},
                  feature_name::N
                 )::Vector{F} where {L, N, F}
    n::Int = length(features)
    feature_values::Vector{F} = Vector{F}(undef, n)

    for (i, (_, (feature,))) in enumerate(features[[feature_name]])
        feature_values[i] = feature
    end

    return feature_values
end

###-----------------------------------------------------------------------------
### Iterable API
###-----------------------------------------------------------------------------

function ndims(features::AbstractFeatureSet)::Int
    return 2
end

function size(features::AbstractFeatureSet)::Tuple{Int, Int}
    return Tuple(size(features, dim) for dim in 1:ndims(features))
end

function size(features::AbstractFeatureSet, dim::Integer)::Int
    if dim == 1
        return label_count(features) * sample_count(features)
    elseif dim == 2
        return feature_count(features)
    end
end

function length(features::AbstractFeatureSet)::Int
    return size(features, 1)
end

# TODO this is the by-sample iteration
function iterate(features::AbstractFeatureSet{L},
                 state = (nothing, nothing)
                )::Maybe{<: Tuple} where {L}
    (label_idx::Maybe{Any}, sample_idx::Maybe{Any}) = state

    if label_idx isa Nothing
        label_idx = firstindex(by_labels(features))
    end

    by_label::Maybe{<: Tuple} = iterate(by_labels(features), label_idx)

    if by_label isa Nothing
        return nothing
    else
        ((label, samples), next_label_idx) = by_label

        if sample_idx isa Nothing
            sample_idx = firstindex(samples)
        end

        by_sample::Maybe{<: Tuple} = iterate(samples, sample_idx)
        if by_sample isa Nothing
            return iterate(features, (next_label_idx, nothing))
        else
            (sample, next_sample_idx) = by_sample
            return (label => sample[feature_names(features)],
                    (label_idx, next_sample_idx))
        end
    end
end

###-----------------------------------------------------------------------------
### Size related callbacks
###-----------------------------------------------------------------------------

function label_count end
function sample_count end
function feature_count end

###-----------------------------------------------------------------------------
### Iteration related callbacks
###-----------------------------------------------------------------------------

function by_labels end

###-----------------------------------------------------------------------------
### XXX API
###-----------------------------------------------------------------------------

function labels end
function feature_names end

function mtx(features::AbstractFeatureSet{L, N, F}
            )::Tuple{Vector{L}, Vector{N}, Matrix{F}} where {L, N, F}
    (n::Int, m::Int) = size(features)
    labels::Vector{L} = Vector{L}(undef, n)
    mtx::Matrix{F} = Matrix{F}(undef, n, m)

    for (i, (label, feature_vector)) in enumerate(features)
        labels[i] = label
        mtx[i, :] = feature_vector
    end

    return (labels,
            feature_names(features),
            mtx)
end

function partition(features::AbstractFeatureSet{L, N, F},
                   n::Int;
                   kwargs...
                  )::Vector{FeatureSubset{L, N, F}} where {L, N, F}
    return [features[part]
            for part in partition(feature_names(features), n; kwargs...)]
end

function build_forest(features::AbstractFeatureSet{L, N, F};
                      config = (;)
                     ) where {L, N, F}
    (labels::Vector{L}, _, feature_matrix::Matrix{F}) = mtx(features)
    return build_forest(labels, feature_matrix; config)
end

function nfoldCV_forest(features::AbstractFeatureSet{L, N, F};
                        config = (;),
                        verbose = false
                       ) where {L, N, F}
    (labels::Vector{L}, _, feature_matrix::Matrix{F}) = mtx(features)
    return nfoldCV_forest(labels, feature_matrix; config, verbose)
end

function feature_importance(features::AbstractFeatureSet{_, N};
                            config = (;)
                           )::Vector{Pair{N, Int}} where {_, N}
    forest::RandomForest = build_forest(features; config)
    importances::Vector{Pair{Int, Int}} = feature_importance(forest)

    return [feature_names(features)[idx] => importance
            for (idx, importance) in importances]
end

##==============================================================================
## Correlation filter
##==============================================================================

struct PearsonCorrelation end

function filter!(::Type{PearsonCorrelation},
                 features::AbstractFeatureSet{L, N, F};
                 threshold::Real = 0.95,
                 count::Real = Inf
                )::AbstractFeatureSet{L, N, F} where {L, N, F}
    correlated::Vector{N} = N[]

    # TODO replace enumerate with some direct indexing
    for (a, b) in subsets(LinearIndices(feature_names(features)), 2)
        correlation::Float64 = cor(features[a], features[b])

        #@info "Correlations" feature_a feature_b correlation threshold
        if correlation > threshold
            push!(correlated, b)
        end

        if length(correlated) >= count
            break
        end
    end

    return deleteat!(features, correlated)
end

###=============================================================================
### Feature set
###=============================================================================

struct FeatureSet{L, N, F} <: AbstractFeatureSet{L, N, F}
    by_labels
end

# TODO
function FeatureSet(features::Matrix{F}, labels::Vector{L}) where {F, L}
    idxs::LinearIndices = keys(labels)

    idxs_by_labels = Dict{L, Vector{Int}}()

    foldl(idxs, init = idxs_by_labels) do acc::Dict{L, Vector{Int}}, idx::Int
        label::L = labels[idx]
        if !haskey(acc, label)
            acc[label] = Int[]
        end
        push!(acc[label], idx)
        return acc
    end

    return FeatureSet{L, Int, F}([label => [@view features[idx, :] for idx in idxs]
                                  for (label, idxs) in idxs_by_labels])
end

function FeatureSet(by_labels::Vector{Pair{L, Vector{Fs}}}
                   ) where {L, F, Fs <: Vector{F}}
    return FeatureSet{L, Int, F}(by_labels)
end

###-----------------------------------------------------------------------------
### Size related callbacks
###-----------------------------------------------------------------------------

function label_count(features::FeatureSet)::Int
    return length(features.by_labels)
end

function sample_count(features::FeatureSet)::Int
    return length(features.by_labels[1][2])
end

function feature_count(features::FeatureSet)::Int
    return length(features.by_labels[1][2][1])
end

###-----------------------------------------------------------------------------
### Iteration related callbacks
###-----------------------------------------------------------------------------

function by_labels(features::FeatureSet)
    return features.by_labels
end

###-----------------------------------------------------------------------------
### XXX API
###-----------------------------------------------------------------------------

function labels(features::FeatureSet)
    return first.(features.by_labels)
end

function feature_names(features::FeatureSet)
    return 1:feature_count(features)
end

###=============================================================================
### Feature subset
###=============================================================================

struct FeatureSubset{L, N, F} <: AbstractFeatureSet{L, N, F}
    features::AbstractFeatureSet{L, N, F}
    feature_names::AbstractVector{N}
end

###-----------------------------------------------------------------------------
### Indexable API
###-----------------------------------------------------------------------------

# TODO improve implementation
function deleteat!(features::F,
                   to_be_deleted::AbstractVector{N}
                  )::F where {_, N, F <: FeatureSubset{_, N}}
    found::Vector{N} = [feature
                        for feature in feature_names(features)
                        if feature in to_be_deleted]
    deleteat!(feature_names(features), found)
    return features
end

# TODO revamp, design
function merge!(a::FeatureSubset, b::FeatureSubset)::FeatureSubset
    # TODO
    @assert by_labels(a) == by_labels(b)
    # TODO
    append!(feature_names(a), feature_names(b)) |> unique!
    return a
end

###-----------------------------------------------------------------------------
### Size related callbacks
###-----------------------------------------------------------------------------

function label_count(features::FeatureSubset)::Int
    return label_count(features.features)
end

function sample_count(features::FeatureSubset)::Int
    return sample_count(features.features)
end

function feature_count(features::FeatureSubset)::Int
    length(features.feature_names)
end

###-----------------------------------------------------------------------------
### Iteration related callbacks
###-----------------------------------------------------------------------------

function by_labels(features::FeatureSubset)
    return by_labels(features.features)
end

###-----------------------------------------------------------------------------
### XXX API
###-----------------------------------------------------------------------------

function labels(features::FeatureSubset)
    return labels(features.features)
end

function feature_names(features::FeatureSubset)
    return features.feature_names
end

end # module
