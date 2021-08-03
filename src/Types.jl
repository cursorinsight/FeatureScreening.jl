###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

module Types

###=============================================================================
### Exports
###=============================================================================

export FeatureSet, labels, names, features
export PearsonCorrelation

###=============================================================================
### Imports
###=============================================================================

###-----------------------------------------------------------------------------
### FeatureSet imports
###-----------------------------------------------------------------------------

# Basic API
import Base: show, getindex, ndims, size, length, iterate, merge, rand
import Base: eachrow, eachcol, iterate
import FeatureScreening.Utilities: partition
import Base: names

# Research API
using DecisionTree: Ensemble as RandomForest
import DecisionTree: build_forest, nfoldCV_forest
using FeatureScreening.Utilities: _build_forest, _nfoldCV_forest
import FeatureScreening: feature_importance

# File API
using HDF5: h5open

###-----------------------------------------------------------------------------
### PearsonCorrelation imports
###-----------------------------------------------------------------------------

using IterTools: subsets
using Statistics: cor

###=============================================================================
### Feature set data API
###=============================================================================

function labels end
function names end
function features end

###=============================================================================
### Feature set
###=============================================================================

"""
    `struct FeatureSet{L, N, F} end`

    - `L` is the type of the labels
    - `N` is the type of the feature names, by default `Int` as a vector index
    - `F` is the type of the feature values, typically some numeric type
"""
struct FeatureSet{L, N, F}
    labels::AbstractVector{L}
    names::AbstractVector{N}
    features::AbstractMatrix{F}

    name_idxs::Dict{N, Int}

    function FeatureSet(labels::AbstractVector{L},
                        names::AbstractVector{N},
                        features::AbstractMatrix{F}
                       ) where {L, N, F}
        @assert (length(labels), length(names)) == size(features)

        name_idxs::Dict{N, Int} =
            Dict(name => i for (i, name) in enumerate(names))
        return new{L, N, F}(labels, names, features, name_idxs)
    end

    function FeatureSet(X::AbstractMatrix{F},
                        y::AbstractVector{L}
                       ) where {L, F}
        return FeatureSet(y, 1:size(X, 2), X)
    end
end

###-----------------------------------------------------------------------------
### Base API
###-----------------------------------------------------------------------------

function show(io::IO, features::FeatureSet{L, N, F})::Nothing where {L, N, F}
    (height, width) = size(features)
    println(io, "$(FeatureSet{L, N, F})<$(height) x $(width)>")
    return nothing
end

function getindex(feature_set::FeatureSet{L, N, F},
                  i,
                  names,
                 )::FeatureSet{L, N, F} where {L, N, F}
    j::Vector{Int} = [feature_set.name_idxs[name] for name in names]

    _labels::AbstractVector{L} = @view labels(feature_set)[i]
    _names::AbstractVector{N} = convert(AbstractVector{N}, names)
    _features::AbstractMatrix{F} = @view features(feature_set)[i, j]
    return FeatureSet(_labels, _names, _features)
end

function ndims(feature_set::FeatureSet)::Int
    return ndims(features(feature_set))
end

function size(feature_set::FeatureSet)::Tuple{Int, Int}
    return size(features(feature_set))
end

function size(feature_set::FeatureSet, dim::Int)::Int
    return size(features(feature_set), dim)
end

function length(feature_set::FeatureSet)::Int
    return size(features(feature_set), 1)
end

function eachrow(feature_set::FeatureSet)
    return zip(labels(feature_set),
               eachrow(features(feature_set)))
end

function eachcol(feature_set::FeatureSet)
    return zip(names(feature_set),
               eachcol(features(feature_set)))
end

function iterate(feature_set::FeatureSet)
    return iterate(eachrow(feature_set))
end

function iterate(feature_set::FeatureSet, state)
    return iterate(eachrow(feature_set), state)
end

# TODO revamp, design
function merge(a::FeatureSet, b::FeatureSet)::FeatureSet
    @assert labels(a) == labels(b)
    # TODO
    @assert features(a) isa SubArray
    @assert features(b) isa SubArray
    @assert features(a).parent === features(b).parent

    unique_names::Vector = [names(a); names(b)] |> unique!
    unique_features::AbstractMatrix =
        SubArray(features(a).parent, (features(a).indices[1],
                                      unique!([features(a).indices[2];
                                               features(b).indices[2]])))

    return FeatureSet(labels(a),
                      unique_names,
                      unique_features)
end

function merge(xs::FeatureSet...)::FeatureSet
    return reduce(merge, xs)
end

# TODO refactor feature generation by:
# - designing the generator function
# - making to be able to set up custom distribution to produce values
"""
This function generates only per-label-BALANCED feature set.
"""
function rand(::Type{FeatureSet{L, N, F}},
              sample_count::Integer = 10,
              feature_count::Integer = 10;
              label_count::Integer = sample_count ÷ 5,
              center::Function = i -> ((i-1) / label_count + 1),
              place::Function = j -> 7j / feature_count,
              random::Function = (i, j) -> randn()
             )::FeatureSet{L, N, F} where {L, N <: Integer, F <: AbstractFloat}
    (d, r) = divrem(sample_count, label_count)
    @assert iszero(r)

    labels::Vector{L} = L.(repeat(1:label_count, inner = d))
    names::Vector{N} = N.(collect(1:feature_count))

    features::Matrix{F} =
        [center(i) * place(j) + random(i, j)
         for i in 1:sample_count, j in 1:feature_count]

    return FeatureSet(labels, names, features)
end

function rand(::Type{FeatureSet},
              sample_count::Integer = 10,
              feature_count::Integer = 10;
              kwargs...
             )::FeatureSet
    return rand(FeatureSet{Int, Int, Float64},
                sample_count,
                feature_count;
                kwargs...)
end

###-----------------------------------------------------------------------------
### Feature set data API
###-----------------------------------------------------------------------------

function labels(feature_set::FeatureSet{L}
               )::AbstractVector{L} where {L}
    return feature_set.labels
end

function names(feature_set::FeatureSet{L, N}
              )::AbstractVector{N} where {L, N}
    return feature_set.names
end

function features(feature_set::FeatureSet{L, N, F}
                 )::AbstractMatrix{F} where {L, N, F}
    return feature_set.features
end

###-----------------------------------------------------------------------------
### Research API
###-----------------------------------------------------------------------------

function build_forest(feature_set::FeatureSet{L, N, F};
                      config = (;)
                     ) where {L, N, F}
    return _build_forest(labels(feature_set), features(feature_set); config)
end

function nfoldCV_forest(feature_set::FeatureSet;
                        config = (;),
                        verbose = false)
    return _nfoldCV_forest(labels(feature_set),
                           features(feature_set);
                           config,
                           verbose)
end

function feature_importance(feature_set::FeatureSet{L, N};
                            config = (;)
                           )::Vector{Pair{N, Int}} where {L, N}
    forest::RandomForest = build_forest(feature_set; config)
    importances::Vector{Pair{Int, Int}} = feature_importance(forest)

    return [names(feature_set)[idx] => importance
            for (idx, importance) in importances]
end

###-----------------------------------------------------------------------------
### I/O: save, load
###-----------------------------------------------------------------------------

function save(feature_set::FeatureSet, filename::AbstractString)::Nothing
    h5open(filename, "w") do fid
        fs = features(feature_set)
        fid["features"] = (fs isa Matrix) ? fs : copy(fs)
        fid["labels"] = labels(feature_set)
        fid["names"] = collect(names(feature_set))
    end

    return nothing
end

function load(filename::AbstractString)::FeatureSet
    return h5open(filename) do fid
        features = read(fid, "features")
        labels = read(fid, "labels")
        names = read(fid, "names")
        return FeatureSet(labels, names, features)
    end
end

##==============================================================================
## Correlation filter
##==============================================================================

struct PearsonCorrelation end

function filter!(::Type{PearsonCorrelation},
                 features::FeatureSet{L, N, F};
                 threshold::Real = 0.95,
                 count::Real = Inf
                )::FeatureSet{L, N, F} where {L, N, F}
    correlated::Vector{N} = N[]

    # TODO replace enumerate with some direct indexing
    for (a, b) in subsets(LinearIndices(names(features)), 2)
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

end # module
