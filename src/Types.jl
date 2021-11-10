###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

module Types

###=============================================================================
### Imports
###=============================================================================

###-----------------------------------------------------------------------------
### Feature set
###-----------------------------------------------------------------------------

### Struct
using Base: @kwdef
using UUIDs: UUID, uuid4
using Dates: DateTime, now, UTC

### Getters
import FeatureScreening.Utilities: id
import Base: names

### Base API
import Base: show, ==, hash, getindex

## Size API
import Base: ndims, axes, size, length

## Iterable API
import Base: iterate, eachrow, eachcol

### Research API
using DecisionTree: Ensemble as RandomForest
import DecisionTree: build_forest, nfoldCV_forest
using FeatureScreening.Utilities: __build_forest, __nfoldCV_forest
import FeatureScreening: feature_importance

### File API
import FeatureScreening.Utilities: save, load, id, created_at
using FeatureScreening.Utilities: FILENAME_DATETIME_FORMAT
using HDF5: h5open, File as HDF5File

### Others
import Base: merge, rand

###=============================================================================
### Feature set
###=============================================================================

"""
    FeatureSet{L, N, F}

This object contains feature values by given labels and feature names.

# Parametric types:

- `L`: Type of the labels.
- `N`: Type of the feature names.
- `F`: Type of the feature values, typically some numeric type.

# Example

|           | "feature 1" | "feature 2" |   ...   | "feature N" |
|:---------:|:-----------:|:-----------:|:-------:|:-----------:|
| "label-1" |   101.001   |   431.331   |   ...   |   20.9221   |
| "label-1" |   121.340   |   421.393   |   ...   |   21.3419   |
|    ...    |     ...     |     ...     |   ...   |     ...     |
| "label-M" |   131.349   |   134.119   |   ...   |   -0.1124   |
| "label-M" |   128.218   |   329.218   |   ...   |   10.0038   |

"""
@kwdef struct FeatureSet{L, N, F}
    id::UUID = uuid4()
    created_at::DateTime = now(UTC)

    labels::AbstractVector{L}
    names::AbstractVector{N}
    features::AbstractMatrix{F}

    __name_indices::Dict{N, Int}
end

"""
    FeatureSet(labels::AbstractVector{L},
               names::AbstractVector{N},
               features::AbstractMatrix{F};
               id::UUID = uuid4(),
               created_at::DateTime = now(UTC)
              )::FeatureSet{L, N, F} where {L, N, F}

"""
function FeatureSet(labels::AbstractVector{L},
                    names::AbstractVector{N},
                    features::AbstractMatrix{F};
                    kwargs...
                   )::FeatureSet{L, N, F} where {L, N, F}
    @assert (length(labels), length(names)) == size(features)

    __name_indices::Dict{N, Int} =
        Dict(name => i for (i, name) in enumerate(names))

    return FeatureSet{L, N, F}(;
                               labels,
                               names,
                               features,
                               __name_indices,
                               kwargs...)
end

"""
    FeatureSet(X, y)

Create a `FeatureSet` from a feature matrix and labels. (Classic data science
API)
"""
function FeatureSet(X::AbstractMatrix{F},
                    y::AbstractVector{L};
                    kwargs...
                   )::FeatureSet{L, Int, F} where {L, F}
    return FeatureSet(y, 1:size(X, 2), X; kwargs...)
end

###-----------------------------------------------------------------------------
### Getters
###-----------------------------------------------------------------------------

function id(feature_set::FeatureSet)::UUID
    return feature_set.id
end

function created_at(feature_set::FeatureSet)::DateTime
    return feature_set.created_at
end

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
### Base API
###-----------------------------------------------------------------------------

function show(io::IO, features::FeatureSet{L, N, F})::Nothing where {L, N, F}
    (height, width) = size(features)
    print(io, "$(FeatureSet{L, N, F})<$(height) x $(width)>")
    return nothing
end

function ==(a::FeatureSet, b::FeatureSet)
    return hash(a) == hash(b)
end

function hash(feature_set::FeatureSet, h::UInt64)::UInt64
    parts = [labels(feature_set), names(feature_set), features(feature_set)]
    return reduce(parts; init = h) do h, part
        return hash(part, h)
    end
end

function getindex(feature_set::FeatureSet{L, N, F},
                  label_indices,
                  name_indices,
                 )::FeatureSet{L, N, F} where {L, N, F}
    i = label_indices
    j = resolve_name_indices(feature_set, name_indices)

    return FeatureSet(@view(labels(feature_set)[i]),
                      @view(names(feature_set)[j]),
                      @view(features(feature_set)[i, j]))
end

function resolve_name_indices(feature_set::FeatureSet, ::Colon)::Colon
    return (:)
end

function resolve_name_indices(feature_set::FeatureSet{_L, N},
                              names::AbstractVector{<: N}
                             )::Vector{Int} where {_L, N}
    return map(names) do name::N
        return feature_set.__name_indices[name]
    end
end

##------------------------------------------------------------------------------
## Size API
##------------------------------------------------------------------------------

function ndims(feature_set::FeatureSet)::Int
    return ndims(features(feature_set))
end

function axes(feature_set::FeatureSet)::Tuple
    return axes.(Ref(feature_set), size(feature_set))
end

function axes(feature_set::FeatureSet, dim::Int)
    @assert dim in [1, 2]
    if dim == 1
        return 1:size(feature_set, 1)
    elseif dim == 2
        return names(feature_set)
    end
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

##------------------------------------------------------------------------------
## Iterable API
##------------------------------------------------------------------------------

function iterate(feature_set::FeatureSet)
    return iterate(eachrow(feature_set))
end

function iterate(feature_set::FeatureSet, state)
    return iterate(eachrow(feature_set), state)
end

function eachrow(feature_set::FeatureSet)
    return zip(labels(feature_set),
               eachrow(features(feature_set)))
end

function eachcol(feature_set::FeatureSet)
    return zip(names(feature_set),
               eachcol(features(feature_set)))
end

###-----------------------------------------------------------------------------
### Research API
###-----------------------------------------------------------------------------

function build_forest(feature_set::FeatureSet{L, N, F};
                      config = (;),
                      kwargs...
                     ) where {L, N, F}
    return __build_forest(labels(feature_set),
                          features(feature_set);
                          config,
                          kwargs...)
end

function nfoldCV_forest(feature_set::FeatureSet;
                        config = (;),
                        verbose = false)
    return __nfoldCV_forest(labels(feature_set),
                            features(feature_set);
                            config,
                            verbose)
end

function feature_importance(feature_set::FeatureSet{L, N};
                            config = (;),
                            kwargs...
                           )::Vector{Pair{N, Int}} where {L, N}
    forest::RandomForest = build_forest(feature_set; config, kwargs...)
    importances::Vector{Pair{Int, Int}} = feature_importance(forest)

    return [names(feature_set)[i] => importance
            for (i, importance) in importances]
end

###-----------------------------------------------------------------------------
### File API
###-----------------------------------------------------------------------------

##------------------------------------------------------------------------------
## Save, load
##------------------------------------------------------------------------------

function save(feature_set::FeatureSet; directory = ".")::Nothing
    path::String = joinpath(directory, filename(feature_set))
    save(path, feature_set)
    return nothing
end

function save(filename::AbstractString, feature_set::FeatureSet)::Nothing
    h5open(filename, "w") do file
        file["id"] = id(feature_set) |> to_hdf5
        file["created_at"] = created_at(feature_set) |> to_hdf5
        file["features"] = features(feature_set) |> to_hdf5
        file["labels"] = labels(feature_set) |> to_hdf5
        file["names"] = names(feature_set) |> to_hdf5
    end

    return nothing
end

function load(::Type{FeatureSet}, path::AbstractString)::FeatureSet
    return h5open(path, "r") do file
        @assert isvalid(FeatureSet, file)

        id = read(file, "id") |> UUID
        created_at = read(file, "created_at") |> DateTime
        features = read(file, "features")
        labels = read(file, "labels")
        names = read(file, "names")
        return FeatureSet(labels, names, features; id, created_at)
    end
end

##------------------------------------------------------------------------------
## Miscs
##------------------------------------------------------------------------------

function filename(feature_set::FeatureSet)::String
    return "$(id(feature_set)).hdf5"
end

function isvalid(::Type{FeatureSet}, path::AbstractString)::Bool
    return h5open(path, "r") do file
        return isvalid(FeatureSet, file)
    end
end

function isvalid(::Type{FeatureSet}, file::HDF5File)::Bool
    return ["id", "created_at", "labels", "names", "features"] ⊆ keys(file)
end

##------------------------------------------------------------------------------
## Internals
##------------------------------------------------------------------------------

"""
CAUTION! This function will create a new array if the input was an array view.
"""
function to_hdf5(x::SubArray{_T, _N, A})::A where {_T, _N, A}
    return copy(x)
end

function to_hdf5(x::AbstractRange)::Vector
    return collect(x)
end

function to_hdf5(x::UUID)::String
    return string(x)
end

function to_hdf5(x::DateTime)::String
    return string(x)
end

function to_hdf5(x)
    return x
end

###-----------------------------------------------------------------------------
### Others
###-----------------------------------------------------------------------------

# TODO https://github.com/cursorinsight/FeatureScreening.jl/issues/13
function merge(a::FeatureSet, b::FeatureSet)::FeatureSet
    @assert labels(a) == labels(b)
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

# TODO https://github.com/cursorinsight/FeatureScreening.jl/issues/12
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

end # module
