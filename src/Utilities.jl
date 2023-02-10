###-----------------------------------------------------------------------------
### Copyright (C) FeatureScreening.jl
###
### SPDX-License-Identifier: MIT License
###-----------------------------------------------------------------------------

module Utilities

###=============================================================================
### Imports
###=============================================================================

# `DecisionTree` wrappers
using DecisionTree: Ensemble as RandomForest
using DecisionTree: build_forest, nfoldCV_forest

# File I/O
using UUIDs: UUID, uuid5

# HDF5
import Base: get, get!
using HDF5: File as HDF5File

# Rest
using Random: AbstractRNG, GLOBAL_RNG, MersenneTwister

###=============================================================================
### API
###=============================================================================

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
### File I/O
###-----------------------------------------------------------------------------

##------------------------------------------------------------------------------
## Save
##------------------------------------------------------------------------------

function save end

##------------------------------------------------------------------------------
## Load
##------------------------------------------------------------------------------

function load end

##------------------------------------------------------------------------------
## Somehow related callbacks
##------------------------------------------------------------------------------
## Paths, filenames, computed identifiers, etc.

function id(nt::C)::UUID where {C <: NamedTuple}
    return uuid5(UUID(hash(nt)), "config")
end

function created_at end

function filename end

function path(directory::AbstractString, filename::AbstractString)::String
    return joinpath(directory, filename)
end

function path(directory::AbstractString, x)::String
    return joinpath(directory, filename(x))
end

###-----------------------------------------------------------------------------
### HDF5
###-----------------------------------------------------------------------------

function get(file::HDF5File, key::AbstractString, default)
    return if haskey(file, key)
        read(file, key)
    else
        default
    end
end

function get!(file::HDF5File, key::AbstractString, default)
    return if haskey(file, key)
        read(file, key)
    else
        file[key] = default
    end
end

###-----------------------------------------------------------------------------
### Pattern matching for functions
###-----------------------------------------------------------------------------

macro with_pattern(f)
    @assert f.head == :function
    f_name::Symbol = f.args[1]
    return quote
        $f
        function $f_name(tokens::Symbol...)
            try
                return $f_name(Val.(tokens)...)
            catch e
                if e isa MethodError
                    @error "Missing pattern $(e.f)($(e.args...))"
                end
                rethrow(e)
            end
        end
    end |> esc
end

macro pattern(f)
    @assert f.head == :function
    (signature, body...) = f.args
    @assert(f.args[1].head == :call,
            "Simple function signature allowed(, no return type assertion)!")
    (function_name::Symbol, tokens...) = f.args[1].args
    arguments::Vector = map(tokens) do token
        return :(::Val{$token})
    end

    return quote
        function $function_name($(arguments...))
            return $(body...)
        end
    end |> esc
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

const FILENAME_DATETIME_FORMAT = "YYYYmmdd-HHMMSS"

end # module
