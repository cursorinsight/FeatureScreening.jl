module Utilities

###=============================================================================
### Imports
###=============================================================================

using DecisionTree: Ensemble as RandomForest

using DecisionTree: build_forest, nfoldCV_forest

using UUIDs: UUID, uuid5

using HDF5: File as HDF5File
import Base: get, get!
using Dates: DateTime
using HDF5: H5T_TIME as HDF5Time, H5S_SCALAR as HDF5Scalar
import HDF5: datatype, dataspace

###=============================================================================
### API
###=============================================================================

const Maybe{T} = Union{Nothing, T}

function partition(xs::AbstractVector{X},
                   n::Int;
                   rest::Bool = false
                  )::Vector{Vector{X}} where {X}
    # TODO remove this conditional function call
    m::Int = (rest ? ceil : floor)(length(xs) / n)
    return [xs[(n*(i-1)+1):min(n*i, length(xs))] for i in 1:m]
end

abstract type AbstractStep end

# TODO
struct ExpStep{T} <: AbstractStep
    base::T

    function ExpStep(base::T) where {T}
        # TODO
        @assert 1 < base
        return new{T}(base)
    end
end

struct Size{T}
    n::T
end

struct ZeroStep <: AbstractStep end

Base.zero(::T) where {T <: AbstractStep} = zero(T)
Base.zero(::Type{T}) where {T <: AbstractStep} = ZeroStep()
Base.:<(::ZeroStep, ::AbstractStep) = true

# TODO
function Base.length(range::StepRange{Int, <: ExpStep})
    return range.stop - range.start + 1
end

function (::Colon)(start::Real, step::S, stop::Real) where {S <: ExpStep}
    return StepRange{Int, S}(ceil(Int, log(step.base, start)), step, stop)
end

function (::Colon)(start::Real, size::S, stop::Real) where {S <: Size}
    return range(start, stop; length = size.n)
end

function Base.steprange_last(start, step::ExpStep, stop)
    # TODO
    if iszero(stop)
        return -1
    else
        return floor(Int, log(step.base, stop))
    end
end

function Base.iterate(range::StepRange{Int, ExpStep{Int}}, state = nothing)
    if state isa Nothing
        state = range.start
    end
    state > range.stop && return nothing

    # TODO
    return (Int(range.step.base ^ float(state)), state+1)
end

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

function save end

function save(; kwargs...)::Function
    return x -> save(x; kwargs...)
end

function load end

function id(nt::C)::UUID where {C <: NamedTuple}
    return uuid5(UUID(hash(nt)), "config")
end

function created_at end

const FILENAME_DATETIME_FORMAT = "YYYYmmdd-HHMMSS"

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

end # module
