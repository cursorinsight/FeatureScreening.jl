module Utilities

###=============================================================================
### Imports
###=============================================================================

using DecisionTree: Ensemble as RandomForest

using DecisionTree: build_forest, nfoldCV_forest

###=============================================================================
### API
###=============================================================================

const Maybe{T} = Union{Nothing, T}

# TODO be careful with the name collision with `IterTools.partition`
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

function _first(range::StepRange{Int, ExpStep{Int}})
    return ceil(log(range.step.base, range.start))
end

function _last(range::StepRange{Int, ExpStep{Int}})
    return floor(log(range.step.base, range.stop))
end

function Base.first(range::StepRange{Int, ExpStep{Int}})
    return _first(range) ^ range.step.base
end

function Base.last(range::StepRange{Int, ExpStep{Int}})
    return _last(range) ^ range.step.base
end

# TODO
function Base.length(range::StepRange{Int, ExpStep{Int}})
    return _last(range) - _first(range) + 1
end

function (::Colon)(start::Real, step::S, stop::Real) where {S <: ExpStep}
    return StepRange{Int, S}(start, step, stop)
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
        state = ceil(Int, log(step.base, range.start))
    end
    state > range.stop && return nothing

    # TODO
    return (Int(range.step.base ^ float(state)), state+1)
end

struct Size{T}
    n::T
end

function (::Colon)(start::Real, size::S, stop::Real) where {S <: Size}
    return range(start, stop; length = size.n)
end

const DEFAULT_BUILD_FOREST_CONFIG =
    (n_subfeatures          = -1,
     n_trees                = 10,
     partial_sampling       = 0.7,
     max_depth              = -1,
     min_samples_leaf       = 1,
     min_samples_split      = 2,
     min_purity_increase    = 0.0)

function _build_forest(labels::AbstractVector{L},
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

function _nfoldCV_forest(labels::AbstractVector,
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

end # module
