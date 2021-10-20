module Utilities

###=============================================================================
### Imports
###=============================================================================

# Range steps
import Base: length, steprange_last, iterate

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
### Range steps
###-----------------------------------------------------------------------------

# TODO https://github.com/cursorinsight/FeatureScreening.jl/issues/14
abstract type AbstractStep end

##------------------------------------------------------------------------------
## Exponential range step
##------------------------------------------------------------------------------

# TODO https://github.com/cursorinsight/FeatureScreening.jl/issues/15
struct ExpStep{T} <: AbstractStep
    base::T

    function ExpStep(base::T) where {T}
        @assert 1 < base
        return new{T}(base)
    end
end

function length(range::StepRange{Int, <: ExpStep})
    return range.stop - range.start + 1
end

function (::Colon)(start::Real, step::S, stop::Real) where {S <: ExpStep}
    return StepRange{Int, S}(ceil(Int, log(step.base, start)), step, stop)
end

function steprange_last(start, step::ExpStep, stop)
    if iszero(stop)
        return -1
    else
        return floor(Int, log(step.base, stop))
    end
end

function iterate(range::StepRange{Int, ExpStep{Int}}, state = nothing)
    if state isa Nothing
        state = range.start
    end
    state > range.stop && return nothing

    return (Int(range.step.base ^ float(state)), state+1)
end

##------------------------------------------------------------------------------
## Fix size range step
##------------------------------------------------------------------------------

struct Size{T}
    n::T
end

function (::Colon)(start::Real, size::S, stop::Real) where {S <: Size}
    return range(start, stop; length = size.n)
end

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

function save(; kwargs...)::Function
    return x -> save(x; kwargs...)
end

function save(x; directory = ".")::Nothing
    save(path(directory, x), x)
    return nothing
end

function save(path::AbstractString, x)::Nothing
    open(path, "w") do io
        save(io, x)
    end
    return nothing
end

function save(io::IO, kvs::AbstractVector{<: Pair})::Nothing
    println.(Ref(io), join.(kvs, "; "))
    return nothing
end

function save(io::IO, x)::Nothing
    println(io, x)
    return nothing
end

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
### Dumping
###-----------------------------------------------------------------------------
# TODO https://github.com/cursorinsight/FeatureScreening.jl/issues/16

"""
`ENV` variable is the source of truth.
"""
function dumping!(enable::Bool)::Nothing
    if enable
        dumping!()
    else
        delete!(ENV, "DUMP")
    end

    return nothing
end

function dumping!(directory::AbstractString = ".")::Nothing
    @assert ispath(directory)
    ENV["DUMP"] = directory
    return nothing
end

macro dump(arguments...)
    return dump__(arguments...) |> esc
end

function dump__(arguments...)
    return quote
        $(dump__def(arguments...))
        if haskey(ENV, "DUMP")
            mkpath(ENV["DUMP"])
            $(dump__save(arguments...))
        end
    end
end

function dump__def(filename, object::O)::O where {O}
    return object
end

function dump__def(object::O)::O where {O}
    return object
end

function dump__save(object)::Expr
    variable::Symbol = __variable(object)
    return :($save($path(ENV["DUMP"], $variable), $variable))
end

function dump__save(filename, object)::Expr
    variable::Symbol = __variable(object)
    return :($save($path(ENV["DUMP"], $filename), $variable))
end

function __variable(variable::Symbol)::Symbol
    return variable
end

function __variable(expr::Expr)::Symbol
    @assert expr.head == :(=)
    object = expr.args[1]

    if object isa Symbol                # case `x = 1`
        return object
    elseif object isa Expr              # case `x::Int = 1`
        @assert object.head == :(::)
        return object.args[1]
    else                                # case "invalid"
        throw(:invalid_variable_expression => object)
    end
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
### Rest
###-----------------------------------------------------------------------------

const Maybe{T} = Union{Nothing, T}

function partition(xs::AbstractVector{X},
                   n::Int;
                   rest::Bool = false
                  )::Vector{Vector{X}} where {X}
    # TODO https://github.com/cursorinsight/FeatureScreening.jl/issues/17
    m::Int = (rest ? ceil : floor)(length(xs) / n)
    return [xs[(n*(i-1)+1):min(n*i, length(xs))] for i in 1:m]
end

make_rng(rng::AbstractRNG = GLOBAL_RNG)::AbstractRNG = rng
make_rng(seed::Integer)::AbstractRNG = MersenneTwister(seed)

function skip(args...; kwargs...)::Nothing
    return nothing
end

const FILENAME_DATETIME_FORMAT = "YYYYmmdd-HHMMSS"

end # module
