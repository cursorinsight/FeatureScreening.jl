###-----------------------------------------------------------------------------
### Copyright (C) FeatureScreening.jl
###
### SPDX-License-Identifier: MIT License
###-----------------------------------------------------------------------------

module Utilities

###=============================================================================
### Imports
###=============================================================================

using Random: AbstractRNG, GLOBAL_RNG, MersenneTwister

###=============================================================================
### Exports
###=============================================================================

export Maybe, make_rng

###=============================================================================
### API
###=============================================================================

const Maybe{T} = Union{Nothing, T}

function make_rng(rng::AbstractRNG = GLOBAL_RNG)::AbstractRNG
    return rng
end

function make_rng(seed::Integer)::AbstractRNG
    return MersenneTwister(seed)
end

end # module
