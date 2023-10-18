###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using Test

using Aqua: test_all as aqua
using Random: seed!

import FeatureScreening

###=============================================================================
### Tests
###=============================================================================

# pad test summaries to equal length
Test.get_alignment(::Test.DefaultTestSet, ::Int) = 30

# fixed random seed
seed!(1)

##------------------------------------------------------------------------------
## Aqua module tests
##------------------------------------------------------------------------------

@testset "Aqua" begin
    aqua(FeatureScreening;
         # StatsBase introduces some ambiguities
         ambiguities = false,
         # DocOpt is only used in `screen.jl`, not in the module itself
         stale_deps = (ignore = [:DocOpt],))
end

##------------------------------------------------------------------------------
## Includes
##------------------------------------------------------------------------------

# Fixtures
include("Fixtures.jl")
using .Fixtures: fixture

include("test.basics.jl")
include("test.importance.jl")
