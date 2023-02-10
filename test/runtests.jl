###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using Test
using Random: seed!

###=============================================================================
### Tests
###=============================================================================

# Fixtures
include("Fixtures.jl")
using .Fixtures: fixture

# pad test summaries to equal length
Test.get_alignment(::Test.DefaultTestSet, ::Int) = 30

# fixed random seed
seed!(1)

include("test.utilities.jl")
include("test.basics.jl")
include("test.feature-set.jl")
include("test.importance.jl")
