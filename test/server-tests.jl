###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================


using FeatureScreening.Server: Link # TODO
using FeatureScreening.Server: start, apply, get_value, stop
using FeatureScreening.Server.Callbacks: Action
import FeatureScreening.Server.Callbacks: execute

function increment(n::Int)::Int
    sleep(1)
    return n+1
end

struct Increment <: Action end
execute(::Increment, n::Int) = increment(n)

@testset "Application" begin

    server = start(42)

    @test server isa Link
    @test get_value(server) == 42

    # Function based execution
    @test apply(server, increment) == 43
    @test get_value(server) == 43

    apply(server, increment; async = true)
    @test get_value(server) == 43
    sleep(1.1)
    @test get_value(server) == 44

    # "Action" based execution
    @test apply(server, Increment()) == 45
    @test get_value(server) == 45

    apply(server, Increment(); async = true)
    @test get_value(server) == 45
    sleep(1.1)
    @test get_value(server) == 46

    stop(server)

end
