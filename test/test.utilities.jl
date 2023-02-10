###-----------------------------------------------------------------------------
### Copyright (C) FeatureScreening.jl
###
### SPDX-License-Identifier: MIT License
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening.Utilities: @with_pattern, @pattern

###=============================================================================
### Tests
###=============================================================================

@testset "Utilities" begin
    @with_pattern function f end

    @test f isa Function
    @test length(methods(f)) == 1 # there is a default wrapper for
    @test hasmethod(f, Tuple{Vararg{Symbol}})

    @pattern function f(:x)
        return "x"
    end

    @test length(methods(f)) == 2
    @test hasmethod(f, Tuple{Val{:x}})

    @pattern function f(:y)
        return 'y'
    end

    @test length(methods(f)) == 3
    @test hasmethod(f, Tuple{Val{:y}})

    # Couldn't create pattern with return type assertion
    try
        # TODO this is how you can catch exception from macro
        quote
            @pattern function f(:y)::Int
                return 3
            end
        end |> eval
    catch exception
        # TODO refactor if previous `eval` was removed
        @test exception isa LoadError
        @test exception.error isa AssertionError
    end
end
