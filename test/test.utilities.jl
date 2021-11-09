###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening.Utilities: ExpStep, Size
using FeatureScreening.Utilities: @dump, dumping!

###=============================================================================
### Tests
###=============================================================================

@testset "Utilities" begin

    @testset "Exponential range" begin
        @test [1]           == (1:ExpStep(2):1)
        @test [1, 2]        == (1:ExpStep(2):2)
        @test [1, 2]        == (1:ExpStep(2):3)
        @test [1, 2, 4]     == (1:ExpStep(2):4)
        @test [1, 2, 4]     == (1:ExpStep(2):5)
        @test [1, 2, 4]     == (1:ExpStep(2):6)
        @test [1, 2, 4]     == (1:ExpStep(2):7)
        @test [1, 2, 4, 8]  == (1:ExpStep(2):8)

        @test []                == (1:Size(0):5)
        @test_throws ArgumentError (1:Size(1):5)
        @test [1, 5]            == (1:Size(2):5)
        @test [1, 3, 5]         == (1:Size(3):5)
    end

    @testset "Dump" begin
        mktempdir() do directory
            let
                # disabled definition dumping
                @dump "2.txt" x = 1 + 1
                @test x == 2
                @test !isfile("$directory/2.txt")
                # disabled stand-alone dumping
                @dump "two.txt" x
                @test !isfile("$directory/two.txt")
            end

            dumping!(directory)

            let
                # enabled definition dumping
                @dump "2.txt" x = 1 + 1
                @test x == 2
                @test isfile("$directory/2.txt")
                @test readlines("$directory/2.txt") == ["2"]
                # enabled stand-alone dumping
                @dump "two.txt" x
                @test isfile("$directory/two.txt")
                @test readlines("$directory/two.txt") == ["2"]
            end

            dumping!(false)

            let
                # disabled definition dumping
                @dump "3.txt" x = 1 + 1 + 1
                @test x == 3
                @test !isfile("$directory/3.txt")
                # disabled stand-alone dumping
                @dump "three.txt" x
                @test !isfile("$directory/three.txt")
            end
        end
    end
end
