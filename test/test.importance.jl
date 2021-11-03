###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening: select, Top, Random, get_count

###=============================================================================
### Testcases
###=============================================================================

@testset "Importance" begin

    @testset "Selection" begin
        feature_importances = [4 => 12,
                               3 => 11,
                               123 => 3,
                               33 => 1]

        # Top count selector method
        let result = select(feature_importances, Top(3))
            @test result isa Vector{Int}
            @test result == [4, 3, 123]
        end

        # Top ratio selector method
        let result = select(feature_importances, Top(0.25))
            @test result isa Vector{Int}
            @test result == [4]
        end

        # Random selector method
        let result = select(feature_importances, Random(3))
            @test result isa Vector{Int}
            @test result == [4, 123, 33]
        end

        # Random ratio selector method
        let result = select(feature_importances, Random(0.77))
            @test result isa Vector{Int}
            @test result == [4, 4, 33]
        end

        # Get count
        @test get_count(1:5, 0) == 0
        @test get_count(1:5, 1) == 1
        @test get_count(1:5, 5) == 5
        @test_throws AssertionError get_count(1:5, -1)
        @test_throws AssertionError get_count(1:5, 44)

        # Get count by ratio
        @test get_count(1:5, 0.0) == 0
        @test get_count(1:5, 1.0) == 5
        @test get_count(1:5, 0.5) == 2
        @test get_count(1:5, 0.3) == 1
        @test get_count(1:5, 0.9) == 4
        @test_throws AssertionError get_count(1:5, -0.5)
        @test_throws AssertionError get_count(1:5, 3.7)

        @test get_count(1:100, 0.25) == 25
        @test get_count(1:100, 1/3) == 33

    end

end
