###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening.Types: FeatureSet, names
using FeatureScreening: screen

###=============================================================================
### Tests
###=============================================================================

@testset "Basics" begin

    @testset "API #1" begin
        LABEL_COUNT::Int    = 5
        SAMPLE_COUNT::Int   = 5
        FEATURE_COUNT::Int  = 11
        STEP_SIZE::Int      = 3
        REDUCED_SIZE::Int   = 3

        selected = screen(fixture(:feature_set);
                          reduced_size  = REDUCED_SIZE,
                          step_size     = STEP_SIZE,
                          config        = fixture(:config, :screen),
                          show_progress = false)

        @test selected isa FeatureSet{Symbol, String, Float64}
        @test LABEL_COUNT * SAMPLE_COUNT == size(selected, 1)
        @test REDUCED_SIZE == size(selected, 2)
        @test_skip names(selected) ⊆ ["8", "9", "10", "11"]
    end

    @testset "API #2" begin
        LABEL_COUNT::Int    = 5
        SAMPLE_COUNT::Int   = 5
        FEATURE_COUNT::Int  = 11
        STEP_SIZE::Int      = 3
        REDUCED_SIZE::Int   = 3

        selected = screen(fixture(:X),
                          fixture(:y);
                          reduced_size  = REDUCED_SIZE,
                          step_size     = STEP_SIZE,
                          config        = fixture(:config, :screen),
                          show_progress = false)

        @test selected isa FeatureSet{Symbol, Int, Float64}
        @test LABEL_COUNT * SAMPLE_COUNT == size(selected, 1)
        @test REDUCED_SIZE == size(selected, 2)
        @test_skip names(selected) ⊆ 8:11
    end

end
