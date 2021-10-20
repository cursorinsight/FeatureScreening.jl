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

    # TODO some computed config values from input dimensions
    config_screen = (n_subfeatures          = -1,
                     n_trees                = 20,
                     partial_sampling       = 0.9,
                     max_depth              = -1,
                     min_samples_leaf       = 2,
                     min_samples_split      = 3,
                     min_purity_increase    = 0.0)

    config_test   = (n_subfeatures          = -1,
                     n_trees                = 10,
                     partial_sampling       = 0.8,
                     max_depth              = -1,
                     min_samples_leaf       = 1,
                     min_samples_split      = 2,
                     min_purity_increase    = 0.01)

    LABEL_COUNT::Int    = 5
    SAMPLE_COUNT::Int   = 5
    FEATURE_COUNT::Int  = 11
    STEP_SIZE::Int      = 3
    REDUCED_SIZE::Int   = 3

    y::Vector{Symbol} =
        [:a, :a, :a, :a, :a,
         :b, :b, :b, :b, :b,
         :c, :c, :c, :c, :c,
         :d, :d, :d, :d, :d,
         :e, :e, :e, :e, :e]

    feature_names::Vector{String} =
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

    X::Matrix{Float64} =
        [0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0]

    X .+= randn(LABEL_COUNT * SAMPLE_COUNT, FEATURE_COUNT)

    feature_set = FeatureSet(y, feature_names, X)

    @testset "API #1" begin
        selected = screen(feature_set;
                          reduced_size = REDUCED_SIZE,
                          step_size    = STEP_SIZE,
                          config       = config_screen)

        @test selected isa FeatureSet{Symbol, String, Float64}
        @test LABEL_COUNT * SAMPLE_COUNT == size(selected, 1)
        @test REDUCED_SIZE == size(selected, 2)
        @test names(selected) ⊆ ["8", "9", "10", "11"]
    end

    @testset "API #2" begin
        selected = screen(X, y;
                          reduced_size = REDUCED_SIZE,
                          step_size    = STEP_SIZE,
                          config       = config_screen)

        @test selected isa FeatureSet{Symbol, Int, Float64}
        @test LABEL_COUNT * SAMPLE_COUNT == size(selected, 1)
        @test REDUCED_SIZE == size(selected, 2)
        @test names(selected) ⊆ 8:11
    end

end
