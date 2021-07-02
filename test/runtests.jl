###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using Test

using FeatureScreening: FeatureSet, names

# Utility tests
using FeatureScreening.Utilities: partition, ExpStep, Size

# API tests
using FeatureScreening: screen, accuracies

using Random: seed!, shuffle

###=============================================================================
### Tests
###=============================================================================

seed!(1)

###=============================================================================
### Tests
###=============================================================================

@testset "`FeatureSet` function" begin
    let feature_set = FeatureSet([1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 ["feature1", "feature2"],
                                 [10 11;
                                  11 11;
                                  13 12;
                                  21 20;
                                  22 21;
                                  21 22;
                                  33 32;
                                  31 34;
                                  39 30]),
        feature_subset = feature_set[["feature1"]]

        @test feature_subset isa FeatureSet
        @test [(1, [10, 11]),
               (1, [11, 11]),
               (1, [13, 12]),
               (2, [21, 20]),
               (2, [22, 21]),
               (2, [21, 22]),
               (3, [33, 32]),
               (3, [31, 34]),
               (3, [39, 30])
              ] == map(identity, feature_set)

        @test [(1, [10]),
               (1, [11]),
               (1, [13]),
               (2, [21]),
               (2, [22]),
               (2, [21]),
               (3, [33]),
               (3, [31]),
               (3, [39])
              ] == map(identity, feature_subset)
    end

    let feature_set = rand(FeatureSet, 80, 30)
        @test feature_set isa FeatureSet{Int, Int, Float64}
        @test size(feature_set) == (80, 30)
    end

    let feature_set = rand(FeatureSet{Char, Int, Float64}, 80, 30)
        @test feature_set isa FeatureSet{Char, Int, Float64}
        @test size(feature_set) == (80, 30)
    end

end

@testset "Utilities" begin

    @testset "Partition" begin
        @test [11:11, 12:12, 13:13, 14:14]  == partition(11:14, 1)
        @test [11:12, 13:14]                == partition(11:14, 2)
        @test [11:13]                       == partition(11:14, 3)
        @test [11:14]                       == partition(11:14, 4)

        @test [11:11, 12:12, 13:13, 14:14]  == partition(11:14, 1; rest = true)
        @test [11:12, 13:14]                == partition(11:14, 2; rest = true)
        @test [11:13, 14:14]                == partition(11:14, 3; rest = true)
        @test [11:14]                       == partition(11:14, 4; rest = true)
    end

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
end

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
                          config       = config_screen,
                          after        = accuracies(config = config_test))

        @test selected isa FeatureSet{Symbol, String, Float64}
        @test LABEL_COUNT * SAMPLE_COUNT == size(selected, 1)
        @test REDUCED_SIZE == size(selected, 2)
        @test names(selected) ⊆ ["8", "9", "10", "11"]
    end

    @testset "API #2" begin
        selected = screen(X, y;
                          reduced_size = REDUCED_SIZE,
                          step_size    = STEP_SIZE,
                          config       = config_screen,
                          after        = accuracies(config = config_test))

        @test selected isa FeatureSet{Symbol, Int, Float64}
        @test LABEL_COUNT * SAMPLE_COUNT == size(selected, 1)
        @test REDUCED_SIZE == size(selected, 2)
        @test names(selected) ⊆ 8:11
    end

end

include("loader-tests.jl")
