###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using Test

using FeatureScreening: FeatureSet, FeatureSubset, feature_names

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

@testset "`AbstractFeatureSet` function" begin
    let feature_set = FeatureSet([1 => [[10, 11], [11, 11], [13, 12]],
                                  2 => [[21, 20], [22, 21], [21, 22]],
                                  3 => [[33, 32], [31, 34], [39, 30]]]),
        feature_subset = feature_set[[1]]

        @test [1 => [10, 11], 1 => [11, 11], 1 => [13, 12],
               2 => [21, 20], 2 => [22, 21], 2 => [21, 22],
               3 => [33, 32], 3 => [31, 34], 3 => [39, 30]
              ] == map(identity, feature_set)

        @test [1 => [10], 1 => [11], 1 => [13],
               2 => [21], 2 => [22], 2 => [21],
               3 => [33], 3 => [31], 3 => [39]
              ] == map(identity, feature_subset)
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
    USER_COUNT::Int = 20
    SAMPLE_COUNT::Int = 20
    FEATURE_COUNT::Int = 20
    STEP_SIZE::Int = 6
    REDUCED_SIZE::Int = 10

    # TODO some computed config values from input dimensions
    config_screen = (n_subfeatures          = -1,
                     n_trees                = 1000,
                     partial_sampling       = 0.9,
                     max_depth              = -1,
                     min_samples_leaf       = 10,
                     min_samples_split      = 10,
                     min_purity_increase    = 0.0)

    config_test   = (n_subfeatures          = -1,
                     n_trees                = 2000,
                     partial_sampling       = 0.8,
                     max_depth              = -1,
                     min_samples_leaf       = 1,
                     min_samples_split      = 2,
                     min_purity_increase    = 0.01)

    features = ["$i" => [randn(FEATURE_COUNT) .+ (0:Size(FEATURE_COUNT):i/2)
                         for _ in 1:SAMPLE_COUNT]
                for i in 1:USER_COUNT] |> FeatureSet

    selected = screen(features;
                      reduced_size = REDUCED_SIZE,
                      step_size    = STEP_SIZE,
                      config       = config_screen,
                      after        = accuracies(config = config_test))

    @test selected isa FeatureSubset{String, Int, Float64}
    @test USER_COUNT * SAMPLE_COUNT == size(selected, 1)
    @test REDUCED_SIZE == size(selected, 2)

    for feature_name in feature_names(selected)
        @test FEATURE_COUNT-2*REDUCED_SIZE <= feature_name <= FEATURE_COUNT
    end
end
