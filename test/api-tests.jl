###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening: FeatureSet, FeatureSubset, feature_names

using FeatureScreening: screen, accuracies

###=============================================================================
### Tests
###=============================================================================

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
                for i in 1:USER_COUNT]

    @testset "Programming API" begin
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

    @testset "Data science API" begin
        X = [feature_vector'
             for (label, feature_vectors) in features
             for feature_vector in feature_vectors] |> Base.splat(vcat)
        y = [label
             for (label, feature_vectors) in features
             for _ in feature_vectors]
        selected = screen(X, y;
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

end
