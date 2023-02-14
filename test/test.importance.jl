###-----------------------------------------------------------------------------
### Copyright (C) FeatureScreening.jl
###
### SPDX-License-Identifier: MIT License
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening: feature_importance
using FeatureScreening.Types: FeatureSet, names
using FeatureScreening: SelectTop, SelectRandom, SelectByImportance
using FeatureScreening: select, get_count, label, importance
using StableRNGs: StableRNG

###=============================================================================
### Testcases
###=============================================================================

@testset "Importance" begin

    @testset "Feature importance" begin
        feature_set::FeatureSet = fixture(:feature_set)
        feature_importances = feature_importance(feature_set)
        @test label.(feature_importances) ⊆ names(feature_set)
        @test importance.(feature_importances) isa Vector{Int}
        @test all(0 .< importance.(feature_importances))
    end

    @testset "SelectTop" begin
        feature_importances = [4 => 12,
                               3 => 11,
                               123 => 3,
                               33 => 1]

        # Top count selection method
        let result = select(feature_importances, SelectTop(3)) .|> label
            @test result isa Vector{Int}
            @test result == [4, 3, 123]
        end

        # Top count selection method in strict mode
        @test_throws AssertionError select(feature_importances,
                                           SelectTop(10; strict = true))

        # Top count selection method without strict mode
        let result = select(feature_importances,
                            SelectTop(10; strict = false)) .|> label
            @test result isa Vector{Int}
            @test result == [4, 3, 123, 33]
        end

        # Top ratio selection method
        let result = select(feature_importances, SelectTop(0.25)) .|> label
            @test result isa Vector{Int}
            @test result == [4]
        end

        # Top ratio selection method in strict mode
        @test_throws AssertionError select(feature_importances,
                                           SelectTop(3.1; strict = true))

        # Top ratio selection method without strict mode
        let result = select(feature_importances,
                            SelectTop(3.1; strict = false)) .|> label
            @test result isa Vector{Int}
            @test result == [4, 3, 123, 33]
        end
    end

    @testset "SelectRandom" begin
        feature_importances = [4 => 12,
                               3 => 11,
                               123 => 3,
                               33 => 1]

        # Random selection method
        let result = select(StableRNG(2),
                            feature_importances,
                            SelectRandom(3)) .|> label
            @test result isa Vector{Int}
            @test result == [4, 3, 33]
        end

        # Random selection in strict mode
        @test_throws AssertionError select(feature_importances,
                                           SelectRandom(10; strict = true))

        # Random selection without strict mode
        let result = select(StableRNG(2),
                            feature_importances,
                            SelectRandom(10; strict = false)) .|> label
            @test result isa Vector{Int}
            @test result == label.(feature_importances)
        end

        # Random selection method with replacement
        let result = select(StableRNG(1),
                            feature_importances,
                            SelectRandom(3; replace = true)) .|> label
            @test result isa Vector{Int}
            @test result == [4, 123, 123]
        end

        # Random ratio selection method
        let result = select(StableRNG(2),
                            feature_importances,
                            SelectRandom(0.77)) .|> label
            @test result isa Vector{Int}
            @test result == [4, 3, 33]
        end

        # Random ratio selection in strict mode
        @test_throws AssertionError select(feature_importances,
                                           SelectRandom(3.1; strict = true))

        # Random ratio selection without strict mode
        let result = select(StableRNG(2),
                            feature_importances,
                            SelectRandom(3.1; strict = false)) .|> label
            @test result isa Vector{Int}
            @test result == label.(feature_importances)
        end

        # Random ratio selection method with replacement
        let result = select(StableRNG(1),
                            feature_importances,
                            SelectRandom(0.77; replace = true)) .|> label
            @test result isa Vector{Int}
            @test result == [4, 123, 123]
        end
    end

    @testset "SelectByImportance" begin
        feature_importances = [4 => 12,
                               3 => 11,
                               123 => 3,
                               33 => 1]

        # Weighted random selection method
        let result = select(StableRNG(2),
                            feature_importances,
                            SelectByImportance(3)) .|> label
            @test result isa Vector{Int}
            @test result == [4, 3, 123]
        end

        # Weighted random selection in strict mode
        @test_throws AssertionError select(feature_importances,
                                           SelectByImportance(10;
                                                              strict = true))

        # Weighted random selection without strict mode
        let result = select(StableRNG(2),
                            feature_importances,
                            SelectByImportance(10; strict = false)) .|> label
            @test result isa Vector{Int}
            @test result == label.(feature_importances)
        end

        # Weighted random ratio selection method
        let result = select(StableRNG(2),
                            feature_importances,
                            SelectByImportance(0.77)) .|> label
            @test result isa Vector{Int}
           @test result == [4, 3, 123]
        end

        # Weighted random ratio selection in strict mode
        @test_throws AssertionError select(feature_importances,
                                           SelectByImportance(3.1;
                                                              strict = true))

        # Weighted random ratio selection without strict mode
        let result = select(StableRNG(2),
                            feature_importances,
                            SelectByImportance(3.1; strict = false)) .|> label
            @test result isa Vector{Int}
            @test result == label.(feature_importances)
        end
    end

    @testset "ComposedSelectionMode" begin
        result = select(1:100, SelectRandom(10) ∘ SelectTop(50))
        @test length(result) == 10
        @test all(<=(50), result)
        @test issorted(result)
        @test result != 1:10
    end

    @testset "Selection utility -- Get count" begin
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
        @test get_count(1:100, 1//3) == 33 # Rational
        @test get_count(1:100, π; strict = false) == 100 # Irrational

        @test get_count(1:100, 0.25) == 25
        @test get_count(1:100, 1/3) == 33
    end

end
