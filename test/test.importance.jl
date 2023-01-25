###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening: feature_importance
using FeatureScreening.Types: FeatureSet, names
using FeatureScreening: select, Top, Random, IndexBased, get_count
using FeatureScreening: label, importance
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

    @testset "Selection -- Top" begin
        feature_importances = [4 => 12,
                               3 => 11,
                               123 => 3,
                               33 => 1]

        # Top count selector method
        let result = select(feature_importances, Top(3)) .|> label
            @test result isa Vector{Int}
            @test result == [4, 3, 123]
        end

        # Top count selector method in strict mode
        @test_throws AssertionError select(feature_importances,
                                           Top(10);
                                           strict = true) .|> label

        # Top count selector method without strict mode
        let result = select(feature_importances,
                            Top(10);
                            strict = false) .|> label
            @test result isa Vector{Int}
            @test result == [4, 3, 123, 33]
        end

        # Top ratio selector method
        let result = select(feature_importances, Top(0.25)) .|> label
            @test result isa Vector{Int}
            @test result == [4]
        end

        # Top ratio selector method in strict mode
        @test_throws AssertionError select(feature_importances,
                                           Top(3.1);
                                           strict = true) .|> label

        # Top ratio selector method without strict mode
        let result = select(feature_importances,
                            Top(3.1);
                            strict = false) .|> label
            @test result isa Vector{Int}
            @test result == [4, 3, 123, 33]
        end
    end

    @testset "Selection -- Random" begin
        feature_importances = [4 => 12,
                               3 => 11,
                               123 => 3,
                               33 => 1]

        # Random selector method
        let result = select(feature_importances,
                            Random(3, StableRNG(1))) .|> label
            @test result isa Vector{Int}
            @test result == [33, 123, 4]
        end

        # Random selector in strict mode
        @test_throws AssertionError select(feature_importances,
                                           Random(10);
                                           strict = true) .|> label

        # Random selector without strict mode
        let result = select(feature_importances,
                            Random(10, StableRNG(1));
                            strict = false) .|> label
            @test result isa Vector{Int}
            @test result == [33, 123, 4, 33]
        end

        # Random ratio selector method
        let result = select(feature_importances,
                            Random(0.77, StableRNG(1))) .|> label
            @test result isa Vector{Int}
            @test result == [33, 123, 4]
        end

        # Random ratio selector in strict mode
        @test_throws AssertionError select(feature_importances,
                                           Random(3.1);
                                           strict = true) .|> label

        # Random ratio selector without strict mode
        let result = select(feature_importances,
                            Random(3.1, StableRNG(1));
                            strict = false) .|> label
            @test result isa Vector{Int}
            @test result == [33, 123, 4, 33]
        end
    end

    @testset "Selection -- IndexBased" begin
        feature_importances = [4 => 12,
                               3 => 11,
                               123 => 3,
                               33 => 1,
                               5 => 1,
                               7 => 1,
                               9 => 0]

        # Index based selector method
        let result = select(feature_importances,
                            IndexBased([3, 1, 2, 1])) .|> label
            @test result isa Vector{Int}
            @test result == [123, 4, 3, 4]
        end

        # Index based selector method
        let result = select(feature_importances, IndexBased(2:2:6)) .|> label
            @test result isa Vector{Int}
            @test result == [3, 33, 7]
        end

        # Index based selector method in strict mode
        @test_throws AssertionError select(feature_importances,
                                           IndexBased(1:100);
                                           strict = true) .|> label

        # Index based selector method without strict mode
        let result = select(feature_importances,
                            IndexBased(1:100);
                            strict = false) .|> label
            @test result isa Vector{Int}
            @test result == [4, 3, 123, 33, 5, 7, 9]
        end
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
