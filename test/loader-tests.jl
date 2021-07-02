###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening.Types: FeatureSet
using FeatureScreening.Types: labels, names, features
using FeatureScreening.Types: save, load

###=============================================================================
### Test sets
###=============================================================================

@testset "Save and load feature set" begin

    filename = tempname() * ".hdf5"

    let feature_set = rand(FeatureSet, 50, 30; label_count = 10)
        save(feature_set, filename)
        @test isfile(filename)
    end

    let feature_set = load(filename)

        @test feature_set isa FeatureSet
        @test size(feature_set) == (50, 30)

        for (label, features) in feature_set
            @test label isa Int
            @test label in 1:10
            @test features isa AbstractVector{Float64}
            @test length(features) == 30
        end
    end

end

@testset "Save and load feature subset" begin

    filename = tempname() * ".hdf5"

    let feature_set = rand(FeatureSet, 50, 30; label_count = 10),
        feature_subset = feature_set[1:5]

        save(feature_subset, filename)
        @test isfile(filename)
    end

    let feature_set = load(filename)

        @test feature_set isa FeatureSet
        @test size(feature_set) == (50, 5)

        for (label, features) in feature_set
            @test label isa Int
            @test label in 1:10
            @test features isa AbstractVector{Float64}
            @test length(features) == 5
        end
    end

end
