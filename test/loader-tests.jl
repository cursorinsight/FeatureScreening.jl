###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening.Types: FeatureSet
using FeatureScreening.Types: id, labels, names, features
using FeatureScreening.Types: save, load

###=============================================================================
### Test sets
###=============================================================================

const FEATURE_SETS =
    [rand(FeatureSet, 50, 30; label_count = 10),
     rand(FeatureSet, 50, 30; label_count = 10)[:, 1:5]]

@testset "Save and load feature set" for feature_set in FEATURE_SETS

    mktempdir() do directory::String

        save(feature_set; directory)
        @test isfile("$directory/$(id(feature_set)).hdf5")

        loaded::FeatureSet = load(FeatureSet, "$directory/$(id(feature_set)).hdf5")
        @test feature_set isa FeatureSet
        @test size(loaded) == size(feature_set)

        for (label, features) in loaded
            @test label isa Int
            @test label in labels(feature_set)
            @test features isa AbstractVector{Float64}
            @test length(features) == size(feature_set, 2)
        end
    end

end
