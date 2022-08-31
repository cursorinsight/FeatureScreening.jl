###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening.Types: FeatureSet
using FeatureScreening.Utilities: id
using FeatureScreening.Types: labels, names, features
using FeatureScreening.Types: save, load
using HDF5: ishdf5, h5open, File

###=============================================================================
### Tests
###=============================================================================

@testset "`FeatureSet` function" begin

    let feature_set = FeatureSet([1, 1, 1, 2, 2, 2, 3, 3, 3],
                                 "feature" .* ('1':'4'),
                                 [10 11 12 13;
                                  11 11 12 13;
                                  13 12 11 12;
                                  21 20 23 24;
                                  22 21 22 23;
                                  21 22 22 22;
                                  33 32 31 32;
                                  31 34 31 34;
                                  39 30 30 32]),

        feature_subset = feature_set[:, ["feature2", "feature3", "feature4"]]
        @test feature_subset isa FeatureSet
        @test labels(feature_subset) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
        @test names(feature_subset) == ["feature2", "feature3", "feature4"]
        @test features(feature_subset) == [11 12 13;
                                           11 12 13;
                                           12 11 12;
                                           20 23 24;
                                           21 22 23;
                                           22 22 22;
                                           32 31 32;
                                           34 31 34;
                                           30 30 32]
        @test [(1, [11, 12, 13]),
               (1, [11, 12, 13]),
               (1, [12, 11, 12]),
               (2, [20, 23, 24]),
               (2, [21, 22, 23]),
               (2, [22, 22, 22]),
               (3, [32, 31, 32]),
               (3, [34, 31, 34]),
               (3, [30, 30, 32])
              ] == collect(eachrow(feature_subset))

        feature_subset2 = feature_subset[1:5, ["feature2", "feature3"]]
        @test feature_subset2 isa FeatureSet
        @test labels(feature_subset2) == [1, 1, 1, 2, 2]
        @test names(feature_subset2) == ["feature2", "feature3"]
        @test features(feature_subset2) == [11 12;
                                           11 12;
                                           12 11;
                                           20 23;
                                           21 22]
        @test [(1, [11, 12]),
               (1, [11, 12]),
               (1, [12, 11]),
               (2, [20, 23]),
               (2, [21, 22])
              ] == collect(eachrow(feature_subset2))

        feature_subset3 = feature_subset[:, :]
        @test feature_subset3 isa FeatureSet
        @test feature_subset == feature_subset3

        feature_subset4 = feature_subset3[1:end, :]
        @test feature_subset4 isa FeatureSet
        @test feature_subset3 == feature_subset4
    end

    let feature_set = rand(FeatureSet, 80, 30)
        @test feature_set isa FeatureSet{Int, Int, Float64}
        @test size(feature_set) == (80, 30)
        @test axes(feature_set, 1) == 1:80
        @test axes(feature_set, 2) == 1:30
        @test axes(feature_set) == (1:80, 1:30)
    end

    let feature_set = rand(FeatureSet{Char, Int, Float64}, 80, 30)
        @test feature_set isa FeatureSet{Char, Int, Float64}
        @test size(feature_set) == (80, 30)
        @test axes(feature_set, 1) == 1:80
        @test axes(feature_set, 2) == 1:30
        @test axes(feature_set) == (1:80, 1:30)
    end

end

const FEATURE_SETS =
    ["feature set" => rand(FeatureSet, 50, 30; label_count = 10),
     "feature subset" => rand(FeatureSet, 50, 30; label_count = 10)[:, 1:5]]

@testset "Save and load $name" for (name, feature_set) in FEATURE_SETS
    mktempdir() do directory::String
        # Save
        @test_logs (:info, "Created file") save(feature_set; directory)

        path = "$directory/$(id(feature_set)).hdf5"

        # File storage content
        @test isfile(path)
        @test ishdf5(path)

        h5open(path, "r") do f
            @test f isa File
            @test ["id", "created_at", "names", "labels", "features"] ⊆ keys(f)
        end

        # Load
        loaded::FeatureSet = load(FeatureSet, path)
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
