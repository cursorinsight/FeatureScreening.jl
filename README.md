# FeatureScreening.jl

[![CI](https://github.com/cursorinsight/FeatureScreening.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/cursorinsight/FeatureScreening.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/cursorinsight/FeatureScreening.jl/branch/master/graph/badge.svg?token=P59CK9SA1Z)](https://codecov.io/gh/cursorinsight/FeatureScreening.jl)

FeatureScreening.jl is a Julia package that provides utilities for finding
significant features in a feature set, based on their importance in
classification.

## Context

Generally speaking, a feature set consists of:

* a set of feature names;
* a set of samples (observations), each belonging to one of several classes;
* a vector of feature values: one value for every feature of every observation.

Using the feature matrix and machine learning, it is possible to build a
classifier which predicts the class of a sample based on its feature values.
Some features will be more useful for this than others, some are just noise and
completely irrelevant. If the total number of features is huge (in the range of
several tens of thousands or even more), finding the most suitable subset that
can achieve the highest possible classification accuracy can be tricky and
time-consuming.

This library implements a random forest based, multi-round screening utility
that can perform a tournament-based sorting and selection of features based on
their importance.

## Installation

*Note:* FeatureScreening.jl depends on a library called [Dumper.jl][], which is,
for the time being, not added to the General registry. Due to limitations of the
Julia package manager, you need to add this dependency manually to your
projects.

```julia
julia>]
pkg> add https://github.com/cursorinsight/Dumper.jl
pkg> add https://github.com/cursorinsight/FeatureScreening.jl
```

## Usage

Load the package via

```julia
julia> using FeatureScreening
```

### Screening features

The `screen` function performs screening on a given feature set, and returns the
matrix of screened features. It provides three methods:

1. `screen(fs::FeatureSet; options...)`
2. `screen(labels::AbstractVector{L},
   names::AbstractVector{N},
   features::AbstractMatrix{F}; options...)`
3. `screen(X::AbstractMatrix{F}, y::AbstractVector{L}; options...)`

The first signature accepts a `FeatureSet` object. The second signature takes
the input arguments of the `FeatureSet` constructor---a sample label vector, a
feature name vector and a feature value matrix---and creates the object itself.
The third signature uses the *de facto* standard data science API, expecting an
`X` feature matrix and a `y` label vector. All three methods return a
`FeatureSet` with the screened features.

The optional keyword parameters (`options`) of all three methods are:

- `reduced_size::Integer`: Expected number of screened features (right now, this
  is an upper bound).
- `step_size::Integer`: Size of each partition.
- `config::NamedTuple`: Parameters of the random forest used for importance
  computation in each round.
- `shuffle::Bool`: Whether to shuffle the features before partitioning.
- `before::Function`: Callback function, which is executed before importance
  computation and feature selection. It is called with the previously selected
  features and the actual partition, its return value is ignored.
- `after::Function`: Callback function, which is executed after importance
  computation and feature selection. It is called with the selected features,
  its return value is ignored.
- `rng::Union{AbstractRNG, Integer}`: Random generator or seed to be used.

Note that this function can run for a long time, elapsing time is proportional
to the number of features and samples, and depends on the forest configuration
as well.

```julia
# Returns a FeatureSet with the screened feature set
julia> selected_features = screen(X, # features
                                  y; # labels
                                  reduced_size = 200,
                                  step_size    = 2000)

# Retrieve the values with getters (see below)
julia> names(selected_features)
julia> features(selected_features)
```

### `FeatureSet`

A `FeatureSet` object stores sample labels, feature names, and feature values
for each label/name combination. It also stores the date of creation and a
unique ID.

There are two constructor methods:

1. `FeatureSet(labels::AbstractVector{L},
   names::AbstractVector{N},
   features::AbstractMatrix{F})`
2. `FeatureSet(X::AbstractMatrix{F},
   y::AbstractVector{L})`

The first signature is the native API, expecting the sample labels and feature
names in vectors, and feature values in a matrix. The second signature uses the
*de facto* standard data science API, expecting an `X` feature matrix and a `y`
sample label vector. In this case, feature names are automatically assigned
integers from 1 going up. Both methods accept `id` and a `created_at` optional
keyword parameter to override the defaults.

```julia
# Stores feature names in a `names` field
julia> fs = FeatureSet([1, 2],       # labels
                       ["f1", "f2"], # names
                       [1 2;
                        3 4])        # features

# The `names` field contains the indices of the features.
julia> fs_without_feature_names = FeatureSet([1 2;
                                              3 4], # X
                                             [1, 2] # y)
```

Getters can be used to retrieve values from a `FeatureSet` object:

```julia
julia> id(fs) # return the unique ID of the feature set
julia> labels(fs) # return the label vector of the feature set
julia> names(fs) # return the name vector of the feature set
julia> features(fs) # return the feature matrix of the feature set
```

### HDF5 persistence

A `FeatureSet` can be written to and loaded from a [HDF5][] file:

```julia
julia> save("feature_sets/saved.hdf5", fs)
julia> save(fs; directory = "feature_sets") # file name generate from ID
julia> fs = load(FeatureSet, "feature_sets/$(fs_id).hdf5")
```

The `load` function accepts an optional `mmap` keyword argument. If that is set
to `true`, the feature matrix is memory mapped instead of fully loaded into the
memory, which can be useful (and significantly faster) for large feature sets.

The following HDF5 datasets are written to (and expected to be readable from)
the file:

* `created_at` (datetime formatted string): timestamp of the time of creation;
* `id` (string): a UUID of the feature set;
* `labels` (vector of *L* items): sample class labels;
* `names` (vector of *N* items): feature names;
* `features` (matrix of *L* rows and *N* columns): feature values.

### CLI interface

The repository contains a Julia script called `screen.jl`, which provides a
simple CLI interface for screening. Input is read from a HDF5 file, output is
written to another, all other parameters are provided as command line switches,
with sensible defaults. The parameters should be self-explanatory.

```
$ ./screen.jl --help
Usage:
  screen.jl [options] INPUT [OUTPUT]
  screen.jl --help | --version

Perform screening on a feature set. Both the input and the output files are HDF5
feature set files.

Arguments:
  INPUT   HDF5 file to read features from
  OUTPUT  HDF5 file to write screened features to [default: "screened-" + INPUT]

Options:
      --help                   show this screen
      --version                show version
      --random-seed=SEED       use random seed for deterministic output
      --random-features=N      number of random features (implies shuffle) [default: 0]
  -r, --reduced-size=SIZE      number of target features [default: 200]
      --shuffle                shuffle features before screening
  -s, --step-size=SIZE         number of features to add per step [default: 2000]
  -v, --verbosity=LEVEL        set level of verbosity [default: 0]
      --n-subfeatures=N        [default: -1]
      --n-trees=N              [default: 1000]
      --partial-sampling=N     [default: 0.9]
      --max-depth=N            [default: -1]
      --min-samples-leaf=N     [default: 10]
      --min-samples-split=N    [default: 10]
      --min-purity-increase=N  [default: 0.0]
```

Setting verbosity of 1 makes the screener dump the current rankings after every
iteration into CSV files in a subdirectory named similarly to the output file
(without the extension). Setting verbosity to 2 makes the script save the
intermediate states into HDF5 files, too (under the same directory).

## Notes

Currently, importance computation is using random forests. Future versions may
add other ranking methods as well.

[Dumper.jl]: https://github.com/cursorinsight/Dumper.jl
[HDF5]: https://www.hdfgroup.org/solutions/hdf5/
