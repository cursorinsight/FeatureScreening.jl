# FeatureScreening.jl

[![CI](https://github.com/cursorinsight/FeatureScreening.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/cursorinsight/FeatureScreening.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/cursorinsight/FeatureScreening.jl/branch/master/graph/badge.svg?token=P59CK9SA1Z)](https://codecov.io/gh/cursorinsight/FeatureScreening.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

FeatureScreening.jl is a Julia package that provides utilities for finding
significant features in a feature set, based on their importance in
classification.

For an in depth discussion of the underlying algorithm, see paper published in
[Machine Learning: Science and Technology][MLST] (an IOPscience journal) at DOI
[10.1088/2632-2153/ad020e](https://doi.org/10.1088/2632-2153/ad020e).

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

FeatureScreening.jl depends on a few other open source packages developed by
[Cursor Insight][] (notably, [Dumper.jl][]), but not registered in the
[General][] registry at the current time. To let the Julia package manager know
about these packages, you need to add our [own registry][CIJR] (in addition to
the General registry) to your Julia environment before installing
`FeatureScreening.jl` itself:

```julia
julia> ]
pkg> registry add https://github.com/cursorinsight/julia-registry
     Cloning registry from "https://github.com/cursorinsight/julia-registry"
       Added registry `CursorInsightJuliaRegistry` to
       `~/.julia/registries/CursorInsightJuliaRegistry`

pkg> add FeatureScreening
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

The first signature accepts a `FeatureSet` object (see documentation of
[FeatureSets.jl][] for details). The second signature takes the input arguments
of the `FeatureSet` constructor---a sample label vector, a feature name vector
and a feature value matrix---and creates the object itself. The third signature
uses the *de facto* standard data science API, expecting an `X` feature matrix
and a `y` label vector. All three methods return a `FeatureSet` with the
screened features.

The optional keyword parameters (`options`) of all three methods are:

- `reduced_size::Integer`: Expected number of screened features (an upper
  bound); defaults to 1/5th of the number of features. Mutually exclusive with
  `selection_mode`.
- `step_size::Integer`: Size of each partition; defaults to 1/10th of the number
  of features.
- `selection_mode::SelectionMode`: a mode to pick selected features after
  importance computation. Mutually exclusive with `reduced_size`. For various
  selection modes, see details below.
- `config::NamedTuple`: Parameters of the random forest used for importance
  computation in each round; with sensible defaults (see `DEFAULT_SCREEN_CONFIG`
  in the source code for details).
- `shuffle::Bool`: Whether to shuffle the features before partitioning.
- `before::Function`: Callback function, that is executed before importance
  computation and feature selection. It is called with the previously selected
  features and the current partition, its return value is ignored.
- `after::Function`: Callback function, that is executed after importance
  computation and feature selection. It is called with the selected features,
  its return value is ignored.
- `rng::Union{AbstractRNG, Integer}`: Random generator or seed to be used.

Specifying `reduced_size` is equivalent to passing `SelectTop(reduced_size;
strict = false)` in the `selection_mode` parameter. Both options cannot be
specified simultaneously.

Note that this function can run for a long time. Runtime is proportional to the
number of features and samples, and depends on the forest configuration as well.

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

### Selection modes

A selection mode is an object encapsulating an item selection strategy. It
controls the behavior of the `select()` function. Currently, the following
selection modes are defined:

* `SelectTop(n::Integer)`, `SelectTop(r::Real)`: take the top/first *n* items,
  or the *r* ratio (0-1) of the total number of items, from the collection.

* `SelectRandom([weights::Function,] n::Integer [; replace::Bool])`,
  `SelectRandom([weights::Function,] r::Real [; replace::Bool])`: take *n*
  random items, or the *r* ratio (0-1) of the total number of items, from the
  collection, with replacement if `replace` is true (defaults to false). The
  probability of picking a specific item is weighted according to the `weights`
  function, which receives the collection, and must return a same length vector
  of corresponding weights. By default, `unit_weights` is used, which assigns
  equal weights to all items.

* `SelectByImportance(n::Int)`: a specific `SelectRandom` take *n* random items
  from a collection of `label => importance` pairs with no replacement, such
  that the importance values serve as weights.

* `ComposedSelectionMode(a, b)`, `a ∘ b`: composes two selection modes, such
  that it performs selection with selection mode `b` first, and then applies
  selection mode `a` on the outcome of the former.

All selection modes also accept a `strict` keyword parameter (defaults to true).
If this is set to true, trying to take more items from the collection than it
contains results in an error. If `strict` is false, the specified number of
items is considered an upper bound only, but it is silently clamped between the
given boundaries.

Custom selection modes may be defined by inheriting from the abstract type
`SelectionMode`, and defining a method `select(rng::AbstractRNG,
collection::AbstractVector, selection_mode)` to perform the item selection.

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

[CIJR]: https://github.com/cursorinsight/julia-registry
[Cursor Insight]: https://www.cursorinsight.com/
[Dumper.jl]: https://github.com/cursorinsight/Dumper.jl
[FeatureSets.jl]: https://github.com/cursorinsight/FeatureSets.jl
[General]: https://github.com/JuliaRegistries/General
[MLST]: https://iopscience.iop.org/journal/2632-2153
