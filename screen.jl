#!/bin/env -S julia --project
###-----------------------------------------------------------------------------
### Copyright (C) 2023- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

using FeatureScreening: DEFAULT_SCREEN_CONFIG

config_args = [(k, "--" * replace(String(k), '_' => '-'), v)
               for (k, v) in pairs(DEFAULT_SCREEN_CONFIG)]

const program = basename(@__FILE__)
const version = begin using Pkg; Pkg.project().version end
const usage = """
Usage:
  $(program) [options] INPUT [OUTPUT]
  $(program) --help | --version

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
""" * join([rpad("      $n=N", 31) * "[default: $v]\n"
            for (_, n, v) in config_args])

###=============================================================================
### Imports
###=============================================================================

using DocOpt: docopt
using Dumper: enable!
using FeatureScreening: FeatureSet, labels, features, names
using FeatureScreening: load, save, screen, skip
using Random: GLOBAL_RNG, MersenneTwister

###=============================================================================
### Implementation
###=============================================================================

args = docopt(usage; version)
input = args["INPUT"]
output = something(args["OUTPUT"], "screened-$(input)")
reduced_size = parse(Int, args["--reduced-size"])
random_features = parse(Int, args["--random-features"])
shuffle = random_features != 0 || args["--shuffle"]
seed = args["--random-seed"] === nothing ? nothing :
    parse(Int, args["--random-seed"])
step_size = parse(Int, args["--step-size"])
verbosity = parse(Int, args["--verbosity"])
config = NamedTuple(k => parse(typeof(v), args[n]) for (k, n, v) in config_args)
rng = seed === nothing ? GLOBAL_RNG : MersenneTwister(seed)

fs = load(FeatureSet, input; mmap = true)

if random_features != 0
    @info "Adding $(random_features) random features..."
    ns = names(fs)
    if eltype(ns) <: Integer
        ns = string.(ns; pad = floor(Int, log10(maximum(ns))) + 1)
    elseif !(eltype(ns) <: AbstractString)
        ns = string.(ns)
    end
    random_fs = rand(rng, length(fs.labels), random_features)
    fs = FeatureSet(labels(fs),
                    [ns; ["rnd_$i" for i in 1:random_features]],
                    [features(fs) random_fs])
end

dump_dir = splitext(output)[1]
if verbosity >= 1
    enable!(dump_dir)
end

after = if verbosity >= 2
    let pass = 0
        function(fs)
            save("$(dump_dir)/pass-$(string(pass += 1; pad = 2)).h5", fs)
            return nothing
        end
    end
else
    skip
end

screened_fs = screen(fs; reduced_size, step_size, config, after, shuffle, rng)

save(output, screened_fs)
