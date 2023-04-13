###-----------------------------------------------------------------------------
### Copyright (C) FeatureScreening.jl
###
### SPDX-License-Identifier: MIT License
###-----------------------------------------------------------------------------

module Fixtures

###=============================================================================
### Exports
###=============================================================================

export fixture

###=============================================================================
### Imports
###=============================================================================

using FeatureScreening.Types: FeatureSet

###=============================================================================
### Implementation
###=============================================================================

macro with_pattern(f)
    @assert f.head == :function
    f_name::Symbol = f.args[1]
    return quote
        $f
        function $f_name(tokens::Symbol...)
            try
                return $f_name(Val.(tokens)...)
            catch e
                if e isa MethodError
                    @error "Missing pattern $(e.f)($(e.args...))"
                end
                rethrow(e)
            end
        end
    end |> esc
end

macro pattern(f)
    @assert f.head == :function
    (signature, body...) = f.args
    @assert(f.args[1].head == :call,
            "Simple function signature allowed(, no return type assertion)!")
    (function_name::Symbol, tokens...) = f.args[1].args
    arguments::Vector = map(tokens) do token
        return :(::Val{$token})
    end

    return quote
        function $function_name($(arguments...))
            return $(body...)
        end
    end |> esc
end

##------------------------------------------------------------------------------

@with_pattern function fixture end

@pattern function fixture(:config, :screen)
    return (n_subfeatures       = -1,
            n_trees             = 20,
            partial_sampling    = 0.9,
            max_depth           = -1,
            min_samples_leaf    = 2,
            min_samples_split   = 3,
            min_purity_increase = 0.0)
end

@pattern function fixture(:feature_set)
    return FeatureSet(fixture(:y), fixture(:names), fixture(:X))
end

@pattern function fixture(:y)
    return [:a, :a, :a, :a, :a,
            :b, :b, :b, :b, :b,
            :c, :c, :c, :c, :c,
            :d, :d, :d, :d, :d,
            :e, :e, :e, :e, :e]
end

@pattern function fixture(:names)
    return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
end

@pattern function fixture(:X)
    X::Matrix{Float64} =
        [0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.2 -0.4 -0.6 -0.8 -1.0 -1.2 -1.4 -1.6 -1.8 -2.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.1 +0.2 +0.3 +0.4 +0.5 +0.6 +0.7 +0.8 +0.9 +1.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0;
         0.0 +0.2 +0.4 +0.6 +0.8 +1.0 +1.2 +1.4 +1.6 +1.8 +2.0]

    return X .+ randn(size(X))
end

end # module
