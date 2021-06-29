###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

module Server

###=============================================================================
### Imports
###=============================================================================

using GenServers: Link
using GenServers: genserver, call, cast, exit!

include("Callbacks.jl")

# Message types
using .Callbacks: F, Action, GetValue

###-----------------------------------------------------------------------------
### API
###-----------------------------------------------------------------------------

function start(initial_state)::Link
    return genserver(Callbacks, initial_state)
end

function apply(f::Function, server::Link; kwargs...)
    return apply(server, f; kwargs...)
end

function apply(server::Link, f; async::Bool = false)
    action::Action = convert(Action, f)

    if !async
        return call(server, action)
    else
        return cast(server, action)
    end
end

function get_value(server::Link)
    return call(server, GetValue())
end

function stop(server::Link)
    return exit!(server)
end

end # module
