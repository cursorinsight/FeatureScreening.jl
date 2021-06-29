###-----------------------------------------------------------------------------
### Copyright (C) 2021- Cursor Insight
###
### All rights reserved.
###-----------------------------------------------------------------------------

module Callbacks

###=============================================================================
### Imports
###=============================================================================

using GenServers: Link
using GenServers: self, cast
using Actors: spawn, send
import Base: convert

###=============================================================================
### Types
###=============================================================================

###-----------------------------------------------------------------------------
### Status
###-----------------------------------------------------------------------------

abstract type Status end
struct Idle <: Status end
struct Running <: Status end

###-----------------------------------------------------------------------------
### State
###-----------------------------------------------------------------------------

mutable struct State
    status::Status
    value
end

###-----------------------------------------------------------------------------
### Message
###-----------------------------------------------------------------------------

abstract type Message end

struct GetValue <: Message end

### Action ---------------------------------------------------------------------

abstract type Action <: Message end

function execute(action::Action, args...)
    throw("Please implement 'execute' for this action: $(typeof(action))")
end

struct F <: Action
    f::Function
end

function convert(::Type{Action}, f::Function)::F
    return F(f)
end

function execute(f::F, args...)
    return f.f(args...)
end

struct Return <: Message
    value
end

###=============================================================================
### Callbacks
###=============================================================================

function init(value)::State
    return State(Idle(), value)
end

function oncall(state::State, args...)
    return oncall(state.status, state, args...)
end

function oncast(state::State, args...)
    return oncast(state.status, state, args...)
end

function terminate end

###-----------------------------------------------------------------------------
### On call
###-----------------------------------------------------------------------------

function oncall(::Idle, state::State, action::Action)
    return state.value = execute(action, state.value)
end

function oncall(::Idle, state::State, ::GetValue)
    return state.value
end

function oncall(::Running, state::State, ::GetValue)
    @warn "There is still a running task! The return value is the previous one!"
    return state.value
end

###-----------------------------------------------------------------------------
### On cast
###-----------------------------------------------------------------------------

function oncast(::Idle, state::State, action::Action)::Nothing
    caller::Link = self()
    worker::Link = spawn(XXX(caller))
    send(worker, action, state.value)
    state.status = Running()
    return nothing
end

function oncast(::Running, state::State, ret::Return)::Nothing
    state.value = ret.value
    state.status = Idle()
    return nothing
end

function XXX(caller)::Function
    return function (action, value)
        return cast(caller, Return(execute(action, value)))
    end
end

end # module
