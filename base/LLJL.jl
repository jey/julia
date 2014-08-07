module LLJL
  include("lljl/ir.jl")

  const base_typeinf_ext = Base.typeinf_ext

  function lljl_typeinf_ext(linfo, atypes::ANY, sparams::ANY, def)
    ccall(:printf, Void, (Ptr{Uint8},), "typeinf: $(linfo.name)\n")
    base_typeinf_ext(linfo, atypes, sparams, def)
  end

  function boot()
    try
      jl_typeinf_func = cglobal(:jl_typeinf_func, Any)
      @assert unsafe_load(jl_typeinf_func) === base_typeinf_ext
      unsafe_store!(jl_typeinf_func, lljl_typeinf_ext)
    catch err
      err.msg == "cglobal: could not find symbol jl_typeinf_func" || error("expected julia v0.3.x")
      warn("LLJL.boot(): typeinf hook not available, booting via monkey patch")
      binding = ccall(:jl_get_binding, Ptr{Any}, (Any, Any), Base, :typeinf_ext)
      # jl_binding_t has layout Cstruct(name::Symbol, value::Any, type::Any, owner::Module, flags)
      @assert unsafe_load(binding, 1) === :typeinf_ext
      @assert unsafe_load(binding, 2) === base_typeinf_ext
      unsafe_store!(binding, lljl_typeinf_ext, 2)
      @assert Base.typeinf_ext === lljl_typeinf_ext
      ccall(:jl_enable_inference, Void, ())
      unsafe_store!(binding, base_typeinf_ext, 2)
      @assert Base.typeinf_ext === base_typeinf_ext
    end
  end
end
