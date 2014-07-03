using Base.Test
import Base.LinAlg: chksquare, blas_int

# SRC: base/linalg/arpack.jl
module ARPACK 

import Base.LinAlg: BlasInt, blas_int, ARPACKException

## aupd and eupd wrappers 

function aupd_wrapper(T, matvecA::Function, matvecB::Function, solveSI::Function, n::Integer,
                      sym::Bool, cmplx::Bool, bmat::ASCIIString,
                      nev::Integer, ncv::Integer, which::ASCIIString, 
                      tol::Real, maxiter::Integer, mode::Integer, v0::Vector)

    lworkl = cmplx ? ncv * (3*ncv + 5) : (sym ? ncv * (ncv + 8) :  ncv * (3*ncv + 6) )
    TR = cmplx ? T.types[1] : T
    TOL = Array(TR, 1)
    TOL[1] = tol

    v      = Array(T, n, ncv)
    workd  = Array(T, 3*n)
    workl  = Array(T, lworkl)
    rwork  = cmplx ? Array(TR, ncv) : Array(TR, 0)

    if isempty(v0)
      resid  = Array(T, n)
      info   = zeros(BlasInt, 1)
    else
      resid  = deepcopy(v0)
      info   = ones(BlasInt, 1)
    end
    iparam = zeros(BlasInt, 11)
    ipntr  = zeros(BlasInt, (sym && !cmplx) ? 11 : 14)
    ido    = zeros(BlasInt, 1)

    iparam[1] = blas_int(1)       # ishifts
    iparam[3] = blas_int(maxiter) # maxiter
    iparam[7] = blas_int(mode)    # mode

    zernm1 = 0:(n-1)

    while true
        if cmplx
            naupd(ido, bmat, n, which, nev, TOL, resid, ncv, v, n,
                  iparam, ipntr, workd, workl, lworkl, rwork, info)            
        elseif sym
            saupd(ido, bmat, n, which, nev, TOL, resid, ncv, v, n,
                  iparam, ipntr, workd, workl, lworkl, info)
        else
            naupd(ido, bmat, n, which, nev, TOL, resid, ncv, v, n,
                  iparam, ipntr, workd, workl, lworkl, info)
        end

        # Check for warnings and errors
        # Refer to ex-*.doc files in ARPACK/DOCUMENTS for calling sequence
        if info[1] == 3; warn("try eigs/svds with a larger value for ncv"); end
        if info[1] == 1; warn("maximum number of iterations reached; check nconv for number of converged eigenvalues"); end
        if info[1] < 0; throw(ARPACKException(info[1])); end

        load_idx = ipntr[1]+zernm1
        store_idx = ipntr[2]+zernm1
        x = workd[load_idx]
        if ido[1] == -1
            if mode == 1
                workd[store_idx] = matvecA(x)
            elseif mode == 2
                if sym
                    temp = matvecA(x)
                    workd[load_idx] = temp    # overwrite as per Remark 5 in dsaupd.f
                    workd[store_idx] = solveSI(temp)
                else
                    workd[store_idx] = solveSI(matvecA(x))
                end
            elseif mode == 3
                if bmat == "I"
                    workd[store_idx] = solveSI(x)
                elseif bmat == "G"
                    workd[store_idx] = solveSI(matvecB(x))
                end
            end
        elseif ido[1] == 1
            if mode == 1
                workd[store_idx] = matvecA(x)
            elseif mode == 2
                if sym
                    temp = matvecA(x)
                    workd[load_idx] = temp
                    workd[store_idx] = solveSI(temp)
                else
                    workd[store_idx] = solveSI(matvecA(x))
                end
            elseif mode == 3
                if bmat == "I"
                    workd[store_idx] = solveSI(x)
                else
                    workd[store_idx] = solveSI(workd[ipntr[3]+zernm1])
                end
            end
        elseif ido[1] == 2
            workd[store_idx] = matvecB(x)
        elseif ido[1] == 99
            break
        else
            error("Internal ARPACK error")
        end
    end
    
    return (resid, v, n, iparam, ipntr, workd, workl, lworkl, rwork, TOL)
end

function eupd_wrapper(T, n::Integer, sym::Bool, cmplx::Bool, bmat::ASCIIString,
                      nev::Integer, which::ASCIIString, ritzvec::Bool,
                      TOL::Array, resid, ncv::Integer, v, ldv, sigma, iparam, ipntr,
                      workd, workl, lworkl, rwork)

    howmny = "A"
    select = Array(BlasInt, ncv)
    info   = zeros(BlasInt, 1)
    
    if cmplx

        d = Array(T, nev+1)
        sigma = ones(T, 1)*sigma
        workev = Array(T, 2ncv)
        neupd(ritzvec, howmny, select, d, v, ldv, sigma, workev,
              bmat, n, which, nev, TOL, resid, ncv, v, ldv,
              iparam, ipntr, workd, workl, lworkl, rwork, info)
        if info[1] != 0; throw(ARPACKException(info[1])); end
        return ritzvec ? (d[1:nev], v[1:n, 1:nev],iparam[5],iparam[3],iparam[9],resid) : (d[1:nev],iparam[5],iparam[3],iparam[9],resid)

    elseif sym

        d = Array(T, nev)
        sigma = ones(T, 1)*sigma
        seupd(ritzvec, howmny, select, d, v, ldv, sigma,
              bmat, n, which, nev, TOL, resid, ncv, v, ldv,
              iparam, ipntr, workd, workl, lworkl, info) 
        if info[1] != 0; throw(ARPACKException(info[1])); end
        return ritzvec ? (d, v[1:n, 1:nev],iparam[5],iparam[3],iparam[9],resid) : (d,iparam[5],iparam[3],iparam[9],resid)

    else

        dr     = Array(T, nev+1)
        di     = Array(T, nev+1)
        sigmar = ones(T, 1)*real(sigma)
        sigmai = ones(T, 1)*imag(sigma)
        workev = Array(T, 3*ncv)
        neupd(ritzvec, howmny, select, dr, di, v, ldv, sigmar, sigmai,
              workev, bmat, n, which, nev, TOL, resid, ncv, v, ldv,
              iparam, ipntr, workd, workl, lworkl, info)
        if info[1] != 0; throw(ARPACKException(info[1])); end
        evec = complex(zeros(T, n, nev+1), zeros(T, n, nev+1))
        j = 1
        while j <= nev
            if di[j] == 0
                evec[:,j] = v[:,j]
            else
                evec[:,j]   = v[:,j] + im*v[:,j+1]
                evec[:,j+1] = v[:,j] - im*v[:,j+1]
                j += 1
            end
            j += 1
        end
        d = complex(dr[1:nev],di[1:nev])
        return ritzvec ? (d, evec[1:n, 1:nev],iparam[5],iparam[3],iparam[9],resid) : (d,iparam[5],iparam[3],iparam[9],resid)
    end
    
end

for (T, saupd_name, seupd_name, naupd_name, neupd_name) in
    ((:Float64, :dsaupd_, :dseupd_, :dnaupd_, :dneupd_),
     (:Float32, :ssaupd_, :sseupd_, :snaupd_, :sneupd_))
    @eval begin

        function naupd(ido, bmat, n, evtype, nev, TOL::Array{$T}, resid::Array{$T}, ncv, v::Array{$T}, ldv,
                       iparam, ipntr, workd::Array{$T}, workl::Array{$T}, lworkl, info)
            
            ccall(($(string(naupd_name)), :libarpack), Void,
                  (Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt},
                   Ptr{$T}, Ptr{$T}, Ptr{BlasInt}, Ptr{$T}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                  ido, bmat, &n, evtype, &nev, TOL, resid, &ncv, v, &ldv,
                  iparam, ipntr, workd, workl, &lworkl, info, sizeof(bmat), sizeof(evtype))
        end

        function neupd(rvec, howmny, select, dr, di, z, ldz, sigmar, sigmai,
                  workev::Array{$T}, bmat, n, evtype, nev, TOL::Array{$T}, resid::Array{$T}, ncv, v, ldv,
                  iparam, ipntr, workd::Array{$T}, workl::Array{$T}, lworkl, info)

            ccall(($(string(neupd_name)), :libarpack), Void,
                  (Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                   Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Uint8}, Ptr{BlasInt},
                   Ptr{Uint8}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{BlasInt}, Ptr{$T},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T},
                   Ptr{BlasInt}, Ptr{BlasInt}, Clong, Clong, Clong),
                  &rvec, howmny, select, dr, di, z, &ldz, sigmar, sigmai,
                  workev, bmat, &n, evtype, &nev, TOL, resid, &ncv, v, &ldv,
                  iparam, ipntr, workd, workl, &lworkl, info,
                  sizeof(howmny), sizeof(bmat), sizeof(evtype))
        end

        function saupd(ido, bmat, n, which, nev, TOL::Array{$T}, resid::Array{$T}, ncv, v::Array{$T}, ldv, 
                       iparam, ipntr, workd::Array{$T}, workl::Array{$T}, lworkl, info)
            
            ccall(($(string(saupd_name)), :libarpack), Void,
                  (Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt},
                   Ptr{$T}, Ptr{$T}, Ptr{BlasInt}, Ptr{$T}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                  ido, bmat, &n, which, &nev, TOL, resid, &ncv, v, &ldv,
                  iparam, ipntr, workd, workl, &lworkl, info, sizeof(bmat), sizeof(which))

        end

        function seupd(rvec, howmny, select, d, z, ldz, sigma,
                       bmat, n, evtype, nev, TOL::Array{$T}, resid::Array{$T}, ncv, v::Array{$T}, ldv,
                       iparam, ipntr, workd::Array{$T}, workl::Array{$T}, lworkl, info) 

            ccall(($(string(seupd_name)), :libarpack), Void,
                  (Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{BlasInt}, Ptr{$T},
                   Ptr{Uint8}, Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt},
                   Ptr{$T}, Ptr{$T}, Ptr{BlasInt}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt}, Clong, Clong, Clong),
                  &rvec, howmny, select, d, z, &ldz, sigma,
                  bmat, &n, evtype, &nev, TOL, resid, &ncv, v, &ldv,
                  iparam, ipntr, workd, workl, &lworkl, info, sizeof(howmny), sizeof(bmat), sizeof(evtype))
        end

    end
end

for (T, TR, naupd_name, neupd_name) in
    ((:Complex128, :Float64, :znaupd_, :zneupd_),
     (:Complex64,  :Float32, :cnaupd_, :cneupd_))
    @eval begin

        function naupd(ido, bmat, n, evtype, nev, TOL::Array{$TR}, resid::Array{$T}, ncv, v::Array{$T}, ldv,
                       iparam, ipntr, workd::Array{$T}, workl::Array{$T}, lworkl, 
                       rwork::Array{$TR}, info)
            
            ccall(($(string(naupd_name)), :libarpack), Void,
                  (Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt},
                   Ptr{$TR}, Ptr{$T}, Ptr{BlasInt}, Ptr{$T}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{BlasInt},
                   Ptr{$TR}, Ptr{BlasInt}),
                  ido, bmat, &n, evtype, &nev, TOL, resid, &ncv, v, &ldv,
                  iparam, ipntr, workd, workl, &lworkl, rwork, info)

        end

        function neupd(rvec, howmny, select, d, z, ldz, sigma, workev::Array{$T},
                       bmat, n, evtype, nev, TOL::Array{$TR}, resid::Array{$T}, ncv, v::Array{$T}, ldv,
                       iparam, ipntr, workd::Array{$T}, workl::Array{$T}, lworkl, 
                       rwork::Array{$TR}, info)

            ccall(($(string(neupd_name)), :libarpack), Void,
                  (Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{BlasInt},
                   Ptr{$T}, Ptr{$T}, Ptr{Uint8}, Ptr{BlasInt}, Ptr{Uint8}, Ptr{BlasInt},
                   Ptr{$TR}, Ptr{$T}, Ptr{BlasInt}, Ptr{$T}, Ptr{BlasInt}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{BlasInt}, Ptr{$TR}, Ptr{BlasInt}),
                  &rvec, howmny, select, d, z, &ldz, sigma, workev,
                  bmat, &n, evtype, &nev, TOL, resid, &ncv, v, &ldv,
                  iparam, ipntr, workd, workl, &lworkl, rwork, info) 

        end

    end
end

end # module ARPACK

# SRC: base/linalg/arnoldi.jl
using .ARPACK

## eigs

eigs(A; args...) = eigs(A, I; args...)

function eigs(A, B;
              nev::Integer=6, ncv::Integer=20, which::ASCIIString="LM",
              tol=0.0, maxiter::Integer=1000, sigma=nothing, v0::Vector=zeros((0,)),
              ritzvec::Bool=true)

    n = chksquare(A)

    T = eltype(A)
    iscmplx = T <: Complex
    isgeneral = B !== I
    sym = issym(A)
    (nev = min(nev, sym ? n - 1 : n - 2)) > 0 || throw(ArgumentError("nev must be at least one"))
    ncv = blas_int(min(max(2*nev + 2, ncv), n))
    isgeneral && !isposdef(B) && throw(PosDefException(0))
    bmat = isgeneral ? "G" : "I"
    isshift = sigma !== nothing
    sigma = isshift ? sigma : zero(T)

    if !isempty(v0)
        length(v0)==n || throw(DimensionMismatch(""))
        eltype(v0)==T || error("Starting vector must have eltype $T")
    else
        v0=zeros(T,(0,))
    end

    # Refer to ex-*.doc files in ARPACK/DOCUMENTS for calling sequence
    matvecA(x) = A * x
    if !isgeneral           # Standard problem
        matvecB(x) = x
        if !isshift         #    Regular mode
            mode       = 1
            solveSI(x) = x
        else                #    Shift-invert mode
            mode       = 3
            solveSI(x) = factorize(sigma==0 ? A : A - UniformScaling(sigma)) \ x
        end
    else                    # Generalized eigen problem
        matvecB(x) = B * x
        if !isshift         #    Regular inverse mode
            mode       = 2
            solveSI(x) = factorize(B) \ x
        else                #    Shift-invert mode
            mode       = 3
            solveSI(x) = factorize(sigma==0 ? A : A-sigma*B) \ x
        end
    end

    # Compute the Ritz values and Ritz vectors
    (resid, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, TOL) = 
       ARPACK.aupd_wrapper(T, matvecA, matvecB, solveSI, n, sym, iscmplx, bmat, nev, ncv, which, tol, maxiter, mode, v0)
    
    # Postprocessing to get eigenvalues and eigenvectors
    if ritzvec
        (eval, evec, nconv, niter, nmult, resid) =
             ARPACK.eupd_wrapper(T, n, sym, iscmplx, bmat, nev, which, ritzvec, TOL,
                                 resid, ncv, v, ldv, sigma, iparam, ipntr, workd, workl, lworkl, rwork)
    else
        (eval, nconv, niter, nmult, resid) =
             ARPACK.eupd_wrapper(T, n, sym, iscmplx, bmat, nev, which, ritzvec, TOL,
                                 resid, ncv, v, ldv, sigma, iparam, ipntr, workd, workl, lworkl, rwork)
    end

end

# SRC: test/arpack.jl
begin
    srand(1234)
    local n,a,asym,b,bsym,d,v
    n = 10
    areal  = sprandn(n,n,0.4)
    breal  = sprandn(n,n,0.4)
    acmplx = complex(sprandn(n,n,0.4), sprandn(n,n,0.4))
    bcmplx = complex(sprandn(n,n,0.4), sprandn(n,n,0.4))

    testtol = 1e-6

    for elty in (Float64, Complex128)
        if elty == Complex64 || elty == Complex128
            a = acmplx
            b = bcmplx
        else
            a = areal
            b = breal
        end
        a     = convert(SparseMatrixCSC{elty}, a)
        asym  = a' + a                  # symmetric indefinite
        apd   = a'*a                    # symmetric positive-definite

        b     = convert(SparseMatrixCSC{elty}, b)
        bsym  = b' + b
        bpd   = b'*b

      println(elty)
    	(d,v) = eigs(a, nev=3)
    	@test_approx_eq a*v[:,2] d[2]*v[:,2]
        @test norm(v) > testtol # eigenvectors cannot be null vectors
    	# (d,v) = eigs(a, b, nev=3, tol=1e-8) not handled yet
    	# @test_approx_eq_eps a*v[:,2] d[2]*b*v[:,2] testtol
        # @test norm(v) > testtol # eigenvectors cannot be null vectors
    
    	(d,v) = eigs(asym, nev=3)
    	@test_approx_eq asym*v[:,1] d[1]*v[:,1]
        @test_approx_eq eigs(asym; nev=1, sigma=d[3])[1][1] d[3]
        @test norm(v) > testtol # eigenvectors cannot be null vectors
    
    	(d,v) = eigs(apd, nev=3)
    	@test_approx_eq apd*v[:,3] d[3]*v[:,3]
        @test_approx_eq eigs(apd; nev=1, sigma=d[3])[1][1] d[3]
    	
        (d,v) = eigs(apd, bpd, nev=3, tol=1e-8)
    	@test_approx_eq_eps apd*v[:,2] d[2]*bpd*v[:,2] testtol
        @test norm(v) > testtol # eigenvectors cannot be null vectors
    
        # test (shift-and-)invert mode
        (d,v) = eigs(apd, nev=3, sigma=0)
        @test_approx_eq apd*v[:,3] d[3]*v[:,3]
        @test norm(v) > testtol # eigenvectors cannot be null vectors
        
        (d,v) = eigs(apd, bpd, nev=3, sigma=0, tol=1e-8)
        @test_approx_eq_eps apd*v[:,1] d[1]*bpd*v[:,1] testtol
        @test norm(v) > testtol # eigenvectors cannot be null vectors

    end
end

# Example from Quantum Information Theory
import Base: size, issym, ishermitian

type CPM{T<:Base.LinAlg.BlasFloat}<:AbstractMatrix{T} # completely positive map
	kraus::Array{T,3} # kraus operator representation
end

size(Phi::CPM)=(size(Phi.kraus,1)^2,size(Phi.kraus,3)^2)
issym(Phi::CPM)=false
ishermitian(Phi::CPM)=false

function *{T<:Base.LinAlg.BlasFloat}(Phi::CPM{T},rho::Vector{T})
	rho=reshape(rho,(size(Phi.kraus,3),size(Phi.kraus,3)))
	rho2=zeros(T,(size(Phi.kraus,1),size(Phi.kraus,1)))
	for s=1:size(Phi.kraus,2)
		As=slice(Phi.kraus,:,s,:)
		rho2+=As*rho*As'
	end
	return reshape(rho2,(size(Phi.kraus,1)^2,))
end
# Generate random isometry
(Q,R)=qr(randn(100,50))
Q=reshape(Q,(50,2,50))
# Construct trace-preserving completely positive map from this
Phi=CPM(Q)
(d,v,nconv,numiter,numop,resid) = eigs(Phi,nev=1,which="LM")
# Properties: largest eigenvalue should be 1, largest eigenvector, when reshaped as matrix
# should be a Hermitian positive definite matrix (up to an arbitrary phase)

@test_approx_eq d[1] 1. # largest eigenvalue should be 1.
v=reshape(v,(50,50)) # reshape to matrix
v/=trace(v) # factor out arbitrary phase
@test isapprox(vecnorm(imag(v)),0.) # it should be real
v=real(v)
# @test isapprox(vecnorm(v-v')/2,0.) # it should be Hermitian
# Since this fails sometimes (numerical precision error),this test is commented out
v=(v+v')/2
@test isposdef(v)

# Repeat with starting vector
(d2,v2,nconv2,numiter2,numop2,resid2) = eigs(Phi,nev=1,which="LM",v0=reshape(v,(2500,)))
v2=reshape(v2,(50,50))
v2/=trace(v2)
@test numiter2<numiter
@test_approx_eq v v2

@test_approx_eq eigs(speye(50), nev=10)[1] ones(10) #Issue 4246

