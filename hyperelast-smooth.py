"""
This script can be used to show the convergence of the shock capturing scheme
to smooth solutions, in the setting of 2D hyperelasticity.
"""

from tIGAr.BSplines import *
from tIGAr.timeIntegration import *
import ufl

# Use TSFC to handle projection of complicated forms to quadrature points.
parameters["form_compiler"]["representation"] = "tsfc"
import sys
sys.setrecursionlimit(10000)

####### Parameters #######

# Number of elements along each side of the domain:
NEL = 16

# Polynomial degree in each direction:
p = 2

# Degree of time integration, attained by using a second-order
# time integrator and scaling time step like (mesh size)^(TIME_DEG/2).
TIME_DEG = p

# Whether or not to use Galerkin's method (i.e., deactivate DC):
GALERKIN = False

# Whether or not to check the error in the stress gradient.
# (This overrides GALERKIN, so that Galerkin's method is always used
# when testing the rate accumulation of stress gradients.)
CHECK_GRAD_SIGMA = False

if(CHECK_GRAD_SIGMA):
    GALERKIN = True

# DC parameters:
C_max = Constant(1.0)
C_DC = Constant(0.25)

####### Spline setup #######

# Parameters determining the position and size of the domain:
x0 = 0.0
y0 = 0.0
Lx = 1.0
Ly = 1.0

# When checking the stress gradient error, one must use the
# fallback option to extract to simplices (useRect=False), due to a bug in
# the implementation of Quadrature-type elements for quad/hex elements.
# However, the triangle elements will have a different $G$ tensor, and are
# there for not desirable for testing shock capturing.
useRect = (not CHECK_GRAD_SIGMA)

# Create a control mesh for which $\Omega = \widehat{\Omega}$.  
splineMesh = ExplicitBSplineControlMesh([p,p],
                                        [uniformKnots(p,x0,x0+Lx,NEL),
                                         uniformKnots(p,y0,y0+Ly,NEL)],
                                        useRect=useRect)

# Create a spline generator for a spline with scalar fields for each
# displacement component on the given control mesh.
splineGenerator = EqualOrderSpline(2,splineMesh)

# Set homogeneous Dirichlet boundary conditions on all fields and all
# sides of the patch.
for field in [0,1]:
    scalarSpline = splineGenerator.getScalarSpline(field)
    for parametricDirection in [0,1]:
        for side in [0,1]:
            sideDofs = scalarSpline.getSideDofs(parametricDirection,side)
            splineGenerator.addZeroDofs(field,sideDofs)

####### Analysis #######

# Choose the quadrature degree to be used throughout the analysis.
QUAD_DEG = 2*p

# Create an extracted spline:
spline = ExtractedSpline(splineGenerator,QUAD_DEG)
d = spline.mesh.geometry().dim()

# Functions to be used in the time integration scheme:
u = Function(spline.V)
u_old = Function(spline.V)
udot_old = Function(spline.V)
uddot_old = Function(spline.V)

u_old.rename("u","u")

# Set up time integration:
RHO_INF = Constant(1.0)
N_STEPS = math.ceil(NEL**(TIME_DEG/2.0))//2
TIME_INTERVAL = 1.0
DELTA_T = Constant(TIME_INTERVAL/float(N_STEPS))
timeInt = GeneralizedAlphaIntegrator(RHO_INF,DELTA_T,u,
                                     (u_old, udot_old, uddot_old))

u_alpha = timeInt.x_alpha()
udot_alpha = timeInt.xdot_alpha()
uddot_alpha = timeInt.xddot_alpha()

# Decide the exact solution a priori.
x = spline.spatialCoordinates()
t_const = Constant(0.0)
t = variable(t_const)
# Choose to have zero acceleration at $t=0$, for convenient
# imposition of initial conditions.
u_exact = Constant(1e-1)*(t**3)*as_vector((sin(pi*x[0]/Constant(Lx))
                                           *sin(pi*x[1]/Constant(Ly)),
                                           sin(pi*x[0]/Constant(Lx))
                                           *sin(pi*x[1]/Constant(Ly))))

# Constitutive parameters for a St. Venant--Kirchhoff model:
rho = Constant(1.0)
mu = Constant(1.0)
K = Constant(1.0)
lam = K - 2.0*mu/3.0
I = Identity(d)

# Re-usable definitions for nonlinear elasticity:
i,j,k,l = ufl.indices(4)
Ctens = as_tensor(lam*I[i,j]*I[k,l]
                  + mu*(I[i,k]*I[j,l] + I[i,l]*I[j,k]),
                  (i,j,k,l))
def F(u):
    return spline.grad(u) + I
def S(u):
    E = 0.5*(F(u).T*F(u))
    i,j,k,l = ufl.indices(4)
    return as_tensor(Ctens[i,j,k,l]*E[k,l],(i,j))
def P(u):
    return F(u)*S(u)
def J(u):
    return det(F(u))
def c(u):
    Fu = F(u)
    A,B,C,D = ufl.indices(4)
    i,j,k,l = ufl.indices(4)
    return (1.0/J(u))*as_tensor(Ctens[A,B,C,D]
                                *Fu[i,A]*Fu[j,B]*Fu[k,C]*Fu[l,D],
                                (i,j,k,l))
def sigma(u):
    return (1.0/J(u))*F(u)*S(u)*(F(u).T)

# Current configuration gradient of f, for displacement field u:
def gradx(u,f):
    # Note that spline.F refers to the mapping from the IGA parametric to the
    # physical domain (as is common in mathematical literature on IGA).  It
    # is not a deformation gradient.  The argument F to spline.grad permits
    # this to be optionally replaced with a different mapping into physical
    # space, which, in this case, is the current configuration defined by
    # displacement u.
    return spline.grad(f,F=spline.F+u)

# What the Truesdell rate of Cauchy stress should be equal to:
def H(u,udot):
    i,j,k,l = ufl.indices(4)
    return as_tensor(c(u)[i,j,k,l]*sym(gradx(u,udot))[k,l],(i,j))

# See the section from the paper on the Truesdell rate for a derivation
# of this formula.
def Kf(u,v,sigma,gradSigma):
    gv = gradx(u,v)
    ggv = gradx(u,gv)
    a,b,c,d = indices(4)
    return as_tensor(-(-sigma[d,b]*ggv[a,d,c] - sigma[a,d]*ggv[b,d,c]
                       -gradSigma[d,b,c]*gv[a,d]-gradSigma[a,d,c]*gv[b,d]
                       +gradSigma[a,b,d]*gv[d,c])
                     -(ggv[d,d,c]*sigma[a,b] + gv[d,d]*gradSigma[a,b,c]),
                     (a,b,c))

if(CHECK_GRAD_SIGMA):
    # Space to store stress gradient at quadrature points
    QE = VectorElement("Quadrature",spline.mesh.ufl_cell(),
                       degree=QUAD_DEG, quad_scheme="default",
                       dim=d*d*d)
    VQ = FunctionSpace(spline.mesh,QE)

    # Convert from and to a flattened vector representation of the
    # stress gradient.
    def toTensor(v):
        tl = []
        for i in range(0,d):
            tl += [[],]
            for j in range(0,d):
                tl[i] += [[],]
                for k in range(0,d):
                    tl[i][j] += [v[i*d*d + j*d + k],]
        return as_tensor(tl)
    def toVector(t):
        vl = []
        for i in range(0,d):
            for j in range(0,d):
                for k in range(0,d):
                    vl += [t[i,j,k],]
        return as_vector(vl)

    # Project a stress gradient to the quadrature space VQ.
    # Convention: The input is a rank-3 tensor, and the return value
    # is a flattened vector.
    def quadProject(gradSigma):
        return project(toVector(gradSigma),VQ,
                       form_compiler_parameters
                       ={"quadrature_degree":QUAD_DEG})
    
# Manufacture a source term:
def resLHS_strong(uddot,u):
    return rho*uddot - spline.div(P(u))
udot_exact = diff(u_exact,t)
uddot_exact = diff(udot_exact,t)
f = resLHS_strong(uddot_exact,u_exact)

# Galerkin weak residual:
w = TestFunction(spline.V)
res_weak = (inner(rho*uddot_alpha,w)
            + inner(P(u_alpha),spline.grad(w))
            - inner(f,w))*spline.dx
# DC term:
def norm2(u,eps=0.0):
    return sqrt(inner(u,u)+eps**2)
c_s = sqrt((K+4.0*mu/3.0)/rho)
dxi_dxiHat = 0.5*ufl.Jacobian(spline.mesh)
dx_dxi = spline.parametricGrad(spline.F+u_alpha)
dx_dxiHat = dx_dxi*dxi_dxiHat
dxiHat_dx = inv(dx_dxiHat)
G = dxiHat_dx.T*dxiHat_dx
h_K2 = tr(inv(G))
h = sqrt(h_K2)
nu_DC_res = h*C_DC*norm2(resLHS_strong(uddot_alpha,u_alpha)-f)\
            /norm2(gradx(u_alpha,udot_alpha),eps=Constant(DOLFIN_EPS))
nu_DC_max = 0.5*rho/J(u_alpha)*C_max*c_s*h
nu_DC = ufl.Min(nu_DC_res,nu_DC_max)
def D_op(u,udot):
    return sym(gradx(u,udot))
res_DC = nu_DC*inner(D_op(u_alpha,udot_alpha),D_op(u_alpha,w))\
         *J(u_alpha)*spline.dx

# Residual of the variational formulation:
res = res_weak
if(not GALERKIN):
    res += res_DC

# Linearization and definition of a nonlinear problem:
tangent = derivative(res,u)
problem = ExtractedNonlinearProblem(spline,res,tangent,u)

# Use a PETSc nonlinear solver for access to robust line search techniques,
# to contend the the highly-nonlinear DC terms:
solver = PETScSNESSolver()
solver.parameters["linear_solver"] = "mumps"
solver.parameters["line_search"] = "bt"
solver.parameters["relative_tolerance"] = 1e-6
extSolver = ExtractedNonlinearSolver(problem,solver)

if(CHECK_GRAD_SIGMA):
    # History variables for updating the stress gradient approximation:
    gradSigma = Function(VQ)
    gradSigma_old = Function(VQ)

# Time-stepping loop:
uFile = File("results/u.pvd")
for i in range(0,N_STEPS):

    print("------ Time step "+str(i+1)+"/"+str(N_STEPS)+" -------")
    t_const.assign((i+timeInt.ALPHA_F)*float(DELTA_T))

    uFile << u_old
    u.assign(timeInt.sameVelocityPredictor())
    extSolver.solve()

    if(CHECK_GRAD_SIGMA):
        # Update the stress gradient using the half-step rotation algorithm.
        gradSigma_tilde = (toTensor(gradSigma_old) + DELTA_T*timeInt.ALPHA_F
                           *Kf(u_alpha,udot_alpha,sigma(u_old),
                               toTensor(gradSigma_old)))
        gradSigma_alpha = (gradSigma_tilde
                           + DELTA_T*gradx(u_alpha,H(u_alpha,udot_alpha)))
        gradSigma.assign(quadProject(gradSigma_alpha
                                     + DELTA_T*(1.0-timeInt.ALPHA_F)
                                     *Kf(u_alpha,udot_alpha,sigma(u_alpha),
                                         gradSigma_alpha)))
        gradSigma_old.assign(gradSigma)

    # Move to the next time step.
    timeInt.advance()

# Compute and print errors:
t_const.assign(TIME_INTERVAL)
import math
def L2norm(u):
    return math.sqrt(assemble(inner(u,u)*spline.dx))
def H1norm(u):
    return L2norm(spline.grad(u))

err_u_H1 = H1norm(u-u_exact)
print("log(h) = "+str(math.log(Lx/NEL)))
print("log(H1 displacement error) = "+str(math.log(err_u_H1)))

err_u_L2 = L2norm(u-u_exact)
print("log(L2 displacement error) = "+str(math.log(err_u_L2)))

if(CHECK_GRAD_SIGMA):
    err_gradSigma = L2norm(toTensor(gradSigma)-gradx(u_exact,sigma(u_exact)))
    err_gradSigma_direct = L2norm(gradx(u,sigma(u))
                                  -gradx(u_exact,sigma(u_exact)))
    print("log(L^2 error in grad(sigma)^h at time T) = "
          +str(math.log(err_gradSigma)))
    print("log(L^2 error in grad(sigma^h) at time T) = "
          +str(math.log(err_gradSigma_direct)))
