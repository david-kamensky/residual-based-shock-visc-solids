"""
This script can be used to study the behavior of the proposed DC scheme in
the setting of the 1D wave equation.  
"""

from tIGAr.BSplines import *
from tIGAr.timeIntegration import *

####### Parameters #######

# Number of elements:
NEL = 512

# Polynomial degrees in each direction:
p = 2

# Parameters determining the position and size of the domain:
Lx = 6.0
x0 = -1.5

# Whether or not to use Galerkin's method:
GALERKIN = False

# Shock capturing parameters:
C_DC = Constant(0.25)
C_max = Constant(1.0)

# Directory to store ParaView output in:
RESULT_DIR = "results"

# Number of time steps to skip between ParaView files:
OUT_SKIP = 5

# Time $T$ at which to end the computation:
TIME_INTERVAL = 2.0

# Number of time steps:
N_STEPS = NEL


####### Spline setup #######

print("Generating extraction.  (This may take some time.)")

splineMesh = ExplicitBSplineControlMesh([p,],
                                        [uniformKnots(p,x0,x0+Lx,NEL),])
field = BSpline([p,],[uniformKnots(p,x0,x0+Lx,NEL,periodic=True),])
splineGenerator = FieldListSpline(splineMesh,[field,])


####### Analysis #######

def meshSizeMeasures(spline):
    """
    Extract mesh size measures from the ``tIGAr`` ``ExtractedSpline`` object 
    ``spline``.  This returns the physical element metric tensor ``G``, 
    scalar physical element size squared, ``h_K2``, and parametric 
    counterparts ``GHat`` and ``h_Q2``, in that order.
    """
    # Measure of element size in physical domain:
    dxi_dxiHat = 0.5*ufl.Jacobian(spline.mesh)
    dx_dxi = spline.parametricGrad(spline.F)
    dx_dxiHat = dx_dxi*dxi_dxiHat
    dxiHat_dx = inv(dx_dxiHat)
    G = dxiHat_dx.T*dxiHat_dx
    h_K2 = tr(inv(G))

    # Element size in parametric domain:
    dxiHat_dxi = inv(dxi_dxiHat)
    GHat = dxiHat_dxi.T*dxiHat_dxi
    h_Q2= tr(inv(GHat))
    
    return G, h_K2, GHat, h_Q2

# Choose the quadrature degree to be used throughout the analysis.
QUAD_DEG = 2*p

# Create an extracted spline:
spline = ExtractedSpline(splineGenerator,QUAD_DEG)
d = spline.mesh.geometry().dim()

# Displacement solution:
u = Function(spline.V)
u_old = Function(spline.V)
udot_old = Function(spline.V)
uddot_old = Function(spline.V)

# Re-name for output:
u_old.rename("u","u")
udot_old.rename("v","v")

# Create a generalized-alpha time integrator for the unknown field.
RHO_INF = Constant(1.0)
DELTA_T = Constant(TIME_INTERVAL/float(N_STEPS))
timeInt = GeneralizedAlphaIntegrator(RHO_INF,DELTA_T,u,
                                     (u_old, udot_old, uddot_old))

# Get $\alpha$-level quantities to form residual.
u_alpha = timeInt.x_alpha()
udot_alpha = timeInt.xdot_alpha()
uddot_alpha = timeInt.xddot_alpha()

# Material parameters:
c = Constant(1.0)

# Variables of "rate form" of wave equation:
v = udot_alpha
v_t = uddot_alpha
sigma = (c**2)*spline.grad(u_alpha)

# Test function:
w = TestFunction(spline.V)

# Information about the local mesh size in UFL:
G, h_K2, GHat, h_Q2 = meshSizeMeasures(spline)

# Strong momentum residual:
resStrong_v = v_t - spline.div(sigma)

# Weak Galerkin residual:
res_Galerkin = inner(v_t,w)*spline.dx + inner(sigma,spline.grad(w))*spline.dx

# Shock-capturing viscosities:
def norm2(v,eps=0.0):
    return sqrt(inner(v,v)+eps**2)
nu = C_DC*sqrt(h_K2)*abs(resStrong_v)/norm2(spline.grad(v),
                                            eps=Constant(DOLFIN_EPS))

nu = ufl.Min(nu,C_max*c*sqrt(h_K2))

# Shock-capturing, defined as a Python function to re-use UFL when checking
# the dissipation rate:
def res_DCf(w):
    # Reasonable choices of $\mathcal{D}$ should all collapse onto the
    # gradient in 1D.
    return nu*inner(spline.grad(v),spline.grad(w))

# Plugging in the test function to obtain the shock-capturing residual:
res_DC = res_DCf(w)*spline.dx

# Full residual:
res = res_Galerkin
if(not GALERKIN):
    res += res_DC

# Consistent linearization:
Dres = derivative(res,u)

# Use a PETSc nonlinear solver for line searching:
problem = ExtractedNonlinearProblem(spline,res,Dres,u)
solver = PETScSNESSolver()
solver.parameters["linear_solver"] = "mumps"
solver.parameters["line_search"] = "bt"
solver.parameters["error_on_nonconvergence"] = True
solver.parameters["relative_tolerance"] = 1e-8
extSolver = ExtractedNonlinearSolver(problem,solver)

# Exact solution:
def exact(t):
    x = spline.parametricCoordinates() - t*as_vector([c,]+(d-1)*[0.0,])
    r = abs(x[0])

    # Periodic:
    return sin(2.0*pi*(x[0]-x0)/Lx) \
        + conditional(lt(r,1.0),1.0-r,Constant(0.0))

# Project initial condition; mass lumping captures the shocks in a
# monotone way.
LUMP = True
u_old.assign(spline.project(exact(0.0),rationalize=False,lumpMass=LUMP))
tau = variable(Constant(0.0))
exactdot = diff(exact(tau),tau)
udot_old.assign(spline.project(exactdot,rationalize=False,lumpMass=LUMP))
exactddot = diff(exactdot,tau)
uddot_old.assign(spline.project(exactddot,rationalize=False,lumpMass=LUMP))

# Time stepping loop:
uFile = File(RESULT_DIR+"/u.pvd")
vFile = File(RESULT_DIR+"/v.pvd")
sFile = File(RESULT_DIR+"/s.pvd")
dFile = File(RESULT_DIR+"/d.pvd")
for i in range(0,N_STEPS):

    print("------ Time step "+str(i)+" -------")
  
    u.assign(timeInt.sameVelocityPredictor())
    
    extSolver.solve()

    if(i%OUT_SKIP == 0):
        uFile << u_old
        vFile << udot_old
        
        # Note that this is the space in which sigma actually lives,
        # since it's the derivative of a $C^1$-quadratic explicit B-spline.
        # The projection is exact and oscillations in the result are NOT
        # artifacts of projecting to CG1.  
        s = project(sigma[0],FunctionSpace(spline.mesh,"CG",1))
        s.rename("s","s")
        sFile << s

        # Must output this between solving and updating u_old for an accurate
        # value of the time derivative in the strong residual.
        dissipation = spline.project(res_DCf(v),
                                     rationalize=False,lumpMass=LUMP)
        dissipation.rename("d","d")
        dFile << dissipation

    timeInt.advance()

# Check the error at time $T$.  (Note: The "old" value has been updated.)
tau = variable(Constant(TIME_INTERVAL))
exactdot = diff(exact(tau),tau)
err_L2 = math.sqrt(assemble(abs(udot_old-exactdot)**2*spline.dx))
print("log(h) = "+str(math.log(Lx/NEL)))
print("log(L2 velocity error) = "+str(math.log(err_L2)))

