import numpy as np 
import csdl_alpha as csdl

from panel_code_ode.core.geometry.gen_panel_mesh import gen_panel_mesh
from panel_code_ode.core.panel_solver_class import PanelSolver

from panel_code_ode.utils.plot import plot_wireframe, plot_pressure_distribution

b = 10. # wingspan
c = 1. # wing chord
ns = 11 # spanwise nodes
nc = 11 # one-way chord-wise nodes (TE to LE)
nt = 40 # number of time steps
dt = 0.05 # time step size 
num_nodes = 1

alpha_deg = 10. # angle of attack
alpha = np.deg2rad(alpha_deg) # aoa

# flow conditions (doesn't really matter here tbh)
mach = 0.15
sos = 340.3
V_inf = np.array([-sos*mach, 0., 0.])
# V_inf = np.array([-10., 0., 0.])
V_inf = np.array([-10., 0., 0.])

# generating mesh of size (2*nc-1, ns, 3)
mesh_orig = gen_panel_mesh(nc, ns, c, b, span_spacing='cosine',  frame='default', plot_mesh=False)

# expanding mesh to include (nt, num_nodes)
mesh = np.zeros((nt, num_nodes) + mesh_orig.shape)
for i in range(nt):
    for j in range(num_nodes):
        mesh[i,j,:] = mesh_orig

# adjusting flow for aoa
V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

# expanded mesh velocity
mesh_velocity = np.zeros_like(mesh)
for i in range(nt):
    for j in range(num_nodes):
        mesh_velocity[i,j,:] = V_inf_rot

recorder = csdl.Recorder(inline=False)
recorder.start()

mesh = csdl.Variable(value=mesh)
mesh_velocity = csdl.Variable(value=mesh_velocity)

# setting initial conditions
TE = (mesh[0,:,0,:,:] + mesh[0,:,-1,:,:])/2.
x_w_0 = csdl.einsum(TE, np.ones((nt,)), action='ijk,l->iljk', ).reshape((num_nodes, (ns*(nt)), 3))
mu_w_0 = csdl.Variable(value=np.zeros((num_nodes, (nt-1)*(ns-1))))

# instantiating solver class
panel_solver = PanelSolver(
    unsteady=True,
    free_wake=False, # FREE WAKE FLAG
    vc=1.e-1 # vortex core size if free_wake=True
)

# adding mesh to solver
panel_solver.add_structured_surface(
    mesh=mesh,
    mesh_velocity=mesh_velocity,
    name='wing'
)

# evaluating ODE + post-processing
outputs = panel_solver.evaluate_ode(
    x_w_0=x_w_0, # IC for wake position
    mu_w_0=mu_w_0, # IC for wake doublets
    nt=nt, 
    dt=dt
)

# toggle jax on or off
jax = True
if jax:
    Cp = outputs['Cp']
    wake_mesh = outputs['wake_mesh']
    mu = outputs['mu']
    mu_wake = outputs['mu_wake']
    CL = outputs['CL']

    recorder.stop()
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[mesh], # list of outputs (put in csdl variable)
        additional_outputs=[mu, mu_wake, wake_mesh, Cp, CL], # list of outputs (put in csdl variable)
    )
    jax_sim.run()

    mesh = jax_sim[mesh]
    Cp = jax_sim[Cp]
    wake_mesh = jax_sim[wake_mesh].reshape((nt, num_nodes, nt, ns, 3))
    mu = jax_sim[mu].reshape((nt, num_nodes, 2*(nc-1), ns-1))
    mu_wake = jax_sim[mu_wake].reshape((nt, num_nodes, nt-1, ns-1))
    CL = jax_sim[CL]

else:
    Cp = outputs['Cp'].value
    wake_mesh = outputs['wake_mesh'].value.reshape((nt, num_nodes, nt, ns, 3))
    mu = outputs['mu'].value.reshape((nt, num_nodes, 2*(nc-1), ns-1))
    mu_wake = outputs['mu_wake'].value.reshape((nt, num_nodes, nt-1, ns-1))
    mesh = mesh.value
    CL = outputs['CL'].value

# output plots; can toggle on or off
# plot pressure distribution
if False:
    plot_pressure_distribution(mesh, Cp, interactive=True, top_view=False)

# generate video of wake propagation
if False:
    # plot_wireframe(mesh, wake_mesh, mu.value, mu_wake.value, nt, interactive=False, backend='cv', name=f'wing_fw_{alpha_deg}')
    plot_wireframe(mesh, wake_mesh, mu, mu_wake, nt, interactive=False, backend='cv', name='free_wake_demo')

