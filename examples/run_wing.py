import numpy as np 
import matplotlib.pyplot as plt
import csdl_alpha as csdl

from panel_code_ode.core.geometry.gen_panel_mesh import gen_panel_mesh
from panel_code_ode.core.panel_solver_class import PanelSolver

from panel_code_ode.utils.plot import plot_wireframe, plot_pressure_distribution

b = 10.
# c = 1.564
c = .8698
ns = 11
nc = 11

alpha_deg = 10.
alpha = np.deg2rad(alpha_deg) # aoa

mach = 0.15
sos = 340.3
V_inf = np.array([-sos*mach, 0., 0.])
# V_inf = np.array([-10., 0., 0.])
V_inf = np.array([-10., 0., 0.])
nt = 10
dt=0.05
num_nodes = 1

mesh_orig = gen_panel_mesh(nc, ns, c, b, span_spacing='default',  frame='default', plot_mesh=False)

mesh = np.zeros((nt, num_nodes) + mesh_orig.shape)
for i in range(nt):
    for j in range(num_nodes):
        mesh[i,j,:] = mesh_orig

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_velocity = np.zeros_like(mesh)
for i in range(nt):
    for j in range(num_nodes):
        mesh_velocity[i,j,:] = V_inf_rot

recorder = csdl.Recorder(inline=False)
recorder.start()

mesh = csdl.Variable(value=mesh)
mesh_velocity = csdl.Variable(value=mesh_velocity)

TE = (mesh[0,:,0,:,:] + mesh[0,:,-1,:,:])/2.
x_w_0 = csdl.einsum(TE, np.ones((nt,)), action='ijk,l->iljk', ).reshape((num_nodes, (ns*(nt)), 3))
mu_w_0 = csdl.Variable(value=np.zeros((num_nodes, (nt-1)*(ns-1))))

panel_solver = PanelSolver(
    unsteady=True,
    free_wake=False
)

panel_solver.add_structured_surface(
    mesh=mesh,
    mesh_velocity=mesh_velocity,
    name='wing'
)

outputs = panel_solver.evaluate_ode(
    x_w_0=x_w_0,
    mu_w_0=mu_w_0,
    nt=nt, 
    dt=dt
)
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

Cp_section = Cp[-1,0,:,int((ns-1)/2)]
coll_points_x = (mesh[-1,0,:-1,:-1,0] + mesh[-1,0,:-1,1:,0] + mesh[-1,0,1:,1:,0] + mesh[-1,0,1:,:-1,0])/4.
Cp_loc = coll_points_x[:,int((ns-1)/2)]

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

# plt.plot(LHJ_data_Re9[:,0], LHJ_data_Re9[:,1], 'vb', fillstyle='none', label='Ladson et al. ')
# plt.plot(Gregory_data[:,0], Gregory_data[:,1], '>r', label='Gregory et al. ')
plt.plot(Cp_loc/c, Cp_section, 'k*-', label='CSDL panel code')

plt.gca().invert_yaxis()
plt.xlabel('Normalized chord')
plt.ylabel('$C_p$')
plt.legend()
plt.grid()
plt.show()


if False:
    plot_pressure_distribution(mesh, Cp, interactive=True, top_view=False)

if False:
    # plot_wireframe(mesh, wake_mesh, mu.value, mu_wake.value, nt, interactive=False, backend='cv', name=f'wing_fw_{alpha_deg}')
    plot_wireframe(mesh, wake_mesh, mu, mu_wake, nt, interactive=False, backend='cv', name='free_wake_demo')

# TODO:
# - connect solutions/output data to paraview format
# - send some panel code sources to Mark