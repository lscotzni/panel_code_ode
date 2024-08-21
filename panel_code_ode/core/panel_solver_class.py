import numpy as np 
import csdl_alpha as csdl
import ozone_alpha as ozone

# from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver

from panel_code_ode.core.unsteady_panel_solver import unsteady_panel_solver
from panel_code_ode.core.source_doublet.post_processor import post_processor

class PanelSolver(object):
    def __init__(self, mode='source-doublet', BC='Dirichlet', unsteady=False, unstructured=False, dt=None, free_wake=False, vc=1.e-6) -> None:
        self.mode = mode
        self.BC = BC        
        self.unsteady = unsteady
        self.mesh_mode = 'structured'
        if unstructured:
            self.mesh_mode = 'unstructured'
        self.surface_index = 0.

        self.dt = dt
        self.free_wake = free_wake
        self.vc = vc

        self.surface_names = []
        self.points = []

        if unstructured:
            self.points = None
            self.cell_adjacency = None
            self.point_velocity = None
            self.TE_edges = None
        else:
            self.mesh = []
            self.mesh_velocity = []

        if unsteady is None and dt is None:
            raise TypeError('Must specify a time increment for unsteady solvers.')
        
        self._retrieve_output_names()
        
    def _retrieve_output_names(self):
        self.output_names = {
            'CL or L': 'per surface',
            'CDi or Di': 'per surface',
            'CP': '(nc, ns)',
            'panel forces': '(nc, ns)'

        }


    def add_structured_surface(self, mesh, mesh_velocity, name=None) -> None:
        if name is None:
            name = f'surface_{self.surface_index}'
            self.surface_names.append(name)
        self.surface_index += 1
        self.mesh.append(mesh)
        self.mesh_velocity.append(mesh_velocity)

    def define_unstructured_surfaces(self, surfaces, TE_edges=None):
        self.surfaces = surfaces # pointer to elements of surfaces to get forces, etc.
        self.TE_edges = TE_edges # pointer to the TE edges to assign wake propagation & Kutta condition
        
    def set_outputs(self) -> None:
        pass

    def evaluate(self, dt, free_wake=False):
        # OPTION FOR 4 SOLVERS HERE:
        # steady or unsteady (prescribed/free)
        # structured or unstructured

        if self.unsteady:
            outputs = unsteady_panel_solver(
                (self.points, self.cell_adjacency),
                self.point_velocity,
                self.TE_edges,
                dt=dt,
                mesh_mode=self.mesh_mode,
                mode=self.mode,
                free_wake=free_wake,

            )
        else:
            raise NotImplementedError('Steady panel method has not been implemented yet.')

        return outputs
    
    def panel_code_ode_function(self, ozone_vars:ozone.FuncVars):
        x_w = ozone_vars.states['x_w']
        mu_w = ozone_vars.states['mu_w']

        mesh_list = []
        mesh_vel_list = []
        for i in range(len(self.mesh)):
            mesh_list.append(ozone_vars.dynamic_parameters[f'surface_{i}_mesh'])
            mesh_vel_list.append(ozone_vars.dynamic_parameters[f'surface_{i}_vel'])

        outputs, d_dt = unsteady_panel_solver(
            mesh_list,
            mesh_vel_list,
            ode_states=[x_w.reshape(x_w.shape[1:]), mu_w.reshape(mu_w.shape[1:])],
            nt=self.nt,
            dt=self.dt,
            mesh_mode=self.mesh_mode,
            mode=self.mode,
            free_wake=self.free_wake,
            vc=self.vc
        )
        
        dxw_dt, dmuw_dt = d_dt[0], d_dt[1]
        ozone_vars.d_states['x_w'] = dxw_dt
        ozone_vars.d_states['mu_w'] = dmuw_dt

        mu = outputs['mu']
        Cp_static = outputs['Cp_static']
        panel_area = outputs['panel_area']
        panel_normal = outputs['panel_normal']
        nodal_cp_velocity = outputs['nodal_cp_velocity']
        coll_pt_velocity = outputs['coll_pt_velocity']
        planform_area = outputs['planform_area']

        ozone_vars.profile_outputs['mu'] = mu
        ozone_vars.profile_outputs['Cp_static'] = Cp_static
        ozone_vars.profile_outputs['panel_area'] = panel_area
        ozone_vars.profile_outputs['panel_normal'] = panel_normal
        ozone_vars.profile_outputs['nodal_cp_velocity'] = nodal_cp_velocity
        # ozone_vars.profile_outputs['coll_pt_velocity'] = coll_pt_velocity
        ozone_vars.profile_outputs['planform_area'] = planform_area
        
        # outputs for post-processor:
        # mesh, normal vectors, coll point velocity, nodal cp velocity, panel_area
        # planform area

# def ode_function(
#         ozone_vars:ozone.FuncVars,
#         asdf,
#     ):
#     print(asdf)
#     a = ozone_vars.dynamic_parameters['a'] # a(t)
#     b = ozone_vars.dynamic_parameters['b'] # b(t)
#     g = ozone_vars.dynamic_parameters['g'] # g(t)
#     x = ozone_vars.states['x'] # x
#     y = ozone_vars.states['y'] # y
#     d = di # di is not a function of time, so it doesn't need to be passed as a dynamic parameter
#     dx_dt = (g*x*y - d*x)
#     dy_dt = (a*y - b*y*x)

#     ozone_vars.d_states['y'] = dy_dt # dy_dt
#     ozone_vars.d_states['x'] = dx_dt # dx_dt

#     ozone_vars.field_outputs['x'] = x # any outputs you want to record summed across time
#     ozone_vars.profile_outputs['x'] = x  # any outputs you want to record across time
#     ozone_vars.profile_outputs[asdf] = x

    
    def evaluate_ode(self, x_w_0, mu_w_0, nt, dt, store_state_history=True):
        self.nt = nt
        self.dt = dt

        step_vector = np.ones(nt-1)*dt

        approach = ozone.approaches.TimeMarching()
        ode_problem = ozone.ODEProblem('ForwardEuler', approach)

        ode_problem.add_state('x_w', x_w_0, store_history=store_state_history)
        ode_problem.add_state('mu_w', mu_w_0, store_history=store_state_history)

        for i in range(len(self.mesh)):
            ode_problem.add_dynamic_parameter(f'surface_{i}_mesh', self.mesh[i])
            ode_problem.add_dynamic_parameter(f'surface_{i}_vel', self.mesh_velocity[i])

        ode_problem.set_timespan(ozone.timespans.StepVector(start=0., step_vector=step_vector))
        ode_problem.set_function(
            self.panel_code_ode_function,
            # self=self,
        )

        ode_outputs = ode_problem.solve()

        
        mu = ode_outputs.profile_outputs['mu']
        Cp_static = ode_outputs.profile_outputs['Cp_static']
        panel_normal = ode_outputs.profile_outputs['panel_normal']
        nodal_cp_velocity = ode_outputs.profile_outputs['nodal_cp_velocity']
        # coll_point_velocity = ode_outputs.profile_outputs['coll_point_velocity']
        panel_area = ode_outputs.profile_outputs['panel_area']
        planform_area = ode_outputs.profile_outputs['planform_area']

        num_nodes, nc, ns = self.mesh[0].shape[1:4]
        num_panels = (nc-1)*(ns-1)
        
        surf_mesh_dict = {
            'panel_normal': panel_normal,
            'nodal_cp_velocity': nodal_cp_velocity,
            # 'coll_point_velocity': coll_point_velocity,
            'panel_area': panel_area,
            'planform_area': planform_area,
            'num_panels': num_panels,
            'nc': nc,
            'ns': ns
        }

        mesh_dict = {'surface_0': surf_mesh_dict}

        pp_output_dict = post_processor(Cp_static, mesh_dict, mu, num_nodes, nt, dt)

        outputs = {}
        outputs['mu_wake'] = ode_outputs.states['mu_w']
        outputs['wake_mesh'] = ode_outputs.states['x_w']

        outputs['mu'] = mu
        outputs['Cp'] = pp_output_dict['surface_0']['Cp']
        outputs['CL'] = pp_output_dict['surface_0']['CL']

        # outputs = unsteady_panel_solver(
        #     (self.points, self.cell_adjacency),
        #     self.point_velocity,
        #     self.TE_edges,
        #     dt=dt,
        #     mesh_mode=self.mesh_mode,
        #     mode=self.mode,
        #     free_wake=self.free_wake,
        # )
        return outputs