import numpy as np 
import csdl_alpha as csdl 

from panel_code_ode.core.source_doublet.source_doublet_solver import source_doublet_solver

# def unsteady_panel_solver(mesh_list, mesh_velocity_list, dt, mesh_mode='structured', mode='source-doublet', connectivity=None, free_wake=False):
def unsteady_panel_solver(*args, ode_states, nt, dt, mesh_mode='structured', mode='source-doublet', free_wake=False, vc=1.e-6):
    '''
    2 modes:
    - source doublet (Dirichlet (no-perturbation potential in the body))
    - vortex rings (Neumann (no-penetration condition))
    '''

    if mesh_mode == 'structured':
        connectivity=False
        mesh_list = args[0]
        mesh_velocity_list = args[1]
        try:
            coll_vel_list = args[2]
            coll_vel = True
        except:
            coll_vel = False

        exp_orig_mesh_dict = {}
        surface_counter = 0
        for i in range(len(mesh_list)):
            surface_name = f'surface_{surface_counter}'
            exp_orig_mesh_dict[surface_name] = {}
            surf_mesh = mesh_list[i]
            surf_vel = mesh_velocity_list[i]
            exp_orig_mesh_dict[surface_name]['mesh'] = surf_mesh.reshape(surf_mesh.shape[1:])
            exp_orig_mesh_dict[surface_name]['nodal_velocity'] = surf_vel.reshape(surf_vel.shape[1:]) * -1. 
            if coll_vel:
                exp_orig_mesh_dict[surface_name]['coll_point_velocity'] = coll_vel_list[i][1:] * -1. 
            else:
                exp_orig_mesh_dict[surface_name]['coll_point_velocity'] = coll_vel
            if i == 0:
                num_nodes = mesh_list[i].shape[1] # NOTE: CHECK THIS LINE
            
            surface_counter += 1

    elif mesh_mode == 'unstructured':
        points = args[0]
        cells, cell_adjacency = args[1][0], args[1][1]
        TE_node_indices, TE_cells = args[2]
        upper_TE_cells, lower_TE_cells = TE_cells[0], TE_cells[1]
        point_velocity = args[3]

        num_nodes = points.shape[0]

        exp_orig_mesh_dict = {}
        exp_orig_mesh_dict['points'] = points
        exp_orig_mesh_dict['nodal_velocity'] = point_velocity
        exp_orig_mesh_dict['cell_point_indices'] = cells
        exp_orig_mesh_dict['cell_adjacency'] = cell_adjacency

        exp_orig_mesh_dict['TE_node_indices'] = TE_node_indices
        exp_orig_mesh_dict['upper_TE_cells'] = upper_TE_cells
        exp_orig_mesh_dict['lower_TE_cells'] = lower_TE_cells


    # NOTE: CAN USE EITHER ARGS OR KWARGS FOR THIS


    if mode == 'source-doublet':
        outputs, d_dt = source_doublet_solver(exp_orig_mesh_dict, ode_states, num_nodes, nt, dt, mesh_mode, free_wake, vc=vc)
        # output_dict = outputs[0]
        # mesh_dict = outputs[1]
        # wake_mesh_dict = outputs[2]
        # mu = outputs[3]
        # sigma = outputs[4]
        # mu_wake = outputs[5]

        mu = outputs['mu']
        Cp_static = outputs['Cp_static']
        panel_area = outputs['panel_area']
        panel_normal = outputs['panel_normal']
        nodal_cp_velocity = outputs['nodal_cp_velocity']
        coll_pt_velocity = outputs['coll_pt_velocity']
        planform_area = outputs['planform_area']

        solver_outputs = {
            'mu': mu,
            'Cp_static': Cp_static,
            'panel_area': panel_area,
            'panel_normal': panel_normal,
            'nodal_cp_velocity': nodal_cp_velocity,
            'coll_pt_velocity': coll_pt_velocity,
            'planform_area': planform_area
        }

        return solver_outputs, d_dt

        # return output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake
    
    else:
        raise TypeError('Invalid solver mode. Options are source-doublet')