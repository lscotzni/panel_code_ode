import numpy as np 
import csdl_alpha as csdl

from panel_code_ode.core.source_doublet.perturbation_velocity_comp import least_squares_velocity 

def compute_static_pressure(mesh_dict, mu, sigma, num_nodes):
    surface_names = list(mesh_dict.keys())
    start, stop = 0, 0

    output_dict = {}
    for i in range(len(surface_names)):
        surf_dict = {}
        surface_name = surface_names[i]
        num_panels = mesh_dict[surface_name]['num_panels']
        nc, ns = mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
        stop += num_panels

        mu_grid = mu[:,start:stop].reshape((num_nodes, nc-1, ns-1))

        # perturbation velocities
        qn = -sigma[:,start:stop].reshape((num_nodes, nc-1, ns-1)) # num_nodes, nt, num_panels for surface

        # region least squares method for perturbation velocities (derivatives)
        delta_coll_point = mesh_dict[surface_name]['delta_coll_point']
        ql, qm = least_squares_velocity(mu_grid, delta_coll_point)
        # endregion

        panel_x_dir = mesh_dict[surface_name]['panel_x_dir']
        panel_y_dir = mesh_dict[surface_name]['panel_y_dir']
        panel_normal = mesh_dict[surface_name]['panel_normal']
        nodal_cp_velocity = mesh_dict[surface_name]['nodal_cp_velocity']
        coll_vel = mesh_dict[surface_name]['coll_point_velocity']
        if coll_vel:
            total_vel = nodal_cp_velocity+coll_vel
        else:
            total_vel = nodal_cp_velocity

        free_stream_l = csdl.einsum(total_vel, panel_x_dir, action='jklm,jklm->jkl')
        free_stream_m = csdl.einsum(total_vel, panel_y_dir, action='jklm,jklm->jkl')
        free_stream_n = csdl.einsum(total_vel, panel_normal, action='jklm,jklm->jkl')

        Ql = free_stream_l + ql
        Qm = free_stream_m + qm
        Qn = free_stream_n + qn
        Q_inf_norm = csdl.norm(total_vel, axes=(3,))

        Cp_static = 1 - (Ql**2 + Qm**2 + Qn**2)/Q_inf_norm**2
        # END HERE

        surf_dict['Cp_static'] = Cp_static
        output_dict[surface_name] = surf_dict

        start += num_panels

    return output_dict