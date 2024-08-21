import numpy as np
import csdl_alpha as csdl

from panel_code_ode.core.source_doublet.pre_processor import pre_processor
from panel_code_ode.core.source_doublet.wake_geometry import wake_geometry
from panel_code_ode.core.source_doublet.mu_sigma_solver import mu_sigma_solver
from panel_code_ode.core.source_doublet.compute_static_pressure import compute_static_pressure
from panel_code_ode.core.source_doublet.wake_propagation import wake_propagation


def source_doublet_solver(exp_orig_mesh_dict, ode_states, num_nodes, nt, dt, mesh_mode, free_wake):
    '''
    steps:
        - take initial condition and input from Ozone for derivatives
            - un-wrap and re-grid wake velocity/position/mu, etc.
        - surface pre-processing (DONE)
        - wake pre-processing (DONE)
        - solve for singularities (DONE)
        - wake propagation
        - derivative computation
    '''
    x_w = ode_states[0] # wake mesh position vectorized across num surfaces, nc_w, ns_w
    mu_w = ode_states[1] # wake doublets vectorized across num surfaces, nc_w, ns_w

    # number of wake nodes per surface: ns * (nt+1), where one row is coincident with the TE
    # number of wake mu per surface: (ns-1) * (nt+1-1)

    surface_names = list(exp_orig_mesh_dict.keys())
    num_surfaces = len(surface_names)
    wake_mesh_dict = {}
    x_start, x_stop = 0, 0
    mu_start, mu_stop = 0, 0

    num_tot_wake_panels = 0
    num_tot_wake_pts = 0
    for i in range(num_surfaces):
        surface_name = surface_names[i]
        ns = exp_orig_mesh_dict[surface_name]['mesh'].shape[2]

        num_surf_wake_pts = ns*nt
        num_surf_wake_panels = (ns-1)*(nt-1)

        x_stop += num_surf_wake_pts
        mu_stop += num_surf_wake_panels

        sub_dict = {}
        sub_dict['mesh'] = x_w[:,x_start:x_stop,:].reshape((num_nodes, nt, ns, 3))
        sub_dict['mu'] = mu_w[:,mu_start:mu_stop].reshape((num_nodes, nt-1, ns-1))

        wake_mesh_dict[surface_name] = sub_dict

        x_start += num_surf_wake_pts
        mu_start += num_surf_wake_panels

        num_tot_wake_pts += num_surf_wake_pts
        num_tot_wake_panels += num_surf_wake_panels


    print('running pre-processing')
    mesh_dict = pre_processor(exp_orig_mesh_dict, mode=mesh_mode)
    wake_mesh_dict = wake_geometry(wake_mesh_dict)

    print('solving for doublet strengths and propagating the wake')
    mu_wake = mu_w
    mu, sigma, wake_mesh_dict = mu_sigma_solver(num_nodes, nt, mesh_dict, wake_mesh_dict, mu_wake, mesh_mode=mesh_mode, free_wake=free_wake)

    pp_outputs = compute_static_pressure(mesh_dict, mu, sigma, num_nodes)

    vel_wake = wake_propagation(mu, sigma, mu_wake, mesh_dict, wake_mesh_dict, num_nodes, free_wake=free_wake)

    # compute derivatives here 
    dmuw_dt = csdl.Variable((1, num_nodes, num_tot_wake_panels), value=0.)
    dxw_dt = csdl.Variable((1, num_nodes, num_tot_wake_pts, 3), value=0.)

    x_start, x_stop = 0, 0
    mu_start, mu_stop = 0, 0
    mu_wake_start, mu_wake_stop = 0, 0
    for i in range(num_surfaces):
        surface_name = surface_names[i]
        nc, ns = mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
        nc_w = wake_mesh_dict[surface_name]['nc'] # this is equal to nt+1

        # region wake doublet strength time derivatives
        num_panels = (ns-1)*(nc-1)
        num_wake_panels = (ns-1)*(nc_w-1)

        mu_stop += num_panels
        mu_wake_stop += num_wake_panels

        mu_surf = mu[:,mu_start:mu_stop].reshape((num_nodes, nc-1, ns-1))
        delta_mu_TE = mu_surf[:,-1,:] - mu_surf[:,0,:]
        mu_wake_surf = mu_wake[:,mu_wake_start:mu_wake_stop].reshape((num_nodes, nc_w-1, ns-1))

        dmuw_dt_surf = csdl.Variable(value=np.zeros((num_nodes, nc_w-1, ns-1)))
        dmuw_dt_surf = dmuw_dt_surf.set(csdl.slice[:,0,:], value=(delta_mu_TE - mu_wake_surf[:,0,:])/dt)
        dmuw_dt_surf = dmuw_dt_surf.set(csdl.slice[:,1:,:], value=(mu_wake_surf[:,:-1,:] - mu_wake_surf[:,1:,:])/dt)

        dmuw_dt = dmuw_dt.set(csdl.slice[0,:,mu_wake_start:mu_wake_stop], value=dmuw_dt_surf.reshape((num_nodes, (ns-1)*(nc_w-1))))

        mu_start += num_panels
        mu_wake_start += num_wake_panels
        # endregion

        # region wake position time derivatives
        num_wake_pts = ns*nc_w
        x_stop += num_wake_pts
        mesh = mesh_dict[surface_name]['mesh']
        TE = (mesh[:,0,:,:] + mesh[:,-1,:,:])/2.
        wake_mesh = wake_mesh_dict[surface_name]['mesh']

        wake_total_vel = vel_wake[:,x_start:x_stop,:].reshape((num_nodes, nc_w, ns, 3))

        dxw_dt_surf = csdl.Variable(shape=(num_nodes, nc_w, ns, 3), value=0.)
        # dxw_dt_surf = dxw_dt_surf.set(csdl.slice[:,0,:,:], value=(TE - wake_mesh[:,0,:,:] + wake_total_vel[:,0,:,:]*dt)/dt)
        dxw_dt_surf = dxw_dt_surf.set(csdl.slice[:,1,:,:], value=(TE - wake_mesh[:,0,:,:] + wake_total_vel[:,1,:,:]*dt)/dt)
        dxw_dt_surf = dxw_dt_surf.set(csdl.slice[:,1:,:,:], value=(wake_mesh[:,:-1,:,:] - wake_mesh[:,1:,:,:] + wake_total_vel[:,1:,:,:]*dt)/dt)

        dxw_dt = dxw_dt.set(csdl.slice[0,:,x_start:x_stop,:], value=dxw_dt_surf.reshape((num_nodes, num_wake_pts, 3)))

        x_start += num_wake_pts

    d_dt = [dxw_dt, dmuw_dt]
    # outputs for post-processor:
    # mesh, normal vectors, coll point velocity, nodal cp velocity, panel_area
    # planform area
    Cp_static = pp_outputs['surface_0']['Cp_static']
    panel_area = mesh_dict['surface_0']['panel_area']
    panel_normal = mesh_dict['surface_0']['panel_normal']
    nodal_cp_velocity = mesh_dict['surface_0']['nodal_cp_velocity']
    planform_area = mesh_dict['surface_0']['planform_area']
    outputs = {
        'mu': mu.reshape((1,) + mu.shape),
        'Cp_static': Cp_static.reshape((1,) + Cp_static.shape),
        'panel_area': panel_area.reshape((1,) + panel_area.shape),
        'panel_normal': panel_normal.reshape((1,) + panel_normal.shape),
        'nodal_cp_velocity': nodal_cp_velocity.reshape((1,) + nodal_cp_velocity.shape),
        'coll_pt_velocity': mesh_dict['surface_0']['coll_point_velocity'],
        'planform_area': planform_area.reshape((1,) + planform_area.shape)
        # NOTE: WILL NEED TO ADD OTHER STUFF SINCE POST-PROCESSOR IS IN THE OUTER LOOP
    }

    return outputs, d_dt