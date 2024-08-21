import numpy as np 
import csdl_alpha as csdl 

from panel_code_ode.core.source_doublet.free_wake_comp import free_wake_comp

def wake_propagation(mu, sigma, mu_wake, mesh_dict, wake_mesh_dict, num_nodes, free_wake=False):
    surface_names = list(mesh_dict.keys())
    num_surfaces = len(surface_names)

    num_wake_nodes = 0
    num_wake_panels = 0
    num_surf_panels = 0
    for surface in surface_names:
        nc_w, ns_w = wake_mesh_dict[surface]['mesh'].shape[-3], wake_mesh_dict[surface]['mesh'].shape[-2]

        num_wake_nodes += nc_w*ns_w
        num_wake_panels += wake_mesh_dict[surface]['num_panels']
        num_surf_panels += mesh_dict[surface]['num_panels']
    
    if free_wake:
        induced_vel = free_wake_comp(num_nodes, mesh_dict, mu, sigma, wake_mesh_dict, mu_wake)
        # induced_vel is of shape (nn, num_wake_points, 3) -> VECTORIZED
        # we need to access it in grid-format

    vel_wake = csdl.Variable(value=np.zeros((num_nodes, num_wake_nodes, 3)))

    start_i_s, stop_i_s = 0, 0
    start_i_w, stop_i_w = 0, 0
    for i in range(num_surfaces):
        surface_name = surface_names[i]
        nc_s, ns_s =  mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
        num_panels_s = mesh_dict[surface_name]['num_panels']
        stop_i_s += num_panels_s

        nc_w, ns_w =  wake_mesh_dict[surface_name]['nc'], wake_mesh_dict[surface_name]['ns']
        num_panels_w = wake_mesh_dict[surface_name]['num_panels']
        num_wake_pts = nc_w*ns_w
        stop_i_w += num_wake_pts

        mesh_velocity = mesh_dict[surface_name]['nodal_velocity']
        TE_velocity = (mesh_velocity[:,0,:,:]+mesh_velocity[:,-1,:,:])/2.

        wake_velocity = csdl.expand(TE_velocity, (num_nodes, nc_w, ns_w, 3), 'ijk->iajk')

        if free_wake:
            total_vel = wake_velocity.reshape((num_nodes, num_wake_pts, 3)) + induced_vel[:,start_i_w:stop_i_w,:]
        else:
            total_vel = wake_velocity.reshape((num_nodes, num_wake_pts, 3))

        vel_wake = vel_wake.set(csdl.slice[:,start_i_w:stop_i_w,:], value=total_vel)
        # NOTE: we vectorize the velocities because surfaces have different discretizations
        # it makes the most sense to store velocities fully vectorized
        # we can unwrap and re-grid the velocities at the beginning of the solver before any computations are done

        start_i_w += num_wake_pts

    return vel_wake
