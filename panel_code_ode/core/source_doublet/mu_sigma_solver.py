import numpy as np
import csdl_alpha as csdl 

from panel_code_ode.core.source_doublet.wake_geometry import wake_geometry
from panel_code_ode.core.source_doublet.source_functions import compute_source_strengths, compute_source_influence, compute_source_influence_new
from panel_code_ode.core.source_doublet.doublet_functions import compute_doublet_influence, compute_doublet_influence_new



def mu_sigma_solver(num_nodes, nt, mesh_dict, wake_mesh_dict, mu_wake, mesh_mode='structured', free_wake=False):
    '''
    STEPS:
        - compute source strengths (DONE)
        - set up static AIC matrices (DONE)
        - compute wake AIC matrix
        - set up kutta condition
        - linear system solve
    '''

    if mesh_mode == 'structured':
        surface_names = list(mesh_dict.keys())
        num_tot_panels = 0
        for surface in surface_names:
            num_tot_panels += mesh_dict[surface]['num_panels']

    # computing source strengths
    sigma = compute_source_strengths(mesh_dict, num_nodes, num_tot_panels, mesh_mode='structured') # shape=(num_nodes, nt, num_surf_panels)

    # static AIC matrices
    AIC_mu, AIC_sigma = static_AIC_computation(mesh_dict, num_nodes, num_tot_panels, surface_names)

    sigma_BC_influence = csdl.einsum(AIC_sigma, sigma, action='jlk,jk->jl')
    # looping through the wake
    # we add part of this to the AIC_mu 
    num_wake_panels = 0
    num_first_wake_panels = 0
    for surface in surface_names:
        num_wake_panels += wake_mesh_dict[surface]['num_panels']
        num_first_wake_panels += wake_mesh_dict[surface]['ns'] - 1

    # wake AIC matrices
    num_surfaces = len(surface_names)
    AIC_wake = csdl.Variable(shape=(num_nodes, num_tot_panels, num_wake_panels), value=0.)
    mu = csdl.Variable(shape=(num_nodes, num_tot_panels), value=0.)
    start_mu, stop_mu = 0, 0
    start_mu_m1, stop_mu_m1 = 0, 0

    start_i, stop_i = 0, 0
    for i in range(num_surfaces): # surface where BC is applied
        surf_i_name = surface_names[i]
        coll_point_i = mesh_dict[surf_i_name]['panel_center'][:,:,:,:] # evaluation point
        nc_i, ns_i = mesh_dict[surf_i_name]['nc'], mesh_dict[surf_i_name]['ns']
        num_panels_i = mesh_dict[surf_i_name]['num_panels']
        stop_i += num_panels_i

        start_j, stop_j = 0, 0
        for j in range(num_surfaces): # looping through wakes
            # print(f'j: {j}')
            surf_j_name = surface_names[j]
            nc_j, ns_j = wake_mesh_dict[surf_j_name]['nc'], wake_mesh_dict[surf_j_name]['ns']
            num_panels_j = wake_mesh_dict[surf_j_name]['num_panels']
            stop_j += num_panels_j

            panel_corners_j = wake_mesh_dict[surf_j_name]['panel_corners'][:,:,:,:,:]
            panel_x_dir_j = wake_mesh_dict[surf_j_name]['panel_x_dir'][:,:,:,:]
            panel_y_dir_j = wake_mesh_dict[surf_j_name]['panel_y_dir'][:,:,:,:]
            panel_normal_j = wake_mesh_dict[surf_j_name]['panel_normal'][:,:,:,:]
            dpij_j = wake_mesh_dict[surf_j_name]['dpij'][:,:,:,:,:]
            dij_j = wake_mesh_dict[surf_j_name]['dij'][:,:,:,:]
            mij_j = wake_mesh_dict[surf_j_name]['mij'][:,:,:,:]

            num_interactions = num_panels_i*num_panels_j

            coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'jklm->jklabm')
            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions, 4, 3))

            panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklmn->jaklmn')
            panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, num_interactions, 4, 3))

            panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
            panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
            panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, num_interactions, 4, 3))


            dpij_j_exp = csdl.expand(dpij_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 2), 'jklmn->jaklmn')
            dpij_j_exp_vec = dpij_j_exp.reshape((num_nodes, num_interactions, 4, 2))
            dij_j_exp = csdl.expand(dij_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            dij_j_exp_vec = dij_j_exp.reshape((num_nodes, num_interactions, 4))
            mij_j_exp = csdl.expand(mij_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            mij_j_exp_vec = mij_j_exp.reshape((num_nodes, num_interactions, 4))

            dp = coll_point_i_exp_vec - panel_corners_j_exp_vec
            # NOTE: consider changing dx,dy,dz and store in just dk with a dimension for x,y,z
            sum_ind = len(dp.shape) - 1
            dx = csdl.sum(dp*panel_x_dir_j_exp_vec, axes=(sum_ind,))
            dy = csdl.sum(dp*panel_y_dir_j_exp_vec, axes=(sum_ind,))
            dz = csdl.sum(dp*panel_normal_j_exp_vec, axes=(sum_ind,))
            rk = (dx**2 + dy**2 + dz**2 + 1.e-12)**0.5
            ek = dx**2 + dz**2
            hk = dx*dy

            mij_list = [mij_j_exp_vec[:,:,ind] for ind in range(4)]
            dpij_list = [[dpij_j_exp_vec[:,:,ind,0], dpij_j_exp_vec[:,:,ind,1]] for ind in range(4)]
            ek_list = [ek[:,:,ind] for ind in range(4)]
            hk_list = [hk[:,:,ind] for ind in range(4)]
            rk_list = [rk[:,:,ind] for ind in range(4)]
            dx_list = [dx[:,:,ind] for ind in range(4)]
            dy_list = [dy[:,:,ind] for ind in range(4)]
            dz_list = [dz[:,:,ind] for ind in range(4)]

            # wake_doublet_influence_vec = compute_doublet_influence(dpij_j_exp_vec, mij_j_exp_vec, ek, hk, rk, dx, dy, dz, mode='potential')
            wake_doublet_influence_vec = compute_doublet_influence(dpij_list, mij_list, ek_list, hk_list, rk_list, dx_list, dy_list, dz_list, mode='potential')

            wake_doublet_influence = wake_doublet_influence_vec.reshape((num_nodes, num_panels_i, num_panels_j)) # GRID OF WAKE AIC (without kutta condition taken into account)

            AIC_wake = AIC_wake.set(csdl.slice[:,start_i:stop_i, start_j:stop_j], value=wake_doublet_influence)

            start_j += num_panels_j

        start_i += num_panels_i

    # solving linear system
    for nn in csdl.frange(num_nodes):
        # RHS
        wake_influence = csdl.matvec(AIC_wake[nn,:,:], mu_wake[nn,:])
        RHS = -sigma_BC_influence[nn,:] - wake_influence
        
        mu_timestep = csdl.solve_linear(AIC_mu[nn,:,:], RHS)
        mu = mu.set(csdl.slice[nn,:], value=mu_timestep)
    
    return mu, sigma, wake_mesh_dict

def static_AIC_computation(mesh_dict, num_nodes, num_tot_panels, surface_names):
    AIC_sigma = csdl.Variable(shape=(num_nodes,num_tot_panels, num_tot_panels), value=0.)
    AIC_mu = csdl.Variable(shape=(num_nodes, num_tot_panels, num_tot_panels), value=0.)

    num_surfaces = len(surface_names)
    start_i, stop_i = 0, 0
    for i in range(num_surfaces):
        surf_i_name = surface_names[i]
        coll_point_i = mesh_dict[surf_i_name]['panel_center'] # evaluation point
        nc_i, ns_i = mesh_dict[surf_i_name]['nc'], mesh_dict[surf_i_name]['ns']
        num_panels_i = mesh_dict[surf_i_name]['num_panels']
        stop_i += num_panels_i

        start_j, stop_j = 0, 0
        for j in range(num_surfaces):
            surf_j_name = surface_names[j]
            nc_j, ns_j = mesh_dict[surf_j_name]['nc'], mesh_dict[surf_j_name]['ns']
            num_panels_j = mesh_dict[surf_j_name]['num_panels']
            stop_j += num_panels_j

            panel_corners_j = mesh_dict[surf_j_name]['panel_corners']
            panel_x_dir_j = mesh_dict[surf_j_name]['panel_x_dir']
            panel_y_dir_j = mesh_dict[surf_j_name]['panel_y_dir']
            panel_normal_j = mesh_dict[surf_j_name]['panel_normal']
            dpij_j = mesh_dict[surf_j_name]['dpij']
            dij_j = mesh_dict[surf_j_name]['dij']
            mij_j = mesh_dict[surf_j_name]['mij']

            num_interactions = num_panels_i*num_panels_j

            coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'jklm->jklabm')
            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions, 4, 3))

            panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklmn->jaklmn')
            panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, num_interactions, 4, 3))

            panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
            panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, num_interactions, 4, 3))
            panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'jklm->jaklbm')
            panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, num_interactions, 4, 3))

            dpij_j_exp = csdl.expand(dpij_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4, 2), 'jklmn->jaklmn')
            dpij_j_exp_vec = dpij_j_exp.reshape((num_nodes, num_interactions, 4, 2))
            dij_j_exp = csdl.expand(dij_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            dij_j_exp_vec = dij_j_exp.reshape((num_nodes, num_interactions, 4))
            mij_j_exp = csdl.expand(mij_j, (num_nodes, num_panels_i, nc_j-1, ns_j-1, 4), 'jklm->jaklm')
            mij_j_exp_vec = mij_j_exp.reshape((num_nodes, num_interactions, 4))

            dp = coll_point_i_exp_vec - panel_corners_j_exp_vec
            # NOTE: consider changing dx,dy,dz and store in just dk with a dimension for x,y,z
            sum_ind = len(dp.shape) - 1
            dx = csdl.sum(dp*panel_x_dir_j_exp_vec, axes=(sum_ind,))
            dy = csdl.sum(dp*panel_y_dir_j_exp_vec, axes=(sum_ind,))
            dz = csdl.sum(dp*panel_normal_j_exp_vec, axes=(sum_ind,))
            rk = (dx**2 + dy**2 + dz**2 + 1.e-12)**0.5
            ek = dx**2 + dz**2
            hk = dx*dy

            mij_list = [mij_j_exp_vec[:,:,ind] for ind in range(4)]
            dij_list = [dij_j_exp_vec[:,:,ind] for ind in range(4)]
            dpij_list = [[dpij_j_exp_vec[:,:,ind,0], dpij_j_exp_vec[:,:,ind,1]] for ind in range(4)]
            ek_list = [ek[:,:,ind] for ind in range(4)]
            hk_list = [hk[:,:,ind] for ind in range(4)]
            rk_list = [rk[:,:,ind] for ind in range(4)]
            dx_list = [dx[:,:,ind] for ind in range(4)]
            dy_list = [dy[:,:,ind] for ind in range(4)]
            dz_list = [dz[:,:,ind] for ind in range(4)]

            # source_influence_vec = compute_source_influence(dij_j_exp_vec, mij_j_exp_vec, dpij_j_exp_vec, dx, dy, dz, rk, ek, hk, mode='potential')
            source_influence_vec = compute_source_influence(dij_list, mij_list, dpij_list, dx_list, dy_list, dz_list, rk_list, ek_list, hk_list, mode='potential')
            source_influence = source_influence_vec.reshape((num_nodes, num_panels_i, num_panels_j))
            AIC_sigma = AIC_sigma.set(csdl.slice[:,start_i:stop_i, start_j:stop_j], value=source_influence)

            # doublet_influence_vec = compute_doublet_influence(dpij_j_exp_vec, mij_j_exp_vec, ek, hk, rk, dx, dy, dz, mode='potential')
            doublet_influence_vec = compute_doublet_influence(dpij_list, mij_list, ek_list, hk_list, rk_list, dx_list, dy_list, dz_list, mode='potential')
            doublet_influence = doublet_influence_vec.reshape((num_nodes, num_panels_i, num_panels_j))
            AIC_mu = AIC_mu.set(csdl.slice[:,start_i:stop_i, start_j:stop_j], value=doublet_influence)

            start_j += num_panels_j
        start_i += num_panels_i
    return AIC_mu, AIC_sigma