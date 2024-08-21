import numpy as np
import csdl_alpha as csdl

def pre_processor(mesh_dict, mode='structured'):
    surface_names = list(mesh_dict.keys())
    if mode == 'structured':
        for i, key in enumerate(surface_names):
            # mesh = mesh_dict[key]['mesh']
            # mesh_shape = mesh.shape
            # nc, ns = mesh_shape[-3], mesh_shape[-2]

            # mesh_dict[key]['num_panels'] = (nc-1)*(ns-1)
            # mesh_dict[key]['nc'] = nc
            # mesh_dict[key]['ns'] = ns

            # R1 = mesh[:,:-1,:-1,:]
            # R2 = mesh[:,1:,:-1,:]
            # R3 = mesh[:,1:,1:,:]
            # R4 = mesh[:,:-1,1:,:]
        
            # S1 = (R1+R2)/2.
            # S2 = (R2+R3)/2.
            # S3 = (R3+R4)/2.
            # S4 = (R4+R1)/2.

            # Rc = (R1+R2+R3+R4)/4.
            # mesh_dict[key]['panel_center'] = Rc

            # panel_corners = csdl.Variable(shape=(Rc.shape[:-1] + (4,3)), value=0.)
            # panel_corners = panel_corners.set(csdl.slice[:,:,:,0,:], value=R1)
            # panel_corners = panel_corners.set(csdl.slice[:,:,:,1,:], value=R2)
            # panel_corners = panel_corners.set(csdl.slice[:,:,:,2,:], value=R3)
            # panel_corners = panel_corners.set(csdl.slice[:,:,:,3,:], value=R4)
            # mesh_dict[key]['panel_corners'] = panel_corners

            # D1 = R3-R1
            # D2 = R4-R2

            # D1D2_cross = csdl.cross(D1, D2, axis=3)
            # D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(3,))
            # panel_area = D1D2_cross_norm/2.
            # mesh_dict[key]['panel_area'] = panel_area

            # normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'ijk->ijka')
            # mesh_dict[key]['panel_normal'] = normal_vec

            # panel_center_mod = Rc - normal_vec*0.001
            # mesh_dict[key]['panel_center_mod'] = panel_center_mod

            # m_dir = S3 - Rc
            # m_norm = csdl.norm(m_dir, axes=(4,))
            # m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'ijk->ijka')
            # l_vec = csdl.cross(m_vec, normal_vec, axis=3)
            # # this also tells us that normal_vec = cross(l_vec, m_vec)

            # mesh_dict[key]['panel_x_dir'] = l_vec
            # mesh_dict[key]['panel_y_dir'] = m_vec

            # SMP = csdl.norm((S2)/2 - Rc, axes=(3,))
            # SMQ = csdl.norm((S3)/2 - Rc, axes=(3,)) # same as m_norm

            # mesh_dict[key]['SMP'] = SMP
            # mesh_dict[key]['SMQ'] = SMQ

            # s = csdl.Variable(shape=panel_corners.shape, value=0.)
            # s = s.set(csdl.slice[:,:,:,:-1,:], value=panel_corners[:,:,:,1:,:] - panel_corners[:,:,:,:-1,:])
            # s = s.set(csdl.slice[:,:,:,-1,:], value=panel_corners[:,:,:,0,:] - panel_corners[:,:,:,-1,:])

            # l_exp = csdl.expand(l_vec, panel_corners.shape, 'ijkm->ijkam')
            # m_exp = csdl.expand(m_vec, panel_corners.shape, 'ijkm->ijkam')
            
            # S = csdl.norm(s+1.e-12, axes=(4,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
            # SL = csdl.sum(s*l_exp, axes=(4,))
            # SM = csdl.sum(s*m_exp, axes=(4,))

            # mesh_dict[key]['S'] = S
            # mesh_dict[key]['SL'] = SL
            # mesh_dict[key]['SM'] = SM

            # delta_coll_point = csdl.Variable(Rc.shape[:-1] + (4,2), value=0.)
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,1:,:,0,0], value=csdl.sum((Rc[:,:-1,:,:]-Rc[:,1:,:,:])*l_vec[:,1:,:,:], axes=(3,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,1:,:,0,1], value=csdl.sum((Rc[:,:-1,:,:]-Rc[:,1:,:,:])*m_vec[:,1:,:,:], axes=(3,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,:-1,:,1,0], value=csdl.sum((Rc[:,1:,:,:]-Rc[:,:-1,:,:])*l_vec[:,:-1,:,:], axes=(3,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,:-1,:,1,1], value=csdl.sum((Rc[:,1:,:,:]-Rc[:,:-1,:,:])*m_vec[:,:-1,:,:], axes=(3,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,2,0], value=csdl.sum((Rc[:,:,:-1,:]-Rc[:,:,1:,:])*l_vec[:,:,1:,:], axes=(3,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,2,1], value=csdl.sum((Rc[:,:,:-1,:]-Rc[:,:,1:,:])*m_vec[:,:,1:,:], axes=(3,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,3,0], value=csdl.sum((Rc[:,:,1:,:]-Rc[:,:,:-1,:])*l_vec[:,:,:-1,:], axes=(3,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,3,1], value=csdl.sum((Rc[:,:,1:,:]-Rc[:,:,:-1,:])*m_vec[:,:,:-1,:], axes=(3,)))

            # mesh_dict[key]['delta_coll_point'] = delta_coll_point

            # nodal_vel = mesh_dict[key]['nodal_velocity']
            # mesh_dict[key]['nodal_cp_velocity'] = (
            #     nodal_vel[:,:-1,:-1,:]+nodal_vel[:,:-1,1:,:]+\
            #     nodal_vel[:,1:,1:,:]+nodal_vel[:,1:,:-1,:]) / 4.

            # # computing planform area
            # panel_width_spanwise = csdl.norm((mesh[:,:,1:,:] - mesh[:,:,:-1,:]), axes=(3,))
            # avg_panel_width_spanwise = csdl.average(panel_width_spanwise, axes=(1,)) # num_nodes, nt, ns - 1
            # surface_TE = (mesh[:,-1,:-1,:] + mesh[:,0,:-1,:] + mesh[:,-1,1:,:] + mesh[:,0,1:,:])/4
            # surface_LE = (mesh[:,int((nc-1)/2),:-1,:] + mesh[:,int((nc-1)/2),1:,:])/2 # num_nodes, nt, ns - 1, 3

            # chord_spanwise = csdl.norm(surface_TE - surface_LE, axes=(2,)) # num_nodes, nt, ns - 1

            # planform_area = csdl.sum(chord_spanwise*avg_panel_width_spanwise, axes=(1,))
            # mesh_dict[key]['planform_area'] = planform_area

            # formulation from K&P
            mesh = mesh_dict[key]['mesh']
            mesh_shape = mesh.shape
            # print(mesh_shape)
            # exit()
            nc, ns = mesh_shape[-3], mesh_shape[-2]

            mesh_dict[key]['num_panels'] = (nc-1)*(ns-1)
            mesh_dict[key]['nc'] = nc
            mesh_dict[key]['ns'] = ns

            p1 = mesh[:,:-1,:-1,:]
            p2 = mesh[:,1:,:-1,:]
            p3 = mesh[:,1:,1:,:]
            p4 = mesh[:,:-1,1:,:]

            panel_center = (p1 + p2 + p3 + p4)/4.
            mesh_dict[key]['panel_center'] = panel_center

            panel_corners = csdl.Variable(shape=(panel_center.shape[:-1] + (4,3)), value=0.)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,0,:], value=p1)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,1,:], value=p2)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,2,:], value=p3)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,3,:], value=p4)
            mesh_dict[key]['panel_corners'] = panel_corners

            panel_diag_1 = p3-p1
            panel_diag_2 = p4-p2
            
            panel_normal_vec = csdl.cross(panel_diag_1, panel_diag_2, axis=3)
            panel_area = csdl.norm(panel_normal_vec, axes=(3,)) / 2.
            mesh_dict[key]['panel_area'] = panel_area

            panel_x_vec = (p3+p2)/2. - (p1+p4)/2.
            panel_y_vec = (p4+p3)/2. - (p1+p2)/2.

            panel_x_dir = -panel_x_vec / csdl.expand((csdl.norm(panel_x_vec, axes=(3,))), panel_x_vec.shape, 'ijk->ijka')
            panel_y_dir = -panel_y_vec / csdl.expand((csdl.norm(panel_y_vec, axes=(3,))), panel_y_vec.shape, 'ijk->ijka')
            panel_normal = -panel_normal_vec / csdl.expand((csdl.norm(panel_normal_vec, axes=(3,))), panel_normal_vec.shape, 'ijk->ijka')

            mesh_dict[key]['panel_x_dir'] = panel_x_dir
            mesh_dict[key]['panel_y_dir'] = panel_y_dir
            mesh_dict[key]['panel_normal'] = panel_normal

            # COMPUTE DISTANCE FROM PANEL CENTER TO SEGMENT CENTERS

            pos_l = (p3+p2)/2.
            neg_l = (p1+p4)/2.
            pos_m = (p4+p3)/2.
            neg_m = (p1+p2)/2.

            pos_l_norm = csdl.norm(pos_l-panel_center, axes=(3,))
            neg_l_norm = -csdl.norm(neg_l-panel_center, axes=(3,))
            pos_m_norm = csdl.norm(pos_m-panel_center, axes=(3,))
            neg_m_norm = -csdl.norm(neg_m-panel_center, axes=(3,))

            mesh_dict[key]['panel_dl_norm'] = [pos_l_norm, -neg_l_norm]
            mesh_dict[key]['panel_dm_norm'] = [pos_m_norm, -neg_m_norm]

            delta_coll_point = csdl.Variable(panel_center.shape[:-1] + (4,2), value=0.)
            # shape is (num_nodes, nt, nc_panels, ns_panels, 4,2)
            # dimension of size 4 is dl backward, dl forward, dm backward, dm forward
            # dimension of size 2 is the projected deltas in the x or y direction
            # for panel i, the delta to adjacent panel j is taken as "j-i"
            delta_coll_point = delta_coll_point.set(csdl.slice[:,1:,:,0,0], value=csdl.sum((panel_center[:,:-1,:,:] - panel_center[:,1:,:,:] ) * panel_x_dir[:,1:,:,:], axes=(3,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:-1,:,1,0], value=csdl.sum((panel_center[:,1:,:,:] - panel_center[:,:-1,:,:]) * panel_x_dir[:,:-1,:,:], axes=(3,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,2,1], value=csdl.sum((panel_center[:,:,:-1,:] - panel_center[:,:,1:,:]) * panel_y_dir[:,:,1:,:], axes=(3,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,3,1], value=csdl.sum((panel_center[:,:,1:,:] - panel_center[:,:,:-1,:]) * panel_y_dir[:,:,:-1,:], axes=(3,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,:,0,0], value=csdl.sum(panel_center[:,:,:-1,:,:] - panel_center[:,:,1:,:,:], axes=(4,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,:,1,0], value=csdl.sum(panel_center[:,:,1:,:,:] - panel_center[:,:,:-1,:,:], axes=(4,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:,1:,2,1], value=csdl.sum(panel_center[:,:,:,:-1,:] - panel_center[:,:,:,1:,:], axes=(4,)))
            # delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:,:-1,3,1], value=csdl.sum(panel_center[:,:,:,1:,:] - panel_center[:,:,:,:-1,:], axes=(4,)))

            mesh_dict[key]['delta_coll_point'] = delta_coll_point

            # panel_center_mod = panel_center - panel_normal*0.00000001/csdl.expand(panel_area, panel_normal.shape, 'ijkl->ijkla')
            panel_center_mod = panel_center + panel_normal*0.0001
            # panel_center_mod = panel_center - panel_normal*0.0001*csdl.expand(panel_area, panel_normal.shape, 'ijkl->ijkla')
            # panel_center_mod = panel_center
            # mesh_dict[key]['panel_center'] = panel_center_mod

            dpij_global = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,3))))
            dpij_global = dpij_global.set(csdl.slice[:,:,:,0,:], value=p2[:,:,:,:]-p1[:,:,:,:])
            dpij_global = dpij_global.set(csdl.slice[:,:,:,1,:], value=p3[:,:,:,:]-p2[:,:,:,:])
            dpij_global = dpij_global.set(csdl.slice[:,:,:,2,:], value=p4[:,:,:,:]-p3[:,:,:,:])
            dpij_global = dpij_global.set(csdl.slice[:,:,:,3,:], value=p1[:,:,:,:]-p4[:,:,:,:])

            mesh_dict[key]['dpij_global'] = dpij_global

            local_coord_vec = csdl.Variable(shape=(panel_center.shape[:-1] + (3,3)), value=0.) # last two indices are for 3 vectors, 3 dimensions
            local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,0,:], value=panel_x_dir)
            local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,1,:], value=panel_y_dir)
            local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,2,:], value=panel_normal)
            
            mesh_dict[key]['local_coord_vec'] = local_coord_vec

            dpij_local = csdl.einsum(dpij_global, local_coord_vec, action='jklma,jklba->jklmb')  # THIS IS CORRECT

            dpij = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,2))))
            dpij = dpij.set(csdl.slice[:,:,:,0,:], value=dpij_local[:,:,:,0,:2])
            dpij = dpij.set(csdl.slice[:,:,:,1,:], value=dpij_local[:,:,:,1,:2])
            dpij = dpij.set(csdl.slice[:,:,:,2,:], value=dpij_local[:,:,:,2,:2])
            dpij = dpij.set(csdl.slice[:,:,:,3,:], value=dpij_local[:,:,:,3,:2])

            mesh_dict[key]['dpij'] = dpij

            dij = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,))))
            dij = dij.set(csdl.slice[:,:,:,0], value=csdl.norm(dpij[:,:,:,0,:], axes=(3,)))
            dij = dij.set(csdl.slice[:,:,:,1], value=csdl.norm(dpij[:,:,:,1,:], axes=(3,)))
            dij = dij.set(csdl.slice[:,:,:,2], value=csdl.norm(dpij[:,:,:,2,:], axes=(3,)))
            dij = dij.set(csdl.slice[:,:,:,3], value=csdl.norm(dpij[:,:,:,3,:], axes=(3,)))

            mesh_dict[key]['dij'] = dij

            mij = csdl.Variable(shape=dij.shape, value=0.)
            mij = mij.set(csdl.slice[:,:,:,0], value=(dpij[:,:,:,0,1])/(dpij[:,:,:,0,0]+1.e-20))
            mij = mij.set(csdl.slice[:,:,:,1], value=(dpij[:,:,:,1,1])/(dpij[:,:,:,1,0]+1.e-20))
            mij = mij.set(csdl.slice[:,:,:,2], value=(dpij[:,:,:,2,1])/(dpij[:,:,:,2,0]+1.e-20))
            mij = mij.set(csdl.slice[:,:,:,3], value=(dpij[:,:,:,3,1])/(dpij[:,:,:,3,0]+1.e-20))

            mesh_dict[key]['mij'] = mij

            nodal_vel = mesh_dict[key]['nodal_velocity']
            mesh_dict[key]['nodal_cp_velocity'] = (nodal_vel[:,:-1,:-1,:]+nodal_vel[:,:-1,1:,:]+nodal_vel[:,1:,1:,:]+nodal_vel[:,1:,:-1,:]) / 4.
            # mesh_dict[key]['coll_point_velocity'] = (nodal_vel[:,:,:-1,:-1,:]+nodal_vel[:,:,:-1,1:,:]+nodal_vel[:,:,1:,1:,:]+nodal_vel[:,:,1:,:-1,:]) / 4.

            # computing planform area
            panel_width_spanwise = csdl.norm((mesh[:,:,1:,:] - mesh[:,:,:-1,:]), axes=(3,))
            avg_panel_width_spanwise = csdl.average(panel_width_spanwise, axes=(1,)) # num_nodes, nt, ns - 1
            surface_TE = (mesh[:,-1,:-1,:] + mesh[:,0,:-1,:] + mesh[:,-1,1:,:] + mesh[:,0,1:,:])/4
            surface_LE = (mesh[:,int((nc-1)/2),:-1,:] + mesh[:,int((nc-1)/2),1:,:])/2 # num_nodes, nt, ns - 1, 3

            chord_spanwise = csdl.norm(surface_TE - surface_LE, axes=(2,)) # num_nodes, nt, ns - 1

            planform_area = csdl.sum(chord_spanwise*avg_panel_width_spanwise, axes=(1,))
            mesh_dict[key]['planform_area'] = planform_area



    elif mode == 'unstructured':
        pass

    return mesh_dict