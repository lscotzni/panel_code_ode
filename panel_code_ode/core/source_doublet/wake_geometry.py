import numpy as np 
import csdl_alpha as csdl

def wake_geometry(wake_mesh_dict):
    surface_names = list(wake_mesh_dict.keys())
    num_surfaces = len(surface_names)

    for i in range(num_surfaces):
        surface_name = surface_names[i]
        mesh = wake_mesh_dict[surface_name]['mesh']
        mesh_shape = mesh.shape
        nt, ns = mesh_shape[-3], mesh_shape[-2]
        num_panels = (nt-1)*(ns-1)
        wake_mesh_dict[surface_name]['num_panels'] = num_panels
        wake_mesh_dict[surface_name]['ns'] = ns
        wake_mesh_dict[surface_name]['nc'] = nt

        # # VSAERO APPROACH

        # R1 = mesh[:,:-1,:-1,:]
        # R2 = mesh[:,1:,:-1,:]
        # R3 = mesh[:,1:,1:,:]
        # R4 = mesh[:,:-1,1:,:]

        # Rc = (R1+R2+R3+R4)/4.
        # wake_mesh_dict[surface_name]['panel_center'] = Rc

        # panel_corners = csdl.Variable(shape=(Rc.shape[:-1] + (4,3)), value=0.)
        # panel_corners = panel_corners.set(csdl.slice[:,:,:,0,:], value=R1)
        # panel_corners = panel_corners.set(csdl.slice[:,:,:,1,:], value=R2)
        # panel_corners = panel_corners.set(csdl.slice[:,:,:,2,:], value=R3)
        # panel_corners = panel_corners.set(csdl.slice[:,:,:,3,:], value=R4)
        # wake_mesh_dict[surface_name]['panel_corners'] = panel_corners

        # D1 = R3-R1
        # D2 = R4-R2

        # D1D2_cross = csdl.cross(D1, D2, axis=3)
        # D1D2_cross_norm = csdl.norm(D1D2_cross+1.e-12, axes=(3,))
        # panel_area = D1D2_cross_norm/2.
        # wake_mesh_dict[surface_name]['panel_area'] = panel_area

        # normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'ijk->ijka')

        # m_dir = (R3+R4)/2. - Rc
        # m_norm = csdl.norm(m_dir, axes=(3,))
        # m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'ijk->ijka')
        # l_vec = csdl.cross(m_vec, normal_vec, axis=3)

        # wake_mesh_dict[surface_name]['panel_x_dir'] = l_vec
        # wake_mesh_dict[surface_name]['panel_y_dir'] = m_vec
        # wake_mesh_dict[surface_name]['panel_normal'] = normal_vec

        # s = csdl.Variable(shape=(panel_corners.shape[0],) + panel_corners.shape[2:], value=0.)
        # s = s.set(csdl.slice[:,:,:,:-1,:], value=panel_corners[:,:,:,1:,:] - panel_corners[:,:,:,:-1,:])
        # s = s.set(csdl.slice[:,:,:,-1,:], value=panel_corners[:,:,:,0,:] - panel_corners[:,:,:,-1,:])

        # l_exp = csdl.expand(l_vec, s.shape, 'ijkl->ijkal')
        # m_exp = csdl.expand(m_vec, s.shape, 'ijkl->ijkal')
        
        # S = csdl.norm(s+1.e-12, axes=(4,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
        # SL = csdl.sum(s*l_exp+1.e-12, axes=(4,))
        # SM = csdl.sum(s*m_exp+1.e-12, axes=(4,))

        # wake_mesh_dict[surface_name]['S'] = S
        # wake_mesh_dict[surface_name]['SL'] = SL
        # wake_mesh_dict[surface_name]['SM'] = SM




        # # K&P APPROACH

        # wake_mesh_dict[surface_name]['num_points'] = nt*ns

        # wake_mesh_dict[surface_name]['num_panels'] = (nt-1)*(ns-1)
        # wake_mesh_dict[surface_name]['nt'] = nt
        # wake_mesh_dict[surface_name]['ns'] = ns

        # p1 = mesh[:,time_ind,:-1,:-1,:]
        # p2 = mesh[:,time_ind,:-1,1:,:]
        # p3 = mesh[:,time_ind,1:,1:,:]
        # p4 = mesh[:,time_ind,1:,:-1,:]

        p1 = mesh[:,:-1,:-1,:]
        p2 = mesh[:,1:,:-1,:]
        p3 = mesh[:,1:,1:,:]
        p4 = mesh[:,:-1,1:,:]

        panel_center = (p1+p2+p3+p4)/4.
        wake_mesh_dict[surface_name]['panel_center'] = panel_center

        panel_corners = csdl.Variable(panel_center.shape[:-1] + (4,3), value=0.)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,0,:], value=p1)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,1,:], value=p2)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,2,:], value=p3)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,3,:], value=p4)
        wake_mesh_dict[surface_name]['panel_corners'] = panel_corners

        panel_diag_1 = p3-p1
        # panel_diag_2 = p2-p4
        panel_diag_2 = p4-p2
        panel_normal_vec = csdl.cross(panel_diag_1, panel_diag_2, axis=3)
        panel_area = csdl.norm(panel_normal_vec+1.e-12, axes=(3,)) / 2.
        wake_mesh_dict[surface_name]['panel_area'] = panel_area

        # panel_x_vec = (p3+p4)/2. - (p1+p2)/2.
        # panel_y_vec = (p2+p3)/2. - (p1+p4)/2.

        panel_x_vec = (p3+p2)/2. - (p1+p4)/2.
        panel_y_vec = (p4+p3)/2. - (p1+p2)/2.

        panel_x_dir = -panel_x_vec / csdl.expand(csdl.norm(panel_x_vec+1.e-12, axes=(3,)), panel_x_vec.shape, 'ijk->ijka')
        panel_y_dir = -panel_y_vec / csdl.expand(csdl.norm(panel_y_vec+1.e-12, axes=(3,)), panel_y_vec.shape, 'ijk->ijka')
        panel_normal = -panel_normal_vec / csdl.expand(csdl.norm(panel_normal_vec+1.e-12, axes=(3,)), panel_normal_vec.shape, 'ijk->ijka')

        wake_mesh_dict[surface_name]['panel_x_dir'] = panel_x_dir
        wake_mesh_dict[surface_name]['panel_y_dir'] = panel_y_dir
        wake_mesh_dict[surface_name]['panel_normal'] = panel_normal

        # dpij_global = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,3))))
        dpij_global = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,3))))
        dpij_global = dpij_global.set(csdl.slice[:,:,:,0,:], value=p2[:,:,:,:]-p1[:,:,:,:])
        dpij_global = dpij_global.set(csdl.slice[:,:,:,1,:], value=p3[:,:,:,:]-p2[:,:,:,:])
        dpij_global = dpij_global.set(csdl.slice[:,:,:,2,:], value=p4[:,:,:,:]-p3[:,:,:,:])
        dpij_global = dpij_global.set(csdl.slice[:,:,:,3,:], value=p1[:,:,:,:]-p4[:,:,:,:])

        local_coord_vec = csdl.Variable(shape=(panel_center.shape[:-1] + (3,3)), value=0.) # last two indices are for 3 vectors, 3 dimensions
        local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,0,:], value=panel_x_dir)
        local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,1,:], value=panel_y_dir)
        local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,2,:], value=panel_normal)

        dpij_local = csdl.einsum(dpij_global, local_coord_vec, action='jklma,jklba->jklmb')  # THIS IS CORRECT
        
        dpij = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,2))))
        # dpij = dpij.set(csdl.slice[:,time_ind,:,:,0,:], value=p2[:,:,:,:2]-p1[:,:,:,:2])
        # dpij = dpij.set(csdl.slice[:,time_ind,:,:,1,:], value=p3[:,:,:,:2]-p2[:,:,:,:2])
        # dpij = dpij.set(csdl.slice[:,time_ind,:,:,2,:], value=p4[:,:,:,:2]-p3[:,:,:,:2])
        # dpij = dpij.set(csdl.slice[:,time_ind,:,:,3,:], value=p1[:,:,:,:2]-p4[:,:,:,:2])
        dpij = dpij.set(csdl.slice[:,:,:,0,:], value=dpij_local[:,:,:,0,:2])
        dpij = dpij.set(csdl.slice[:,:,:,1,:], value=dpij_local[:,:,:,1,:2])
        dpij = dpij.set(csdl.slice[:,:,:,2,:], value=dpij_local[:,:,:,2,:2])
        dpij = dpij.set(csdl.slice[:,:,:,3,:], value=dpij_local[:,:,:,3,:2])
        wake_mesh_dict[surface_name]['dpij'] = dpij

        dij = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,))))
        dij = dij.set(csdl.slice[:,:,:,0], value=csdl.norm(dpij[:,:,:,0,:]+1.e-12, axes=(3,)))
        dij = dij.set(csdl.slice[:,:,:,1], value=csdl.norm(dpij[:,:,:,1,:]+1.e-12, axes=(3,)))
        dij = dij.set(csdl.slice[:,:,:,2], value=csdl.norm(dpij[:,:,:,2,:]+1.e-12, axes=(3,)))
        dij = dij.set(csdl.slice[:,:,:,3], value=csdl.norm(dpij[:,:,:,3,:]+1.e-12, axes=(3,)))
        wake_mesh_dict[surface_name]['dij'] = dij

        mij = csdl.Variable(shape=dij.shape, value=0.)
        mij = mij.set(csdl.slice[:,:,:,0], value=(dpij[:,:,:,0,1])/(dpij[:,:,:,0,0]+1.e-12))
        mij = mij.set(csdl.slice[:,:,:,1], value=(dpij[:,:,:,1,1])/(dpij[:,:,:,1,0]+1.e-12))
        mij = mij.set(csdl.slice[:,:,:,2], value=(dpij[:,:,:,2,1])/(dpij[:,:,:,2,0]+1.e-12))
        mij = mij.set(csdl.slice[:,:,:,3], value=(dpij[:,:,:,3,1])/(dpij[:,:,:,3,0]+1.e-12))
        wake_mesh_dict[surface_name]['mij'] = mij

        # nodal_vel = wake_mesh_dict[surface_name][key]['nodal_velocity']
        # wake_mesh_dict[surface_name][key]['coll_point_velocity'] = (nodal_vel[:,:,:-1,:-1,:]+nodal_vel[:,:,:-1,1:,:]+nodal_vel[:,:,1:,1:,:]+nodal_vel[:,:,1:,:-1,:]) / 4.

    return wake_mesh_dict