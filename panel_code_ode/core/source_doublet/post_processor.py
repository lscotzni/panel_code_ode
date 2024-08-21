import numpy as np 
import csdl_alpha as csdl

def post_processor(Cp_static, mesh_dict, mu, num_nodes, nt, dt):
    surface_names = list(mesh_dict.keys())
    start, stop = 0, 0
    x_dir_global = np.array([1., 0., 0.])
    z_dir_global = np.array([0., 0., 1.])

    output_dict = {}
    for i in range(len(surface_names)):
        surf_dict = {}
        surface_name = surface_names[i]
        num_panels = mesh_dict[surface_name]['num_panels']
        nc, ns = mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
        stop += num_panels

        mu_grid = mu[:,:,start:stop].reshape((nt, num_nodes, nc-1, ns-1))

        panel_normal = mesh_dict[surface_name]['panel_normal']
        nodal_cp_velocity = mesh_dict[surface_name]['nodal_cp_velocity']
        
        try:
            coll_vel = mesh_dict[surface_name]['coll_point_velocity']
        except:
            coll_vel = False

        if coll_vel:
            total_vel = nodal_cp_velocity+coll_vel
        else:
            total_vel = nodal_cp_velocity

        Q_inf_norm = csdl.norm(total_vel, axes=(4,))

        dmu_dt = csdl.Variable(shape=Q_inf_norm.shape, value=0)
        dmu_dt = dmu_dt.set(csdl.slice[:,1:,:,:], value=(mu_grid[:,:-1,:,:] - mu_grid[:,1:,:,:])/dt)
        Cp_dynamic = -dmu_dt*2./Q_inf_norm**2
        Cp = Cp_static + Cp_dynamic
        # Cp = 1 - (Ql**2 + Qm**2 + Qn**2)/Q_inf_norm**2 - dmu_dt*2./Q_inf_norm**2
        # Cp = 1 - (Ql**2 + Qn**2)/Q_inf_norm**2 - dmu_dt*2./Q_inf_norm**2
        
        panel_area = mesh_dict[surface_name]['panel_area']

        rho = 1.225
        # rho = 1000.
        dF_no_normal = -0.5*rho*Q_inf_norm**2*panel_area*Cp
        dF = csdl.expand(dF_no_normal, panel_normal.shape, 'ijkl->ijkla') * -panel_normal

        Fz_panel = csdl.tensordot(dF, z_dir_global, axes=([4],[0]))
        Fx_panel = csdl.tensordot(dF, x_dir_global, axes=([4],[0]))

        nc_panels = int(num_panels/(ns-1))

        LE_velocity = (total_vel[:,:,int((nc_panels/2)-1),:,:] + total_vel[:,:,int(nc_panels/2),:,:])/2.
        aoa = csdl.arctan(LE_velocity[:,:,:,2]/LE_velocity[:,:,:,0])

        aoa_exp = csdl.expand(aoa, Fz_panel.shape, 'ijk->ijak')

        cosa, sina = csdl.cos(aoa_exp), csdl.sin(aoa_exp)

        panel_L = Fz_panel*cosa - Fx_panel*sina
        panel_Di = Fz_panel*sina + Fx_panel*cosa

        L = csdl.sum(panel_L, axes=(2,3))
        Di = csdl.sum(panel_Di, axes=(2,3))

        Q_inf = csdl.norm(csdl.average(LE_velocity, axes=(2,)), axes=(2,))

        planform_area = mesh_dict[surface_name]['planform_area']
        CL = L/(0.5*rho*planform_area*Q_inf**2)
        CDi = Di/(0.5*rho*planform_area*Q_inf**2)

        surf_dict['Cp'] = Cp
        surf_dict['CL'] = CL
        surf_dict['CDi'] = CDi
        surf_dict['Fx_panel'] = Fx_panel
        surf_dict['Fz_panel'] = Fz_panel
        surf_dict['panel_forces'] = dF
        
        output_dict[surface_name] = surf_dict

        # print(CL.value)
        # print(CDi.value)

        start += num_panels


    return output_dict

