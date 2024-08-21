import numpy as np 
import csdl_alpha as csdl

from VortexAD.utils.atan2_switch import atan2_switch

def compute_doublet_influence(dpij, mij, ek, hk, rk, dx, dy, dz, mu=1., mode='potential'):
    # each input is a list of length 4, holding the values for the corners
    # ex: mij = [mij_1. mij_2, mij_3, mij_4]

    num_edges = len(dx) # tells us the number of vertices per panel (hence the edges)
    if mode == 'potential':

        atan2_terms = []
        for i in range(num_edges):
            n = i+1
            if i == num_edges-1: # end
                n = 0

            t_y = dz[i]*dpij[i][0] * ((ek[i]*dpij[i][1]-hk[i]*dpij[i][0])*rk[n] - (ek[n]*dpij[i][1]-hk[n]*dpij[i][0])*rk[i])
            t_x = dz[i]**2*rk[i]*rk[n]*dpij[i][0]**2 + (ek[i]*dpij[i][1]-hk[i]*dpij[i][0])*(ek[n]*dpij[i][1]-hk[n]*dpij[i][0])

            # atan2_term = 2*csdl.arctan(t_y / ((t_x**2 + t_y**2 + 1.e-12)**0.5 + t_x))
            atan2_term = 2*csdl.arctan(((t_x**2 + t_y**2)**0.5 - t_x) / (t_y + 1.e-12))
            atan2_terms.append(atan2_term)

        doublet_potential = mu/(4*np.pi) * sum(atan2_terms)


        t1_y = dz[0]*dpij[0][0] * ((ek[0]*dpij[0][1]-hk[0]*dpij[0][0])*rk[1] - (ek[1]*dpij[0][1]-hk[1]*dpij[0][0])*rk[0])
        t2_y = dz[1]*dpij[1][0] * ((ek[1]*dpij[1][1]-hk[1]*dpij[1][0])*rk[2] - (ek[2]*dpij[1][1]-hk[2]*dpij[1][0])*rk[1])
        t3_y = dz[2]*dpij[2][0] * ((ek[2]*dpij[2][1]-hk[2]*dpij[2][0])*rk[3] - (ek[3]*dpij[2][1]-hk[3]*dpij[2][0])*rk[2])
        t4_y = dz[3]*dpij[3][0] * ((ek[3]*dpij[3][1]-hk[3]*dpij[3][0])*rk[0] - (ek[0]*dpij[3][1]-hk[0]*dpij[3][0])*rk[3])

        t1_x = dz[0]**2*rk[0]*rk[1]*dpij[0][0]**2 + (ek[0]*dpij[0][1]-hk[0]*dpij[0][0])*(ek[1]*dpij[0][1]-hk[1]*dpij[0][0])
        t2_x = dz[1]**2*rk[1]*rk[2]*dpij[1][0]**2 + (ek[1]*dpij[1][1]-hk[1]*dpij[1][0])*(ek[2]*dpij[1][1]-hk[2]*dpij[1][0])
        t3_x = dz[2]**2*rk[2]*rk[3]*dpij[2][0]**2 + (ek[2]*dpij[2][1]-hk[2]*dpij[2][0])*(ek[3]*dpij[2][1]-hk[3]*dpij[2][0])
        t4_x = dz[3]**2*rk[3]*rk[0]*dpij[3][0]**2 + (ek[3]*dpij[3][1]-hk[3]*dpij[3][0])*(ek[0]*dpij[3][1]-hk[0]*dpij[3][0])

        # atan2_1 = 2*csdl.arctan(t1_y / ((t1_x**2 + t1_y**2 + 1.e-12)**0.5 + t1_x)) # 1
        # # atan2_2 = 2*csdl.arctan(t2_y / ((t2_x**2 + t2_y**2 + 1.e-12)**0.5 + t2_x)) # 2 
        # atan2_3 = 2*csdl.arctan(t3_y / ((t3_x**2 + t3_y**2 + 1.e-12)**0.5 + t3_x)) # 3
        # # atan2_4 = 2*csdl.arctan(t4_y / ((t4_x**2 + t4_y**2 + 1.e-12)**0.5 + t4_x)) # 4

        # # atan2_1 = 2*csdl.arctan(((t1_x**2 + t1_y**2)**0.5 - t1_x) / (t1_y + 1.e-12)) # 5
        # atan2_2 = 2*csdl.arctan(((t2_x**2 + t2_y**2)**0.5 - t2_x) / (t2_y + 1.e-12)) # 6
        # # atan2_3 = 2*csdl.arctan(((t3_x**2 + t3_y**2)**0.5 - t3_x) / (t3_y + 1.e-12)) # 7
        # atan2_4 = 2*csdl.arctan(((t4_x**2 + t4_y**2)**0.5 - t4_x) / (t4_y + 1.e-12)) # 8

        # # NOTE-S
        # # 1-3-6-8 works compared to Jiayao's code
        # # test other combinations

        doublet_potential = mu/4/np.pi*(
            csdl.arctan((mij[0]*ek[0]-hk[0])/(dz[0]*rk[0]+1.e-12)) - csdl.arctan((mij[0]*ek[1]-hk[1])/(dz[0]*rk[1]+1.e-12)) + 
            csdl.arctan((mij[1]*ek[1]-hk[1])/(dz[1]*rk[1]+1.e-12)) - csdl.arctan((mij[1]*ek[2]-hk[2])/(dz[1]*rk[2]+1.e-12)) + 
            csdl.arctan((mij[2]*ek[2]-hk[2])/(dz[2]*rk[2]+1.e-12)) - csdl.arctan((mij[2]*ek[3]-hk[3])/(dz[2]*rk[3]+1.e-12)) + 
            csdl.arctan((mij[3]*ek[3]-hk[3])/(dz[3]*rk[3]+1.e-12)) - csdl.arctan((mij[3]*ek[0]-hk[0])/(dz[3]*rk[0]+1.e-12))
        ) # note that dk[:,i,2] is the same for all i

        # doublet_potential = mu/4/np.pi*( atan2_1 + atan2_2 + atan2_3 + atan2_4) # note that dk[:,i,2] is the same for all i

        return doublet_potential
    elif mode == 'velocity':
        core_size = 1.e-4
        doublet_velocity = csdl.Variable(shape=dx[0].shape + (3,), value=0.)
        r1r2 = rk[0]*rk[1]
        r2r3 = rk[1]*rk[2]
        r3r4 = rk[2]*rk[3]
        r4r1 = rk[3]*rk[0]
        
        u = mu/(4*np.pi) * (
            (dz[0] * -dpij[0][1] * (rk[0]+rk[1]))/(r1r2*(r1r2-(dx[0]*dx[1]+dy[0]*dy[1]+dz[0]*dz[1])) + core_size) + 
            (dz[1] * -dpij[1][1] * (rk[1]+rk[2]))/(r2r3*(r2r3-(dx[1]*dx[2]+dy[1]*dy[2]+dz[1]*dz[2])) + core_size) + 
            (dz[2] * -dpij[2][1] * (rk[2]+rk[3]))/(r3r4*(r3r4-(dx[2]*dx[3]+dy[2]*dy[3]+dz[2]*dz[3])) + core_size) + 
            (dz[3] * -dpij[3][1] * (rk[3]+rk[0]))/(r4r1*(r4r1-(dx[3]*dx[0]+dy[3]*dy[0]+dz[3]*dz[0])) + core_size) 
        )

        v = mu/(4*np.pi) * (
            (dz[0] * dpij[0][0] * (rk[0]+rk[1]))/(r1r2*(r1r2-(dx[0]*dx[1]+dy[0]*dy[1]+dz[0]*dz[1])) + core_size) + 
            (dz[1] * dpij[1][0] * (rk[1]+rk[2]))/(r2r3*(r2r3-(dx[1]*dx[2]+dy[1]*dy[2]+dz[1]*dz[2])) + core_size) + 
            (dz[2] * dpij[2][0] * (rk[2]+rk[3]))/(r3r4*(r3r4-(dx[2]*dx[3]+dy[2]*dy[3]+dz[2]*dz[3])) + core_size) + 
            (dz[3] * dpij[3][0] * (rk[3]+rk[0]))/(r4r1*(r4r1-(dx[3]*dx[0]+dy[3]*dy[0]+dz[3]*dz[0])) + core_size) 
        )

        w = mu/(4*np.pi) * (
            (2*dz[0]**2*dpij[0][1]*dpij[0][0]*rk[0] - (dpij[0][1]*ek[0]-hk[0]*dpij[0][0])*(rk[0]+dz[0]**2/(rk[0] + core_size))*dpij[0][0]) / ((dz[0]*rk[0]*dpij[0][0])**2 + (dpij[0][1]*ek[0]-hk[0]*dpij[0][0])**2 + core_size) - # Term 1
            (2*dz[1]**2*dpij[0][1]*dpij[0][0]*rk[1] - (dpij[0][1]*ek[1]-hk[1]*dpij[0][0])*(rk[1]+dz[1]**2/(rk[1] + core_size))*dpij[0][0]) / ((dz[0]*rk[1]*dpij[0][0])**2 + (dpij[0][1]*ek[1]-hk[1]*dpij[0][0])**2 + core_size) + # Term 2

            (2*dz[1]**2*dpij[1][1]*dpij[1][0]*rk[1] - (dpij[1][1]*ek[1]-hk[1]*dpij[1][0])*(rk[1]+dz[1]**2/(rk[1] + core_size))*dpij[1][0]) / ((dz[1]*rk[1]*dpij[1][0])**2 + (dpij[1][1]*ek[1]-hk[1]*dpij[1][0])**2 + core_size) - # Term 3
            (2*dz[2]**2*dpij[1][1]*dpij[1][0]*rk[2] - (dpij[1][1]*ek[2]-hk[2]*dpij[1][0])*(rk[2]+dz[2]**2/(rk[2] + core_size))*dpij[1][0]) / ((dz[1]*rk[2]*dpij[1][0])**2 + (dpij[1][1]*ek[2]-hk[2]*dpij[1][0])**2 + core_size) + # Term 4

            (2*dz[2]**2*dpij[2][1]*dpij[2][0]*rk[2] - (dpij[2][1]*ek[2]-hk[2]*dpij[2][0])*(rk[2]+dz[2]**2/(rk[2] + core_size))*dpij[2][0]) / ((dz[2]*rk[2]*dpij[2][0])**2 + (dpij[2][1]*ek[2]-hk[2]*dpij[2][0])**2 + core_size) - # Term 5
            (2*dz[3]**2*dpij[2][1]*dpij[2][0]*rk[3] - (dpij[2][1]*ek[3]-hk[3]*dpij[2][0])*(rk[3]+dz[3]**2/(rk[3] + core_size))*dpij[2][0]) / ((dz[2]*rk[3]*dpij[2][0])**2 + (dpij[2][1]*ek[3]-hk[3]*dpij[2][0])**2 + core_size) + # Term 6

            (2*dz[3]**2*dpij[3][1]*dpij[3][0]*rk[3] - (dpij[3][1]*ek[3]-hk[3]*dpij[3][0])*(rk[3]+dz[3]**2/(rk[3] + core_size))*dpij[3][0]) / ((dz[3]*rk[3]*dpij[3][0])**2 + (dpij[3][1]*ek[3]-hk[3]*dpij[3][0])**2 + core_size) - # Term 7
            (2*dz[0]**2*dpij[3][1]*dpij[3][0]*rk[0] - (dpij[3][1]*ek[0]-hk[0]*dpij[3][0])*(rk[0]+dz[0]**2/(rk[0] + core_size))*dpij[3][0]) / ((dz[3]*rk[0]*dpij[3][0])**2 + (dpij[3][1]*ek[0]-hk[0]*dpij[3][0])**2 + core_size) # Term 8
        )

        # NOTE: we can't set this to a csdl variable here because the shapes of the inputs change (like dx[0] based on the number of timesteps)
        # doublet_velocity = doublet_velocity.set(csdl.slice[], value=u)
        # doublet_velocity = doublet_velocity.set(csdl.slice[], value=v)
        # doublet_velocity = doublet_velocity.set(csdl.slice[], value=w)
        return u, v, w
    
def compute_doublet_influence_new(A, AM, B, BM, SL, SM, A1, PN, mode='potential', mu=1.):
    '''
    Function to compute induced potential of a doublet panel based on the 
    VSAERO documentation found here: https://ntrs.nasa.gov/citations/19900004884

    This function takes inputs related to panel parameters and data about
    the point where potential is induced.

    Each input is a list (all of the same length), where the number of panel
    sides equals the length of the lists.
    '''
    if mode == 'potential':

        panel_segment_potential = []
        num_sides = len(A)
        
        for i in range(num_sides):

            PA = PN[i]**2*SL[i] + A1[i]*AM[i]
            PB = PN[i]**2*SL[i] + A1[i]*BM[i]

            RNUM = SM[i]*PN[i]*(B[i]*PA - A[i]*PB)
            DNOM = PA*PB + PN[i]**2*A[i]*B[i]*SM[i]**2
    
            asdf = csdl.arctan(RNUM/(DNOM+1.e-12)) # NOTE: add some numerical softening here

            asdf = 2*csdl.arctan(((RNUM**2 + DNOM**2)**0.5 - DNOM) / (RNUM+1.e-12)) # half angle formula
            # asdf = 2*csdl.arctan((RNUM/((RNUM**2+DNOM**2)**0.5+DNOM)))

            panel_segment_potential.append(asdf)

        doublet_potential = mu/(4*np.pi) * sum(panel_segment_potential)

        return doublet_potential
    
def compute_vortex_line_ind_vel(p1, p2, p_eval, gamma=1., mode='surface', vc=None):

    if mode == 'surface':
        expand_str = 'ijk->ijka'
    elif mode == 'wake':
        expand_str = 'ij->ija'

    r1 = p_eval-p1
    r2 = p_eval-p2
    input_shape = r1.shape
    xyz_dim = len(input_shape) - 1

    # compute norms of r1 and r2
    r1_norm = csdl.norm(r1+1.e-12, axes=(xyz_dim, ))
    r1_norm_exp = csdl.expand(r1_norm, r1.shape, expand_str)
    r2_norm = csdl.norm(r2+1.e-12, axes=(xyz_dim, ))
    r2_norm_exp = csdl.expand(r2_norm, r2.shape, expand_str)

    # find cross product and norm
    r1r2_cross = csdl.cross(r1, r2, axis=xyz_dim)
    r1r2_cross_norm = csdl.norm(r1r2_cross+1.e-12, axes=(xyz_dim,))
    r1r2_cross_norm_exp = csdl.expand(r1r2_cross_norm, r1.shape, expand_str)

    if vc is None:
        # compute dot products 
        r0 = r1-r2
        r0r1_dot = csdl.sum(r0*r1, axes=(xyz_dim,))
        r0r1_dot_exp = csdl.expand(r0r1_dot, r1.shape, expand_str)
        r0r2_dot = csdl.sum(r0*r2, axes=(xyz_dim,))
        r0r2_dot_exp = csdl.expand(r0r2_dot, r2.shape, expand_str)

        induced_vel = gamma/(4*np.pi)*r1r2_cross/(r1r2_cross_norm_exp + 1.e-8)**2 * \
                    (r0r1_dot_exp/r1_norm_exp - r0r2_dot_exp/r2_norm_exp)
    else:
        # finite core from BYU VortexLattice approach:
        # if finite_core
        #     rdot = dot(r1, r2)
        #     r1s, r2s, εs = nr1^2, nr2^2, core_size^2
        #     f1 = cross(r1, r2)/(r1s*r2s - rdot^2 + εs*(r1s + r2s - 2*nr1*nr2))
        #     f2 = (r1s - rdot)/sqrt(r1s + εs) + (r2s - rdot)/sqrt(r2s + εs)

        # Vhat = (f1*f2)/(4*pi)
        rdot = csdl.sum(r1*r2, axes=(xyz_dim,))
        rdot_exp = csdl.expand(rdot, r1.shape, 'ij->ija')

        r1s = r1_norm_exp**2
        r2s = r2_norm_exp**2
        eps_s = vc**2

        f1 = r1r2_cross/(r1s*r2s - rdot_exp**2 + vc*(r1s+r2s-2*r1_norm_exp*r2_norm_exp) + 1.e-12)
        f2 = (r1s - rdot_exp)/(r1s+eps_s)**0.5 + (r2s - rdot_exp)/(r2s+eps_s)**0.5
        induced_vel = f1*f2/4./np.pi
    
    return induced_vel