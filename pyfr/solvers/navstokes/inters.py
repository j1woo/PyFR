import numpy as np
import numpy.linalg as la

from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)
from pyfr.solvers.euler.inters import (FluidIntIntersMixin,
                                       FluidMPIIntersMixin,
                                       MassFlowBCMixin)


class TplargsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        shock_capturing = self.cfg.get('solver', 'shock-capturing')
        if shock_capturing == 'entropy-filter':
            self.p_min = self.cfg.getfloat('solver-entropy-filter', 'p-min',
                                           1e-6)
        else:
            self.p_min = self.cfg.getfloat('solver-interfaces', 'p-min',
                                           5*self._be.fpdtype_eps)

        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, visc_corr=visc_corr,
                             shock_capturing=shock_capturing, c=self.c,
                             p_min=self.p_min)


class NavierStokesIntInters(TplargsMixin,
                            FluidIntIntersMixin,
                            BaseAdvectionDiffusionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'intconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs,
            ulout=self._comm_lhs, urout=self._comm_rhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artviscl=self._artvisc_lhs, artviscr=self._artvisc_rhs,
            nl=self._pnorm_lhs
        )


class NavierStokesMPIInters(TplargsMixin,
                            FluidMPIIntersMixin,
                            BaseAdvectionDiffusionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'mpiconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs, ulout=self._comm_lhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artviscl=self._artvisc_lhs, artviscr=self._artvisc_rhs,
            nl=self._pnorm_lhs
        )


class NavierStokesBaseBCInters(TplargsMixin, BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional BC specific template arguments
        self._tplargs['bctype'] = self.type
        self._tplargs['bccfluxstate'] = self.cflux_state

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bcconu')
        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'bcconu', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ulin=self._scal_lhs,
            ulout=self._comm_lhs, nlin=self._pnorm_lhs,
            **self._external_vals
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self._scal_lhs,
            gradul=self._vect_lhs, nl=self._pnorm_lhs,
            artviscl=self._artvisc_lhs, **self._external_vals
        )

        if self._ef_enabled:
            self._be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.bccent'
            )

            self.kernels['comm_entropy'] = lambda: self._be.kernel(
                'bccent', tplargs=self._tplargs, dims=[self.ninterfpts],
                extrns=self._external_args, entmin_lhs=self._entmin_lhs,
                nl=self._pnorm_lhs, ul=self._scal_lhs, **self._external_vals
            )

class NavierStokesStaticRobinBCInters(NavierStokesBaseBCInters):
    type = 'static-robin'
    cflux_state = 'ghost-robin'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        # Key parameters

        # Define objects for system
        #self.intg = intg
        #self.elemap_bc = elemap
        #self.comm_BC = comm_BC
        

        #Logistics parameters
        #self.counter_init = 0 #for hakcing elemap problem
        #self.time_index = 1


        # Collecting model parameters and constants
        #gamma = self.cfg.getfloat('constants', 'gamma')
        #self.gamma = gamma 
        #Pr = self.cfg.getfloat('constants', 'Pr')
        #mu = self.cfg.getfloat('constants', 'mu')
        #R = self.cfg.getfloat('constants', 'R')
        

        # Setting up matrices for flux point coordinate data
        spec = f'in fpdtype_t[{self.ndims}]'
        value = self._const_mat(lhs, 'get_ploc_for_inter')
        self._set_external('ploc', spec, value=value)

        f_coords_be = self._external_vals['ploc']
        f_coords = f_coords_be.get()
        custom_shape = (self.ndims,f_coords.shape[1])
        
        # 3D Effects to be communicated to mako files
        #Velocity tangent normalized
        #self.u_hat = be.matrix((custom_shape),tags = {'align'}, dtype = be.fpdtype)
        #self._set_external('u_hat',f'in fpdtype_t[{custom_shape[0]}]',value=self.u_hat)
        
        #Wall model Gradient slip velocity variable
        #self.u_slip = be.matrix((1,f_coords.shape[1]),tags = {'align'}, dtype = be.fpdtype)
        #self._set_external('u_slip',f'in fpdtype_t[{1}]',value=self.u_slip)
        
        # Spanwise basis vector
        #self.khat = be.matrix((custom_shape),tags = {'align'}, dtype = be.fpdtype)
        #self._set_external('khat',f'in fpdtype_t[{custom_shape[0]}]',value=self.khat)

        # Collecting normal vectors:
        #pn1 = self._pnorm_lhs.get()
        # Direct approach to obtaining flux point normals (either way works)
        pn1_obj = self._const_mat(lhs, 'get_pnorms_for_inter')
        pn1 = pn1_obj.get()
        #raise Exception('size of pn1: ',pn1.shape, self.ndims,f_coords.shape)
        pn1_norms = la.norm(pn1,axis=0)
        pn1_norm_tile = np.tile(pn1_norms,(self.ndims,1))
        #Normalized vector
        pn1_hat = -pn1/pn1_norm_tile # minus because all flux norms point outward, we will work into the grid
        self.pn1_hat = pn1_hat.T

        # Grabbing the jacobian determinant per flux point element
        rcpdjac_obj = self._const_mat(lhs,'get_rcpdjac_for_inter')
        djac = 1/rcpdjac_obj.get().flatten() # we called the reciprocal so undo

        #Computing the wall height based on delta h of wall adjacent element
        self.wall_height = 2/(pn1_norms/djac) # based on comp. element length and a somewhat unsovled eq.
        #self.sample_height = self.wall_height.copy() # keep consistent for now
        #Scaling parameters
        #raise Exception('wall height by inters: ',self.wall_height)

        # Static constant
        #c_s = 0.1 # smagorinsky constant
        c_s, = self._eval_opts(['c_s'])
        self.dl = self.wall_height.copy() *c_s # scaling parameter
        #self.uslip_i= (np.zeros_like(self.sample_height))


        # Outputting wall height as delta value
        self.delta = be.matrix((1,f_coords.shape[1]),tags = {'align'}, dtype = be.fpdtype)
        self._set_external('delta',f'in fpdtype_t[{1}]',value=self.delta)

        self.delta.set(np.atleast_2d(self.dl)) 




class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost-imperm'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c['cpTw'], = self._eval_opts(['cpTw'])
        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'
    cflux_state = 'ghost-imperm'


class NavierStokesSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-adia-wall'
    cflux_state = None


class NavierStokesCharRiemInvBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )


class NavierStokesSubInflowFtpttangBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-ftpttang'
    cflux_state = 'ghost'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        gamma = self.cfg.getfloat('constants', 'gamma')

        # Pass boundary constants to the backend
        self.c['cpTt'], = self._eval_opts(['cpTt'])
        self.c['pt'], = self._eval_opts(['pt'])
        self.c['Rdcp'] = (gamma - 1.0)/gamma

        # Calculate u, v velocity components from the inflow angle
        theta = self._eval_opts(['theta'])[0]*np.pi/180.0
        velcomps = np.array([np.cos(theta), np.sin(theta), 1.0])

        # Adjust u, v and calculate w velocity components for 3-D
        if self.ndims == 3:
            phi = self._eval_opts(['phi'])[0]*np.pi/180.0
            velcomps[:2] *= np.sin(phi)
            velcomps[2] *= np.cos(phi)

        self.c['vc'] = velcomps[:self.ndims]


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(['p'], lhs)


class NavierStokesCharRiemInvMassFlowBCInters(MassFlowBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-mass-flow'
    cflux_state = 'ghost'
