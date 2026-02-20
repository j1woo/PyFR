<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
//#include <stdio.h>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur,grad_ul' externs='ploc, t, delta'>
    ur[0] = ul[0];

    //Preparing for robin condition
    fpdtype_t rcprho = 1.0/ur[0];
    
    // 2D Version 
% if ndims == 2:
    fpdtype_t u = rcprho*ul[1], v = rcprho*ul[2];

    // Velocity derivatives (rho*grad[u,v])
    fpdtype_t u_x = rcprho*(grad_ul[0][1] - u*grad_ul[0][0]);
    fpdtype_t u_y = rcprho*(grad_ul[1][1] - u*grad_ul[1][0]);

    fpdtype_t v_x = rcprho*(grad_ul[0][2] - v*grad_ul[0][0]);
    fpdtype_t v_y = rcprho*(grad_ul[1][2] - v*grad_ul[1][0]);

    //fpdtype_t xploc = ploc[0];
    //printf("Gradient: %f and u: %f and x: %f\n", u_y, u, xploc);

    // Convert to wall normal derivative
    fpdtype_t u_n = -(u_x*nl[0] + u_y*nl[1]);
    fpdtype_t v_n = -(v_x*nl[0] + v_y*nl[1]);


    // Compute robin condition for value
    fpdtype_t u_rob = delta[0]*u_n;
    //u_rob = delta[0]*903.7087;
    fpdtype_t v_rob = delta[0]*v_n;

    //Remove normal component
    fpdtype_t u_ortho = u_rob*nl[0] + v_rob*nl[1];
    u_rob = u_rob-u_ortho*nl[0];
    v_rob = v_rob- u_ortho*nl[1];

    //printf("Checking vrob: %f \n",v_rob);


    // Convert back to primitive and assign to right side
    ur[1] =-ul[1] + 2*u_rob*ur[0];
    ur[2] =-ul[2] + 2*v_rob*ur[0];
    //ur[2] =-0.5*ul[2] + 2*v_rob*ur[0];

    fpdtype_t p_term = ul[${nvars - 1}] - (0.5/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))};
    ur[${nvars - 1}] = p_term + 0.5*ul[0]*(u_rob*u_rob + v_rob*v_rob); 

    
    // 3D Version 
% elif ndims == 3:
    fpdtype_t u = rcprho*ul[1], v = rcprho*ul[2], w = rcprho*ul[3];

    // Velocity derivatives (rho*grad[u,v,w])
    fpdtype_t u_x = rcprho*(grad_ul[0][1] - u*grad_ul[0][0]);
    fpdtype_t u_y = rcprho*(grad_ul[1][1] - u*grad_ul[1][0]);
    fpdtype_t u_z = rcprho*(grad_ul[2][1] - u*grad_ul[2][0]);
    fpdtype_t v_x = rcprho*(grad_ul[0][2] - v*grad_ul[0][0]);
    fpdtype_t v_y = rcprho*(grad_ul[1][2] - v*grad_ul[1][0]);
    fpdtype_t v_z = rcprho*(grad_ul[2][2] - v*grad_ul[2][0]);
    fpdtype_t w_x = rcprho*(grad_ul[0][3] - w*grad_ul[0][0]);
    fpdtype_t w_y = rcprho*(grad_ul[1][3] - w*grad_ul[1][0]);
    fpdtype_t w_z = rcprho*(grad_ul[2][3] - w*grad_ul[2][0]);

    // Convert to wall normal derivative
    fpdtype_t u_n = -(u_x*nl[0] + u_y*nl[1] + u_z*nl[2]);
    fpdtype_t v_n = -(v_x*nl[0] + v_y*nl[1] + v_z*nl[2]);
    fpdtype_t w_n = -(w_x*nl[0] + w_y*nl[1] + w_z*nl[2]);

    // Compute robin condition for value
    fpdtype_t u_rob = delta[0]*u_n;
    fpdtype_t v_rob = delta[0]*v_n;
    fpdtype_t w_rob = delta[0]*w_n;

    // Convert back to primitive and assign to right side
    ur[1] =-ul[1] + 2*u_rob*ur[0];
    ur[2] =-ul[2] + 2*v_rob*ur[0];
    ur[3] =-ul[3] + 2*w_rob*ur[0];

    fpdtype_t p_term = ul[${nvars - 1}] - (0.5/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))};
    ur[${nvars - 1}] = p_term + 0.5*ul[0]*(u_rob*u_rob + v_rob*v_rob + w_rob*w_rob); 
% endif
</%pyfr:macro>





<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t'>
    ur[0] = ul[0];
% for i in range(ndims):
    ur[${i + 1}] = ul[${i+1}];
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}];
    //- (0.5/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:macro name='bc_ldg_grad_state' params='ur, nl, grad_ul, grad_ur'>
    fpdtype_t rcprho = 1.0/ur[0];

% if ndims == 2:
    fpdtype_t u = rcprho*ur[1], v = rcprho*ur[2];

    fpdtype_t u_x = grad_ul[0][1] - u*grad_ul[0][0];
    fpdtype_t u_y = grad_ul[1][1] - u*grad_ul[1][0];
    fpdtype_t v_x = grad_ul[0][2] - v*grad_ul[0][0];
    fpdtype_t v_y = grad_ul[1][2] - v*grad_ul[1][0];

    // Compute temperature derivatives (c_v*rho*dT/d[x,y,z])
    fpdtype_t Tl_x = grad_ul[0][3] - (rcprho*grad_ul[0][0]*ur[3]
                                      + u*u_x + v*v_x);
    fpdtype_t Tl_y = grad_ul[1][3] - (rcprho*grad_ul[1][0]*ur[3]
                                      + u*u_y + v*v_y);

    // Copy all fluid-side gradients across to wall-side gradients
    ${pyfr.expand('bc_common_grad_copy', 'ur', 'nl', 'grad_ul', 'grad_ur')};

    // Correct copied across in-fluid temp gradients to in-wall gradients
    grad_ur[0][3] -= nl[0]*nl[0]*Tl_x + nl[0]*nl[1]*Tl_y;
    grad_ur[1][3] -= nl[1]*nl[0]*Tl_x + nl[1]*nl[1]*Tl_y;

% elif ndims == 3:
    fpdtype_t u = rcprho*ur[1], v = rcprho*ur[2], w = rcprho*ur[3];

    // Velocity derivatives (rho*grad[u,v,w])
    fpdtype_t u_x = grad_ul[0][1] - u*grad_ul[0][0];
    fpdtype_t u_y = grad_ul[1][1] - u*grad_ul[1][0];
    fpdtype_t u_z = grad_ul[2][1] - u*grad_ul[2][0];
    fpdtype_t v_x = grad_ul[0][2] - v*grad_ul[0][0];
    fpdtype_t v_y = grad_ul[1][2] - v*grad_ul[1][0];
    fpdtype_t v_z = grad_ul[2][2] - v*grad_ul[2][0];
    fpdtype_t w_x = grad_ul[0][3] - w*grad_ul[0][0];
    fpdtype_t w_y = grad_ul[1][3] - w*grad_ul[1][0];
    fpdtype_t w_z = grad_ul[2][3] - w*grad_ul[2][0];

    // Compute temperature derivatives (c_v*rho*dT/d[x,y,z])
    fpdtype_t Tl_x = grad_ul[0][4] - (rcprho*grad_ul[0][0]*ur[4]
                                      + u*u_x + v*v_x + w*w_x);
    fpdtype_t Tl_y = grad_ul[1][4] - (rcprho*grad_ul[1][0]*ur[4]
                                      + u*u_y + v*v_y + w*w_y);
    fpdtype_t Tl_z = grad_ul[2][4] - (rcprho*grad_ul[2][0]*ur[4]
                                      + u*u_z + v*v_z + w*w_z);

    // Copy all fluid-side gradients across to wall-side gradients
    ${pyfr.expand('bc_common_grad_copy', 'ur', 'nl', 'grad_ul', 'grad_ur')};

    // Correct copied across in-fluid temp gradients to in-wall gradients
    grad_ur[0][4] -= nl[0]*nl[0]*Tl_x + nl[0]*nl[1]*Tl_y + nl[0]*nl[2]*Tl_z;
    grad_ur[1][4] -= nl[1]*nl[0]*Tl_x + nl[1]*nl[1]*Tl_y + nl[1]*nl[2]*Tl_z;
    grad_ur[2][4] -= nl[2]*nl[0]*Tl_x + nl[2]*nl[1]*Tl_y + nl[2]*nl[2]*Tl_z;
% endif
</%pyfr:macro>
