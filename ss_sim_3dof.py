import numpy as np
import scipy as sp
import math as mt
import matplotlib.pyplot as plt

import cvxpy as cp

"""
For open loop simulation of the 
3DOF dynamics of the spacecraft simulator 
Given : Control Input, Intial Condition and Model Parameters 
Euler Method for Time Integration
"""

def ss_3dof_dyn(x,u,param,dt):

    m = param[0]
    I = param[1]
    l = param[2]
    b = param[3]

    state = x

    mi = 1/m
    lI = l/(2*I)
    bI = b/(2*I)

    B = np.array([[-mi,-mi,0,0,mi,mi,0,0],[0,0,-mi,-mi,0,0,mi,mi],[-lI,lI,-bI,bI,-lI,lI,-bI,bI]])

    R = np.array([[mt.cos(state[2]),mt.sin(state[2]),0],[-mt.sin(state[2]),mt.cos(state[2]),0],[0,0,1]])

    F = np.mat(R)*np.mat(B)*np.mat(u)

    A = np.zeros([6,6])
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1
    
    # Viscous Friction

    A[3,3] = 0
    A[4,4] = 0
    A[5,5] = 0
    
    # Dynamics 

    dxdt = np.mat(A)*np.mat(np.reshape(state,(6,1))) + np.concatenate((np.array([[0],[0],[0]]),F),axis=0) 

    return x + np.reshape(np.array(dxdt),(6))*dt


def EulerInt(dxdt,dt,xt):
    xt1 = xt + dxdt*dt   
    return xt1


## nonlinear controller with exploration feedforward term 

def ss_3dof_control(x,xd,ud,param,gain):

    # m = param[0]
    # I = param[1]
    # l = param[2]
    # b = param[3]

    # state = x

    # mi = 1/m
    # lI = l/(2*I)
    # bI = b/(2*I)

    # B = np.array([[-mi,-mi,0,0,mi,mi,0,0],[0,0,-mi,-mi,0,0,mi,mi],[-lI,lI,-bI,bI,-lI,lI,-bI,bI]])
    # R = np.array([[mt.cos(state[2]),mt.sin(state[2]),0],[-mt.sin(state[2]),mt.cos(state[2]),0],[0,0,1]])

    # force_allocation = np.array(np.mat(np.concatenate((np.zeros((3,3)),R),axis=0))*np.mat(B))

    Kp = gain[0]*np.identity(3)
    Kd = gain[1]*np.identity(3)

    Kdummy = np.concatenate((Kp,Kd),axis=1)
    K = np.concatenate((np.zeros((3,6)),Kdummy),axis=0)
    error = xd - x
    pd = np.array(np.mat(K)*np.mat(error.reshape((6,1))))
    # F =  np.array(np.mat(np.linalg.pinv(force_allocation))*pd)

    # uc = ud.reshape((8)) + F.reshape((8))
    return pd 


def psuedo_inverse_allocation(pd,x,ud,param,u_min,u_max,actuator_fault):
    m = param[0]
    I = param[1]
    l = param[2]
    b = param[3]

    state = x

    mi = 1/m
    lI = l/(2*I)
    bI = b/(2*I)

    B = np.array([[-mi,-mi,0,0,mi,mi,0,0],[0,0,-mi,-mi,0,0,mi,mi],[-lI,lI,-bI,bI,-lI,lI,-bI,bI]])
    R = np.array([[mt.cos(state[2]),mt.sin(state[2]),0],[-mt.sin(state[2]),mt.cos(state[2]),0],[0,0,1]])

    force_allocation = np.array(np.mat(np.concatenate((np.zeros((3,3)),R),axis=0))*np.mat(B))
    F =  np.array(np.mat(np.linalg.pinv(force_allocation))*pd)
    uc = ud.reshape((8)) + F.reshape((8))
    uc = saturation_filter(uc,u_min,u_max,8)
    return actuator_fault@uc 


def optimization_allocation(pd,x,ud,param,u_min,u_max,actuator_fault):

    m = param[0]
    I = param[1]
    l = param[2]
    b = param[3]

    state = x

    mi = 1/m
    lI = l/(2*I)
    bI = b/(2*I)

    B = np.array([[-mi,-mi,0,0,mi,mi,0,0],[0,0,-mi,-mi,0,0,mi,mi],[-lI,lI,-bI,bI,-lI,lI,-bI,bI]])
    R = np.array([[mt.cos(state[2]),mt.sin(state[2]),0],[-mt.sin(state[2]),mt.cos(state[2]),0],[0,0,1]])

    force_allocation = np.array(np.mat(np.concatenate((np.zeros((3,3)),R),axis=0))*np.mat(B))

    u_opt = cp.Variable(8)
    delta = cp.Variable(1)
    delta_u = cp.Variable(1)

    cost = delta_u 
    cost += 1e2*delta
    constr =[u_opt<=u_max, u_opt>=u_min]
    constr.append(cp.quad_form(pd.reshape((6)) - force_allocation@(actuator_fault@u_opt-ud.reshape((8))),np.eye(6))<=delta)
    constr.append(cp.quad_form(actuator_fault@u_opt,np.eye(8))<=delta_u)
    problem = cp.Problem(cp.Minimize(cost),constr)
            
    try:
        result = problem.solve(solver=cp.ECOS)#,verbose=True)
    except:
       print('Exception!')

    #print(delta1.value)
    return u_opt.value.reshape((8))


def optimization_stability_allocation(pd,x,xd,ud,param,u_min,u_max,u_old,actuator_fault):

    m = param[0]
    I = param[1]
    l = param[2]
    b = param[3]
    num_states = 6

    state = x
    stated = xd
    error = x - xd 

    v = (np.linalg.norm(error)**2)/2

    mi = 1/m
    lI = l/(2*I)
    bI = b/(2*I)

    B = np.array([[-mi,-mi,0,0,mi,mi,0,0],[0,0,-mi,-mi,0,0,mi,mi],[-lI,lI,-bI,bI,-lI,lI,-bI,bI]])
    R = np.array([[mt.cos(state[2]),mt.sin(state[2]),0],[-mt.sin(state[2]),mt.cos(state[2]),0],[0,0,1]])

    force_allocation = np.array(np.mat(np.concatenate((np.zeros((3,3)),R),axis=0))*np.mat(B))


    A = np.zeros([6,6])
    A[0,3] = 1
    A[1,4] = 1
    A[2,5] = 1
    
    # Viscous Friction

    A[3,3] = -0.00
    A[4,4] = -0.00
    A[5,5] = 0
    
    
    Rd = np.array([[mt.cos(stated[2]),mt.sin(stated[2]),0],[-mt.sin(stated[2]),mt.cos(stated[2]),0],[0,0,1]])

    force_allocationd = np.mat(np.concatenate((np.zeros((3,3)),Rd),axis=0))*np.mat(B)
    force_allocationd_array = np.array(force_allocationd)

    BB = np.array(np.mat(error.reshape([1,num_states]))*force_allocation)
    BBd = np.array(np.mat(error.reshape([1,num_states]))*force_allocationd*np.mat(ud.reshape([8,1])))
    AA = np.array( np.mat(error.reshape([1,num_states]))*np.mat(A)*np.mat(error.reshape([num_states,1])) )

    u_opt = cp.Variable(8)
    delta = cp.Variable(1)
    delta_v = cp.Variable(1)
    delta_u1 = cp.Variable(1)
    delta_u2 = cp.Variable(1)
    # gamma = cp.Variable(1)


    cost = delta_u1
    # cost += gamma
    cost += delta
    cost += 1e2*delta_v
    # cost+= delta_u2
    
    constr =[u_opt<=u_max, u_opt>=u_min]
    # constr.append(cp.quad_form(pd - force_allocation@(u_opt-ud.reshape((8))),np.eye(6))<=delta)

        # stability
    constr.append(AA + BB@actuator_fault@u_opt - BBd <= -1e3*v + delta)
    constr.append(cp.quad_form(pd.reshape((6)) - force_allocation@actuator_fault@u_opt \
        + force_allocation@ud.reshape((8)),1e3*np.eye(6))<=delta_v)
    constr.append(cp.quad_form(actuator_fault@u_opt,np.eye(8))<=delta_u1)
    # constr.append(cp.norm2(u_opt-u_old.reshape((8)))<=delta_u2)

    problem = cp.Problem(cp.Minimize(cost),constr)
            
    try:
        result = problem.solve(solver=cp.ECOS)#,verbose=True)
    except:
       print('Exception!')

    #print(delta1.value)
    return u_opt.value.reshape((8))



if __name__ == "__main__":
    pass

