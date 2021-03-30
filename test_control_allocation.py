import numpy as np
import scipy as sp

import matplotlib  
import matplotlib.pyplot as plt

import plotting_tools as plotting_tools
import ss_sim_3dof as dyn


def main():

    
    obstacle_state = [] #update obstacle state list
    sc4 = [np.array([0.3,1,0,0,0,0]),0.5]
    obstacle_state.append(sc4) 

    dt = 0.25
    T = 100
    num_states = 6 
    num_control = 8

    mass = 10
    inertia = 1.62
    l = 0.4
    b = 0.4
    
    param = np.array([mass,inertia,l,b])

    gain = np.array([6,3])

    u_min = 0
    u_max = 0.45

    _, ax1 = plt.subplots(6,1,figsize=(3,5))
    _, ax2 = plt.subplots(8,1,figsize=(3,5))

    desired_state = np.load('x_scp1.npy')
    desired_control = np.load('u_scp1.npy')

    # ax1.plot(desired_state[:,0],desired_state[:,1],color='black',linestyle='dashed',linewidth=2,label='Desired\nTrajectory')

    for i in range(8):
        ax2[i].plot(np.linspace(0,T-2,T-1),desired_control[:,i],color='black')
    x_desired = np.zeros((6,T))
    u_desired = np.zeros((8,T-1))
    for tt in range(T):
        for oo in range(6):
            x_desired[oo,tt] = desired_state[tt,oo]
    for tt in range(T-1):
        for uu in range(8):
            u_desired[uu,tt] = desired_control[tt,uu]
    
    # trying different controllers 

    x1 = np.zeros((num_states,T))
    u1 = np.zeros((num_control,T-1))

    x2 = np.zeros((num_states,T))
    u2 = np.zeros((num_control,T-1))

    x3 = np.zeros((num_states,T))
    u3 = np.zeros((num_control,T-1))


    # setup initial condition
    x1[:,0] = x_desired[:,0] + np.array([0.2,0.0,0.0,0,0,0]) 
    x2[:,0] = x1[:,0]#desired[oo,0] + 0.01*np.random.rand(1)
    x3[:,0] = x1[:,0]#desired[oo,0] + 0.01*np.random.rand(1)

    
    u_old = np.zeros(8)
    actuator_fault = np.eye(8)
    actuator_fault[2,2] = 1
    actuator_fault[7,7] = 0
    actuator_fault[6,6] = 0

    for tt in range(T-1):
            # actuator_fault[3,3] = 0
        u1_force = dyn.ss_3dof_control(x1[:,tt].reshape((6,1)),x_desired[:,tt].reshape((6,1)),u_desired[:,tt].reshape((8,1)),param,gain)   
        u1[:,tt] = dyn.psuedo_inverse_allocation(u1_force,x1[:,tt].reshape((6,1)),u_desired[:,tt],param,u_min,u_max,actuator_fault)
        u1[:,tt] = actuator_fault@u1[:,tt]
        print(u1[:,tt])
        x1[:,tt+1]  = dyn.ss_3dof_dyn(x1[:,tt],u1[:,tt].reshape([8,1]),param,dt) # For Markov Propagation

        u2_force = dyn.ss_3dof_control(x2[:,tt].reshape((6,1)),x_desired[:,tt].reshape((6,1)),u_desired[:,tt].reshape((8,1)),param,gain)   
        u2[:,tt] = dyn.optimization_allocation(u2_force,x2[:,tt].reshape((6,1)),u_desired[:,tt],param,u_min,u_max,actuator_fault)
        u2[:,tt] = actuator_fault@u2[:,tt]
        x2[:,tt+1]  = dyn.ss_3dof_dyn(x2[:,tt],u2[:,tt].reshape([8,1]),param,dt) # For Markov Propagation

        # u2_force = dyn.ss_3dof_control(x2[:,tt].reshape((6,1)),x_desired[:,tt].reshape((6,1)),u_desired[:,tt].reshape((8,1)),param,gain)
        u3_force = dyn.ss_3dof_control(x3[:,tt].reshape((6,1)),x_desired[:,tt].reshape((6,1)),u_desired[:,tt].reshape((8,1)),param,gain)     
        u3[:,tt] = dyn.optimization_stability_allocation(u3_force,x3[:,tt].reshape((6,1)),x_desired[:,tt].reshape((6,1)),u_desired[:,tt].reshape((8,1)),param,u_min,u_max,u_old,actuator_fault)
        u3[:,tt] = actuator_fault@u3[:,tt]
        x3[:,tt+1]  = dyn.ss_3dof_dyn(x3[:,tt],u3[:,tt].reshape([8,1]),param,dt) # For Markov Propagation
        u_old = u3[:,tt]

    state_name = ['x','y','\theta','\dot{x}','\dot{y}','\dot{\theta}']
    for i in range(6):
        ax1[i].plot(np.linspace(0,T-1,T),desired_state[:,i],color='black',linestyle='dashed',linewidth=2,label='desired')
        ax1[i].plot(np.linspace(0,T-1,T),x1[i,:],color='red',linestyle='dashed',linewidth=2,label='LinAlg')
        ax1[i].set_xlabel('time')
        ax1[i].set_ylabel(state_name[i])
        # state_name = ['x','y','\theta','\dot{x}','\dot{y}','\dot{\theta}']
    for i in range(6):
        ax1[i].plot(np.linspace(0,T-1,T),x2[i,:],color='green',linestyle='dashed',linewidth=2,label='Opt')
        # ax1.set_xlabel('time')
        # ax1.set_ylabel(state_name[i])
    for i in range(6):
        ax1[i].plot(np.linspace(0,T-1,T),x3[i,:],color='blue',linestyle='dashed',linewidth=2,label='Opt_Stable')
        # ax1.set_xlabel('time')
        # ax1.set_ylabel(state_name[i])

        # a.plot(x2[0,:],x2[1,:],color='green',linestyle='dashed',linewidth=2,label='Opt')
        # ax1.plot(x3[0,:],x3[1,:],color='blue',linestyle='dashed',linewidth=2,label='Opt_Stability')


    #ax1.set_ylim((100,130))
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Differential\nEntropy (nats)')
    # # ax2.set_ylabel('Fisher Information')
    # #ax2.set_ylim((20,70))

    # ax3.set_xlabel('Epoch')
    # ax3.set_ylabel('Collisions Without\nSafety Filter')
    
    
    # ax1.grid(True,color='lightgrey')
    # ax2.grid(True,color='lightgrey')
    # ax3.grid(True,color='lightgrey')
    # ax1.legend(fontsize='7',loc='best')


    for i in range(8):
        ax2[i].plot(np.linspace(0,T-2,T-1),u1[i,:],color='red')
        ax2[i].set_xlabel('time')
        ax2[i].set_ylabel('T'+str(i))

    for i in range(8):
        ax2[i].plot(np.linspace(0,T-2,T-1),u2[i,:],color='green')
        
    for i in range(8):
        ax2[i].plot(np.linspace(0,T-2,T-1),u3[i,:],color='blue')

    # plt.tight_layout()
    plt.legend(fontsize='7',loc='best')

    plt.show()

    return 1 

if __name__ == "__main__":

    main()


