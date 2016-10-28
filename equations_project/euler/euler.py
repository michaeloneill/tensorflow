import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pdb


# set up plot environment
newparams = {'axes.labelsize': 11, 'axes.linewidth': 1, 'savefig.dpi': 300,
             'ytick.labelsize': 10, 'xtick.labelsize': 10,
             'legend.fontsize': 10, 'legend.frameon': True,
             'legend.handlelength': 1.5}

plt.rcParams.update(newparams)


def build_train_graph(loss):

    with tf.name_scope('train'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)

        optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
        train_step = optimizer.apply_gradients(zip(grads, tvars))

        return train_step


def energy(rho, u, p, GAMMA):
    return 0.5 * rho * u ** 2 + p/(GAMMA - 1)

def pressure(rho, u, E, GAMMA):
    return (GAMMA - 1) * (E - 0.5 * rho * u ** 2)


def tf_flux(Q, GAMMA):
    
    ''' Computes cell fluxes '''
    
    rho, u, E = Q[0], Q[1]/Q[0], Q[2]
    p = pressure(rho, u, E, GAMMA)

    F_1 = rho * u
    F_2 = rho * u ** 2 + p
    F_3 = u * (E + p)
    return tf.pack([F_1, F_2, F_3], 0)


def flux(Q, GAMMA):
    
    ''' Computes cell fluxes '''
    
    rho, u, E = Q[0], Q[1]/Q[0], Q[2]
    p = pressure(rho, u, E, GAMMA)
    F = np.empty_like(Q)
    F[0] = rho * u
    F[1] = rho * u ** 2 + p
    F[2] = u * (E + p)
    return F


def tf_timestep(Q, params):
    rho, u, E = Q[0], Q[1]/Q[0], Q[2]
    a = tf.sqrt(params['GAMMA'] * pressure(rho, u, E, params['GAMMA']) / rho)
    S_max = tf.reduce_max(tf.abs(u) + a)
    return params['c'] * params['dx'] / S_max


def timestep(Q, params):
    rho, u, E = Q[0], Q[1]/Q[0], Q[2]
    a = np.sqrt(params['GAMMA'] * pressure(rho, u, E, params['GAMMA']) / rho)
    S_max = np.max(np.abs(u) + a)
    return params['c'] * params['dx'] / S_max


def tf_force(Q, params):

    ''' Computes inter-cell fluxes using the FORCE scheme '''

    dt = tf_timestep(Q, params)
    
    Q_L = Q[:, :-1]
    Q_R = Q[:, 1:]


    F_L = tf_flux(Q_L, params['GAMMA'])
    F_R = tf_flux(Q_R, params['GAMMA'])

    Q_0 = 0.5 * (Q_L + Q_R) + 0.5 * dt / params['dx'] * (F_L - F_R)
    F_0 = tf_flux(Q_0, params['GAMMA'])

    return 0.5 * (F_0 + 0.5 * (F_L + F_R)) - 0.25 * params['dx'] / dt * (Q_R - Q_L)


def force(Q, params):

    ''' Computes inter-cell fluxes using the FORCE scheme '''

    dt = timestep(Q, params)
    
    Q_L = Q[:, :-1]
    Q_R = Q[:, 1:]


    F_L = flux(Q_L, params['GAMMA'])
    F_R = flux(Q_R, params['GAMMA'])

    Q_0 = 0.5 * (Q_L + Q_R) + 0.5 * dt / params['dx'] * (F_L - F_R)
    F_0 = flux(Q_0, params['GAMMA'])

    return 0.5 * (F_0 + 0.5 * (F_L + F_R)) - 0.25 * params['dx'] / dt * (Q_R - Q_L)


def main():

    
    params = {'GAMMA': 1.4, # adiabatic index for air
              'N': 100, # number of cells
              'dx': 1./100, # spatial step size
              'c': 0.9, # CFL coefficient
              'T': 0.25 # max time
    }

    # set up spatial domain with ghost cells at either end for boundary conditions
    x = np.linspace(-0.5 * params['dx'], 1 + 0.5 * params['dx'], params['N'] + 2) 


    # initialise state vector Q according to
    # a tube with a membrane separating air
    # of two different densities (and pressures)
    # a.k.a. Sod's shock tube

    Q = np.zeros((3, len(x))) 
    
    # density
    Q[0, x <= 0.5] = 1.0
    Q[0, x > 0.5] = 0.125

    # momentum
    Q[1] = 0.0

    # energy
    Q[2, x <= 0.5] = energy(1.0, 0.0, 1.0, params['GAMMA'])
    Q[2, x > 0.5] = energy(0.125, 0.0, 0.1, params['GAMMA'])


    # evolve system
    
    # t = 0
    # while t < params['T']:
    for i in range(100):
        # Compute time step at current Q
        dt = timestep(Q, params)

        # if t + dt > params['T']:
        #     dt = params['T']-t # Ensure we end at T

        Q[:, 0] = Q[:, 1] # left boundary
        Q[:, params['N'] + 1] = Q[:, params['N']] # right boundary
        
        # Compute fluxes using FORCE scheme
        F = force(Q, params)

        # Conservative update formula: the discretised
        # integral form of the Euler equations
        Q[:, 1:-1] += dt/params['dx'] * (F[:, :-1] - F[:, 1:])

        # # Move to next time step
        # t += dt

    # plot results
    
    labels = [r'$\rho$', r'$u$', r'$p$']
    fig, axes = plt.subplots(2, 2, sharey='row', sharex='col')
    axes = axes.flatten()
    for ax, q, label in zip(axes[:3], [Q[0], Q[1]/Q[0], pressure(Q[0], Q[1]/Q[0], Q[2], params['GAMMA'])], labels):
        ax.plot(x, q, 'or', fillstyle='none')
        ax.set_ylabel(label)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.1])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    fig.savefig('euler.png')




    # #######################################################

    obs_interval = 10
    tf_target = Q[:, obs_interval:params['N']:obs_interval]

    sess = tf.Session()

    tf_GAMMA_init = 1.3
    tf_GAMMA = tf.Variable(tf_GAMMA_init, tf.float32, name='GAMMA')

    tf_params = {'GAMMA': tf_GAMMA,
                 'N': 100,
                 'dx': 1./100,
                 'c': 0.9,
                 'T': 0.25
    }

    tf_Q_init = np.empty((3, len(x))) 
    
    # density
    tf_Q_init[0, x <= 0.5] = 1.0
    tf_Q_init[0, x > 0.5] = 0.125

    # momentum
    tf_Q_init[1] = 0.0
    
    # energy
    tf_Q_init[2, x <= 0.5] = energy(1.0, 0.0, 1.0, tf_GAMMA_init)
    tf_Q_init[2, x > 0.5] = energy(0.125, 0.0, 0.1, tf_GAMMA_init)
    
    tf_Q = tf.constant(tf_Q_init, tf.float32)
    
    # evolve system

    # t = tf.constant(0, tf.float32)

    for i in range(100):
 
        dt = tf_timestep(tf_Q, tf_params)
        
        # sort boundaries
        left = tf_Q[:, 1]
        right = tf_Q[:, params['N']] # using normal params!

        tf_Q = tf.concat(1, [left[:, None], tf_Q[:, 1:-1], right[:, None]])
        
        # Compute fluxes using FORCE scheme
        F = tf_force(tf_Q, tf_params)

        # Conservative update formula: the discretised
        # integral form of the Euler equations
        
        dQ = dt/tf_params['dx'] * (F[:, :-1] - F[:, 1:])
        
        tf_Q = tf.concat(1, [Q[:, 0][:, None], tf_Q[:, 1:-1] + dQ, Q[:, params['N'] + 1][:, None]])        
        
        # # Move to next time step
        # t = t + dt

    tf_Q_at_obs_points = tf_Q[:, obs_interval:params['N']:obs_interval]
    
    # abs_diff = tf.abs(tf_Q_at_obs_points - tf_target)

    # loss = tf.reuce_sum(abs_diff)
    loss = tf.reduce_mean(tf.square(tf_Q_at_obs_points - tf_target))

    train_op = build_train_graph(loss)
    
    sess.run(tf.initialize_all_variables())
    
    print 'starting training...'
    for i in range(500):
        result = sess.run([train_op, loss, tf_params['GAMMA']])
        if i % 10 == 0:
            print 'loss: {}'.format(result[1])
            print 'gamma: {}'.format(result[2])

if __name__=='__main__':
    main()
    




    
    # def cond(t, tf_Q):
    #     return tf.less(t, tf_params['T'])

    # def body(t, tf_Q):

    #     dt = tf_timestep(tf_Q, tf_params)
        
    #     dt = tf.cond(tf.greater(t+dt, tf_params['T']),
    #                  lambda: dt,
    #                  lambda: tf_params['T'] - t
    #     )

    #     # sort boundaries
    #     left = tf_Q[:, 1]
    #     right = tf_Q[:, params['N']] # using normal params!

    #     tf_Q = tf.concat(1, [left[:, None], tf_Q[:, 1:-1], right[:, None]])
        
    #     # Compute fluxes using FORCE scheme
    #     F = tf_force(tf_Q, tf_params)

    #     # Conservative update formula: the discretised
    #     # integral form of the Euler equations
        
    #     dQ = dt/tf_params['dx'] * (F[:, :-1] - F[:, 1:])

    #     tf_Q = tf.concat(1, [Q[:, 0][:, None], dQ, Q[:, params['N'] + 1][:, None]])        

    #     # Move to next time step
    #     t = t + dt

    #     return t, tf_Q


    # tf.while_loop(cond, body, [t, tf_Q])
