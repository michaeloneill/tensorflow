import tensorflow as tf
import numpy as np
import os

def build_train_graph(loss, tvars=None, masks=None):

    with tf.name_scope('train'):
        if tvars is None:
            tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)

        optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
        if masks is not None:
            assert len(masks)==len(tvars)
            grads = [grad*mask for grad, mask in zip(grads, masks)]
        train_step = optimizer.apply_gradients(zip(grads, tvars))

        return train_step


def get_grad(N):

    delta = 1./N
    
    g = -np.diag(np.ones(N-1), -1) + np.diag(np.ones(N-1), 1)
    g[0,0] = -2
    g[0,1] = 2
    g[N-1,N-1] = 2
    g[N-1, N-2] = -2
    g /= 2*delta

    return g


def get_v_dot_grad_v(v, g, N):

    ''' v is 2 x N x N '''

    v_x = v[0]
    v_y = v[1]
    
    grad_x_vx = tf.pack([tf.squeeze(tf.matmul(g, v_x[i][:, None])) for i in range(N)])
    grad_y_vx = tf.pack([tf.squeeze(tf.matmul(g, v_x[:, i][:, None])) for i in range(N)], axis=1)
    grad_x_vy = tf.pack([tf.squeeze(tf.matmul(g, v_y[i][:, None])) for i in range(N)])
    grad_y_vy = tf.pack([tf.squeeze(tf.matmul(g, v_y[:, i][:, None])) for i in range(N)], axis=1)


    return tf.pack([v_x*grad_x_vx + v_y*grad_y_vy, v_x*grad_x_vy + v_y*grad_y_vy])



def main():

    results_dir = input('Enter results directory: ')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    N = 20 # no. cells

    data = np.load('navier_fake_dataset.npz')
    v_ground_truth = data['v']
    # f_ground_truth = data['f']

    stride = 2
    v_obs = v_ground_truth[:, stride:N:stride, stride:N:stride]

    # mark positions of observations
    mask = tf.placeholder(tf.float32, shape=(2, N, N))

    mask_obs = np.zeros((2, N, N))
    mask_obs[:, stride:N:stride, stride:N:stride] = 1

    v_init = np.random.rand(2, N, N).astype(np.float32)
    v_init[:, stride:N:stride, stride:N:stride] = v_obs
    v = tf.Variable(v_init, name='v')
     
    f_init = np.random.rand(2, N, N).astype(np.float32)
    f = tf.Variable(f_init, name='f')
    # f = f_ground_truth
    
    g = get_grad(N).astype(np.float32)

    v_dot_grad_v = get_v_dot_grad_v(v, g, N)

    loss = tf.reduce_mean(tf.square(v_dot_grad_v - f))
    tf.scalar_summary('loss', loss)
    train_op_v = build_train_graph(loss, [v], [mask])
    train_op_f = build_train_graph(loss, [f])

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(results_dir+'logs/train', sess.graph)
    
    print 'starting training...'

    for i in range(1000):
        _, cost_v, summary_v = sess.run([train_op_v, loss, merged], feed_dict={mask: 1-mask_obs})
        _, cost_f, summary_f = sess.run([train_op_f, loss, merged])
        if i % 10 == 0:
            print 'loss_v on iteration {}: {}'.format(i, cost_v)
            print 'loss_f on iteration {}: {}'.format(i, cost_f)

        train_writer.add_summary(summary_v, i)
        train_writer.add_summary(summary_f, i)

    
if __name__=='__main__':

    main()
    


    # # define coordinates of obsA and obsB points
    
    # obsA_x = [i for i in range(1,N,2)]
    # obsA_y = [i for i in range(1,N,2)]
    # xxA, yyA = np.meshgrid(obsA_x, obsA_y)

    # obsB_x = [i for i in range(0, N, 2)]
    # obsB_y = [i for i in range(0, N, 2)]
    # xxB, yyB = np.meshgrid(obsB_x, obsB_y)
    
    # obsA_coords = [xxA.ravel(), yyA.ravel()]
    # obsB_coords = [xxB.ravel(), yyB.ravel()]

    # # set values at observation points

    # v_init[:, obsA_coords[0], obsA_coords[1]] = 1
    # v_init[:, obsB_coords[0], obsB_coords[1]] = 2


    
    # # create mask to mark positions of observations

    # mask = tf.placeholder(tf.float32, shape=(2, N, N))
    
    # maskA = np.zeros((2, N, N))
    # maskA[:, obsA_coords[0], obsA_coords[1]]=1
    # maskB = np.zeros((2, N, N))
    # maskB[:, obsB_coords[0], obsB_coords[1]]=1


    
