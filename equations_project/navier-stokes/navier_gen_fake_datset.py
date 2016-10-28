import tensorflow as tf
import numpy as np
from navier import build_train_graph, get_grad, get_v_dot_grad_v

def main():

    N = 20 # no. cells

    v_init = np.random.rand(2, N, N).astype(np.float32)

    v = tf.Variable(v_init, name='v')

    f = np.zeros((2, N, N)).astype(np.float32)
    for i in range(N):
        for j in range(N):
            f[0,i,j] = -j
            f[1,i,j] = i

    g = get_grad(N).astype(np.float32)

    v_dot_grad_v = get_v_dot_grad_v(v, g, N)

    loss = tf.reduce_mean(tf.square(v_dot_grad_v - f))
    train_op_v = build_train_graph(loss, [v])

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    print 'starting training...'

    for i in range(10000):
        _, cost = sess.run([train_op_v, loss])
        if i % 10 == 0:
            print 'loss on iteration {}: {}'.format(i, cost)

    with open('navier_fake_dataset.npz', 'wb') as out:
        np.savez(out, v=sess.run(v), f=f)

if __name__=='__main__':
    main()
    


    








    
