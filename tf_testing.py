import tensorflow as tf
import numpy as np

def main():

    N = 5
    
    g = tf.constant(np.eye(N))

    v = np.random.rand(2, N, N)
    vt = tf.constant(np.random.rand(2, N, N))

    x = [i for i in range(0, 5, 2)]
    y = [i for i in range(0,5,2)]

    xx, yy = np.meshgrid(x,y)

    coords = [xx.ravel(), yy.ravel()]

    indexed = v[:, coords[0].tolist(), coords[1].tolist()]

    print indexed
    
    indexedt = v[:, coords[0], coords[1]]
    
    sess = tf.Session()

    sess.run(tf.initialize_all_variables())

    print sess.run(indexed)

    
if __name__=='__main__':
    main()
