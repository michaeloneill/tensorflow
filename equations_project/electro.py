import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy.linalg as nla


nx = 100
dx = 1./(nx-1)
femat = 1./(dx*dx) * (np.diag(np.ones(nx-3),1)+np.diag(np.ones(nx-3),-1)-2*np.eye(nx-2))
print femat, femat.shape

plt.rcParams["figure.figsize"] = (3,2)
scale_factor = 10

x = np.arange(0,1,1./nx)
ground_truth_RHS = 1*np.sin(2*np.pi*x-1)*np.exp(-2*x)
plt.plot(x, ground_truth_RHS,"r")
plt.show()
ground_truth = np.concatenate([[0], nla.solve(femat, ground_truth_RHS[1:-1]), [0]])
plt.plot(x, ground_truth,"r")

observations = ground_truth[scale_factor:nx:scale_factor]
obs_x = x[scale_factor:nx:scale_factor]
plt.plot(obs_x, observations,"o")
plt.show()



sess = tf.Session()

tf_femat = tf.constant(femat)
#f = np.reshape(ground_truth_RHS[1:-1], [-1,1])
f = 0.01 * np.random.randn(nx-2,1)
tf_f = tf.Variable(f)
tf_target = tf.constant(observations)

#sess.run(tf.initialize_all_variables())
tf_solve = tf.matrix_solve(tf_femat, tf_f)

solution_at_obs_points = tf_solve[(scale_factor-1)::scale_factor]
abs_diff = tf.abs(tf.transpose(solution_at_obs_points) - tf.reshape(tf.transpose(tf_target), [1,-1]))

loss = tf.reduce_mean(abs_diff)+ #0.01*tf.reduce_max(abs(tf_f[1:]-tf_f[:-1]))
train_op = tf.train.AdamOptimizer(0.01, 0.9).minimize(loss)
sess.run(tf.initialize_all_variables())

for i in range(500):
    result = sess.run([train_op, loss])
    if i % 100 == 0:
        print result[1]
        plt.plot(range(scale_factor,nx,scale_factor),observations,"r")
        plt.plot(range(0,nx,1),np.concatenate([[[0]],sess.run(tf_solve),[[0]]]), 'b')
        plt.plot(range(scale_factor,nx,scale_factor),sess.run(solution_at_obs_points), 'bo')
        plt.show()


plt.plot(sess.run(tf_f), 'r')
plt.plot(ground_truth_RHS, 'b')
plt.show()
