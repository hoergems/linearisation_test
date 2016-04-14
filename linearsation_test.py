import numpy as np
from sympy import *
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class LinearisationTest:
    def __init__(self):
        x = Matrix([[symbols("x0")],
                    [symbols("x1")]])
        u = Matrix([[symbols("u0")],
                    [symbols("u1")]])
        w = Matrix([[symbols("w0")],
                    [symbols("w1")]])
        self.vars = [x, u, w]
        
        x_nominal = Matrix([[0.0],
                            [2.0]])
        u_nominal = Matrix([[0.5],
                            [1.0]])
        f = self.f(x, u, w)  
        
        mean_noise = np.array([0.0, 0.0])
        cov_noise = np.array([[0.05, 0.0],
                              [0.0, 0.05]])
        
        mean_0 = np.array([0.0, 0.0])
        P_0 = np.array([[0.02, 0.0],
                        [0.0, 0.02]])
        num_samples = 1000
        true_samples = self.propagate(f, mean_0, P_0, u_nominal, mean_noise, cov_noise, num_samples=num_samples)
        mean_1, P_1 = self.predict(f, mean_0, u_nominal, P_0, cov_noise)
        print mean_1
        print P_1
        gauss_samples = self.draw_random_samples(mean_1, P_1, num_samples)
        
        res = []
        res2 = []
        for i in xrange(len(gauss_samples)):
            res.append(true_samples[i])
            res2.append(gauss_samples[i])
        x1_d = np.array([res[i][0] for i in xrange(len(res))])
        y1_d = np.array([res[i][1] for i in xrange(len(res))])
        x2_d = np.array([res2[i][0] for i in xrange(len(res2))])        
        y2_d = np.array([res2[i][1] for i in xrange(len(res2))])
        self.plot_sets([x1_d, y1_d], [x2_d, y2_d])    
        
        '''w_actual = Matrix([[875],
                           [1.0]])
        self.vars = [x, u, w]
        
        
        samples = self.draw_random_samples(mean, cov, 1000)
        
        for s in samples:
            res.append(self.evaluate_f(f1, x_nominal, u_nominal, s))
            res2.append(self.linearise_f(f2, x_nominal, u_nominal, x_nominal, u_nominal, Matrix([[s[0]], [s[1]]])))
        x1_d = np.array([res[i][0] for i in xrange(len(res))])
        y1_d = np.array([res[i][1] for i in xrange(len(res))])
        x2_d = np.array([res2[i][0] for i in xrange(len(res2))])        
        y2_d = np.array([res2[i][1] for i in xrange(len(res2))])
        self.plot_sets([x1_d, y1_d], [x2_d, y2_d])'''
        
    def propagate(self, f, mean_0, P_0, control, mean_noise, P_noise, num_samples=1000):
        samples = self.draw_random_samples(mean_0, P_0, num_samples)
        samples_result = []
        for s in samples:
            w = self.draw_random_samples(mean_noise, P_noise, 1)            
            r = self.evaluate_f(f, s, control, w)
            samples_result.append(np.array([r[0], r[1]]))
        return samples_result
    
    def plot_sets(self, set1, set2):        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)        
        plt.scatter(set1[0], set1[1], c='b')
        plt.scatter(set2[0], set2[1], c='r')
        plt.xlim([-5, 5])
        plt.ylim([-20, 20])
        plt.show()
        
        
    def draw_random_samples(self, mu, cov, num):
        return multivariate_normal.rvs(mu, cov, num)
    
    def predict(self, f, x_nominal, u_nominal, P_t, noise_cov):
        mean = self.evaluate_f(f, x_nominal, u_nominal, [0.0, 0.0])        
        A = f.jacobian([self.vars[0][i] for i in xrange(len(self.vars[0]))])
        A = A.subs(self.vars[0][0], x_nominal[0])
        A = A.subs(self.vars[0][1], x_nominal[1])
        A = A.subs(self.vars[1][0], u_nominal[0])
        A = A.subs(self.vars[1][1], u_nominal[1])
        A = A.subs(self.vars[2][0], 0.0)
        A = A.subs(self.vars[2][1], 0.0)        
        A_a = np.array([[A[i, j] for j in xrange(A.cols)] for i in xrange(A.rows)])
        
        cov = np.dot(np.dot(A_a, P_t), np.transpose(A_a)) + noise_cov       
        
        return np.array([mean[0], mean[1]]), cov
        
        
    def evaluate_f(self, f, x, u, w):        
        f0 = f.subs(self.vars[0][0], x[0])
        f0 = f0.subs(self.vars[0][1], x[1])
        f0 = f0.subs(self.vars[1][0], u[0])
        f0 = f0.subs(self.vars[1][1], u[1])
        f0 = f0.subs(self.vars[2][0], w[0])
        f0 = f0.subs(self.vars[2][1], w[1])
        return f0
        
    def f(self, x, u, w): 
        '''A = Matrix([[2.0, 0.0],
                    [0.0, 2.0]])
        B = Matrix([[0.5, 0.3],
                    [0.0, 2.0]])
        
        f = A * x + B * u + w'''
               
        
        f = Matrix([[cos(x[0]) + 2 * sin(u[0]) + w[0]],
                    [2 * x[1] + 5.0 * sin(x[0]) * u[1] + w[1]]])
        return f
        return simplify(f)
    
    def linearise_f(self, f, x, u, x_nominal, u_nominal, w_actual):
        A = f.jacobian([self.vars[0][i] for i in xrange(len(self.vars[0]))])
        B = f.jacobian([self.vars[1][i] for i in xrange(len(self.vars[0]))])
        C = f.jacobian([self.vars[2][i] for i in xrange(len(self.vars[0]))])
        
        f0 = f.subs(self.vars[0][0], x_nominal[0])
        f0 = f0.subs(self.vars[0][1], x_nominal[1])
        f0 = f0.subs(self.vars[1][0], u_nominal[0])
        f0 = f0.subs(self.vars[1][1], u_nominal[1])
        f0 = f0.subs(self.vars[2][0], 0.0)
        f0 = f0.subs(self.vars[2][1], 0.0)
        
        A = A.subs(self.vars[0][0], x_nominal[0])
        A = A.subs(self.vars[0][1], x_nominal[1])
        A = A.subs(self.vars[1][0], u_nominal[0])
        A = A.subs(self.vars[1][1], u_nominal[1])
        A = A.subs(self.vars[2][0], 0.0)
        A = A.subs(self.vars[2][1], 0.0)
        
        B = B.subs(self.vars[0][0], x_nominal[0])
        B = B.subs(self.vars[0][1], x_nominal[1])
        B = B.subs(self.vars[1][0], u_nominal[0])
        B = B.subs(self.vars[1][1], u_nominal[1])
        B = B.subs(self.vars[2][0], 0.0)
        B = B.subs(self.vars[2][1], 0.0)
        
        C = C.subs(self.vars[0][0], x_nominal[0])
        C = C.subs(self.vars[0][1], x_nominal[1])
        C = C.subs(self.vars[1][0], u_nominal[0])
        C = C.subs(self.vars[1][1], u_nominal[1])
        C = C.subs(self.vars[2][0], 0.0)
        C = C.subs(self.vars[2][1], 0.0) 
        
        
        
        res = f0 #+ A * (x - x_nominal) + B * (u - u_nominal) + C * w_actual
        return res
        
if __name__ == "__main__":
    LinearisationTest()