import numpy as np

# ------------------------------- Reference --------------------------------
# S. Liu, H. Wang, W. Yao and W. Peng, "Surrogate-Assisted Environmental Selection for Fast Hypervolume-based Many-Objective Optimization,"
# in IEEE Transactions on Evolutionary Computation, doi: 10.1109/TEVC.2023.3243632.
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2022 HandingWangXD Group. Permission is granted to copy and
# use this code for research, noncommercial purposes, provided this
# copyright notice is retained and the origin of the code is cited. The
# code is provided "as is" and without any warranties, express or implied.
# ------------------------------- Developer --------------------------------
# This code is written by Shulei Liu. Email: shuleiliu@126.com

class RBFN(object):

    def __init__(self, input_shape, hidden_shape, kernel='gau'):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        # self.hidden_shape = int(np.sqrt(input_shape))

        self.sigma = None
        self.centers = None
        self.weights = None
        self.bias = None

        def Gaussianfun(center, data_point):#高斯函数
            return np.exp(-0.5 * np.power(np.linalg.norm(center - data_point, ord=2) / self.sigma, 2))
            # return np.exp(-0.5 * np.power(self.Hamming_distance(center, data_point) / self.sigma, 2))
        def Reflectedfun(center, data_point):#反演S型函数
            return 1/(1+ np.exp(np.power(np.linalg.norm(center - data_point) / self.sigma, 2)))
        def Multiquadric(center, data_point):#多二次函数
            return np.sqrt(np.power(np.linalg.norm(center - data_point), 2) + np.power(self.sigma, 2))
        def INMultiquadric(center, data_point):#逆多二次函数
            return 1/np.sqrt(np.power(np.linalg.norm(center - data_point), 2) + np.power(self.sigma, 2))

        if kernel == 'gau':
            self.kernel_ = Gaussianfun
        elif kernel == 'reflect':
            self.kernel_ = Reflectedfun
        elif kernel == 'mul':
            self.kernel_ = Multiquadric
        elif kernel == 'inmul':
            self.kernel_ = INMultiquadric

    def _calculate_interpolation_matrix(self,X):
        X = np.array(X)
        G = np.zeros((X.shape[0], self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg,center_arg] = self.kernel_(center, data_point)
        return G

    def calsigma(self):
        max = 0.0
        # num = 0
        total = 0.0
        for i in range(self.hidden_shape):
            dis = np.linalg.norm(self.centers - self.centers[i], ord=2, axis=1)
            sum_dis = np.sum(dis)
            # total = total + sum_dis
            if sum_dis > max:
                max = sum_dis
            # for j in range(i+1,self.hidden_shape):
            #     # 欧式距离
            #     dis = np.linalg.norm(self.centers[i] - self.centers[j], ord=1)
            #     total = total + dis
            #     num += 1
            #     if dis >max:
            #         max = dis
        # k1
        self.sigma = max/np.sqrt(self.hidden_shape*2)
        if self.sigma == 0:
            self.sigma = 0.01
        # self.sigma = 2*total/(num+0.001)
        # self.sigma = max
        # self.sigma = total/self.hidden_shape
        # self.sigma = total/np.sqrt(self.hidden_shape)
        # print(max, self.sigma)
        # print(self.sigma)


    def fit(self,X,Y):
        """ Fits self.weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        X = np.array(X)
        Y = np.array(Y)
        self.centers = X
        self.calsigma()
        G = self._calculate_interpolation_matrix(X)
        temp = np.ones((len(X)))
        temp = np.column_stack((G, temp))
        temp = np.dot(np.linalg.pinv(temp), Y)
        self.weights = temp[:self.hidden_shape]
        self.bias = temp[self.hidden_shape]

    def predict(self,X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        X = np.array(X)
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights) + self.bias
        return predictions