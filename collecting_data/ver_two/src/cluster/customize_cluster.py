__author__ = "vietth5, datnt527"
import numpy as np
from collections import defaultdict
from math import sqrt
import sys

sys.setrecursionlimit(100000000)

class CustomizeCluster():
    def __init__(self):
        self.adj_mat = np.empty((0, 0))
        self.all_emb = np.empty((0, 512))
        self.to_save = []
    
    def __compute_eps(self, adj_mat):
        """
        Building k-distance graph to compute EPSILON
        this is a faster version of compute_eps function
        """
        # adj_mat = np.matmul(embedding_arr, embedding_arr.transpose())
        n = adj_mat.shape[0]
        MIN_SAMPLES = int(0.1 * n)
        # print('MIN_SAMPLES',MIN_SAMPLES)
        k_distance = 1. - adj_mat
        # matrix = np.sort(k_distance, axis=1) # too slow
        matrix = np.partition(k_distance, MIN_SAMPLES, axis=1)
        matrix_k = matrix[:,:MIN_SAMPLES]

        matrix_k = matrix_k.sum(axis=1)/MIN_SAMPLES
        matrix_k = np.sort(matrix_k)

        x_axis = np.array(range(0,n)).astype(int)
        table = np.vstack((x_axis, matrix_k)).T

        # compute distance from all points to the line start-end
        start_point = np.array([0, matrix_k[0]])
        end_point = np.array([n-1, matrix_k[n-1]])
        b = end_point-start_point
        b_unit = b/(sqrt((b**2).sum()))

        ps = table - start_point
        ds = ps - (b_unit*ps).sum(axis=1).reshape(n,1)*b_unit.reshape(1,2)
        dis = (ds**2).sum(axis=1)
        index = np.argmax(dis)

        # plt.plot(range(n),matrix_k)
        # plt.plot(index, matrix_k[index],'b^')
        # plt.plot([0, n-1], [matrix_k[0], matrix_k[n-1]])
        # plt.show()

        # return matrix_k[index], k_distance
        return 0.35, k_distance

    def __customize_cluster(self, points, ref_indexs, elbow, k_distance):
        """vietth's customized cluster algorithms
        :param points: point to be cluster
        :param ref_indexs: indexes of 3 reference points in 'points'
        :param elbow:
        :param k_distance:
        :type points: numpy array of shape (n, 512)
        :type ref_point: list or numpy array of 3 integer
        :returns: index of point to save in points
        """
        n = points.shape[0]
        MIN_SAMPLES = int(0.01 * n)
        nb = np.zeros(n)          # nb[i]: number of neighbors of i
        mark = np.zeros(n+1)      # mark[i] = 1 if i is marked/visited
        graph = defaultdict(list) # graph[u]: list of neighbors of node u

        # CLUSTERING
        EPSILON = 1 - elbow
        # EPSILON = 0.5
        sim_matrix = 1 - k_distance
        sim_matrix = sim_matrix > EPSILON


        test1=np.ones((sim_matrix.shape[0], sim_matrix.shape[0]))
        test2=(1.-np.tril(test1))*sim_matrix
        neighbor = test2.sum(axis=1) + test2.sum(axis=0)
        idx, idy = np.where(test2==1.)

        for item in zip(idx,idy):
            graph[item[0]].append(item[1])
            graph[item[1]].append(item[0])
        nb = neighbor

        # DFS
        def dfs(i):
            mark[i] = 1
            if (nb[i] < MIN_SAMPLES):
                return
            for j in graph[i]:
                if (mark[j] == 0):
                    mark[j] = 1
                    dfs(j)

        dfs(ref_indexs[0])
        # dfs(ref_indexs[1])
        # dfs(ref_indexs[2])
        num_retain = (mark==1).sum()

        save_indexs = []
        for i in range(n):
            if mark[i] == 1:
                # save
                save_indexs.append(i)
                pass
            else:
                # remove
                pass
        return save_indexs

    def __matmul(self, init_emb, add_emb, cur_matrix):
        """this function used for updating matrix
        :param init_emb: latest embeddings vector (N, 512)
        :param add_emb: embeddings vector to add (M, 512)
        :param cur_matrix:  adjacency matrix of init_emb
        :returns:
            cur_emb: concatenate of (init_emb, add_emb)
            out_matrix: adjacency matrix of cur_emb
        """
        tmp1 = np.dot(init_emb, add_emb.transpose())
        tmp2 = np.dot(add_emb, add_emb.transpose())
        row1 = np.concatenate((cur_matrix, tmp1), axis=1)
        row2 = np.concatenate((tmp1.transpose(),tmp2),axis=1)
        out_matrix = np.concatenate((row1,row2),axis=0)
        cur_emb = np.concatenate((init_emb, add_emb), axis=0)
    
        return cur_emb, out_matrix

    def update(self, addition_points, ref):
        """Add addition_points
        :param addition_points: vector of (N, 512)
        :param ref: index of 3 reference points, eg [1, 2 ,3]
        """
        emb, mat = self.__matmul(self.all_emb, addition_points, self.adj_mat)
        elbow, k_distance = self.__compute_eps(mat)
        to_save = self.__customize_cluster(emb, ref, elbow, k_distance)
        self.adj_mat = mat
        self.all_emb = emb
        self.to_save = to_save

    def get_result(self):
        return self.to_save

if __name__ == '__main__':
    from sklearn.preprocessing import normalize
    customize_cluster = CustomizeCluster()

    for _ in range(10):
        addition_points = normalize(np.random.rand(100, 512))
        customize_cluster.update(addition_points, ref=[0, 1, 2])

    points = customize_cluster.all_emb
    saved = customize_cluster.get_result()
    
    print(points.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(points[:,0], points[:,1])
    plt.scatter(points[saved,0], points[saved,1], c='red')
    plt.show()