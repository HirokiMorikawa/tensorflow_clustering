import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

def scale(data):
    sigma = np.mean(data, axis=0)
    mu = np.std(data,axis=0)
    return (data - sigma) / mu

# 次元削減
def pressDimention(data, type="PCA"):
    if type == "PCA":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(data)
        data = pca.transform(data)
        return data
    elif type == "tSNE":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)
        data = tsne.fit_transform(data)
        return data
    else:
        return None
        
    
    

# データセットの作成
def create_data(size, dim, x_mean, x_stddiv, y_mean, y_stddiv):
    data = np.random.randn(size, dim)
    for i in data:
        i[0] = i[0] * x_stddiv + x_mean
        i[1] = i[1]  * y_stddiv + y_mean
    return data


# 掛け算
def multiply(x, r):
    mul = np.zeros(x.shape)
    for i, ele in enumerate(zip(x, r)):
        x_ele, r_ele = ele
        mul[i] = x_ele * r_ele
    return mul.astype(np.float64)
        
# 所属クラスタの決定
def findcenter(r, m):
    r = np.zeros(r.shape).T
    #r = r.T
    for i in range(r.shape[0]):
        r[i][m[i]] = 1
    return r.astype(np.float64).T


# ユークリッド2乗距離を求める
def squered_euclidean_distance(x, v):
    return tf.reduce_sum(tf.pow(x - v, 2), 1)

# centroidの計算
def centroid(data, num_clusters):
    # num_cluster分クラス重心を求める
    d = data[np.random.choice(range(data.shape[0]), num_clusters, replace=False)].astype(np.float64)
    return d    

def test(data, num_of_execute=30, num_clusters = 2):
    # num_clusters = 2 # クラスタ数

    np.set_printoptions(formatter={'float': '{: 0.15f}'.format})
    data = scale(data)

    # N * P
    X = tf.placeholder_with_default(data, shape=(data.shape[0], data.shape[1]), name="input") # データセット
    # num_clusters * P
    V = tf.Variable(centroid(data, num_clusters), dtype=tf.float64, name="centers") # クラスタ中心
    # num_clusters * N
    R = tf.Variable(np.zeros([num_clusters, data.shape[0]]), dtype=tf.float64, name="cluster")

    
    # 2乗距離 num_clusters * N
    D = tf.map_fn(lambda v: squered_euclidean_distance(X, v), V, name="distance")
    # 転置 N * num_clusters
    D = tf.transpose(D)

    # 行ごとに最小値を選んでそのindexを取得 N * 1
    mini = tf.argmin(D, 1)
    
    # データごとに所属クラスタ割り当て num_clusters * N
    select_cluster = tf.py_func(findcenter, [R, mini], tf.float64, name="select_c")
    # num_clusters * N
    update_r = tf.assign(R, select_cluster, name="update_r")

    # クラスタ中心更新
    center_fun = lambda r: tf.div(
        tf.reduce_sum(tf.py_func(multiply, [X, r], tf.float64), 0),
        tf.reduce_sum(r, 0))
    # 新たなVの計算 num_clusters * P
    calc_v = tf.map_fn(center_fun, R, name="calc_v")
    update_v = tf.assign(V, calc_v, name="update_v")

    # Rのリセット
    reset_R = tf.assign(R, np.zeros([num_clusters, data.shape[0]]))

    initializer = tf.global_variables_initializer()
    
    with tf.Session() as s:
        s.run(initializer)
        print("更新前のクラスタ重心")
        print()
        result = s.run(V)
        print(result)
        print()
        print("クラスタ重心を更新します")
        for num in range(num_of_execute):
            print("{}回目".format(num))
            # s.run(D)
            # s.run(mini)
            # s.run(select_cluster)
            
            l_r = s.run(update_r)
            print("cluster : ", l_r)
            # s.run(calc_v)
            print("center : ", s.run(update_v))
            if num < num_of_execute - 1:
                s.run(reset_R)
        print()
        print("更新後のクラスタ重心")
        print()
        inputData = s.run(X)
        result_v = s.run(V)
        result_r = s.run(R)
        result_select_cluster = s.run(select_cluster)
        print("cluster : ", result_r)
        print("center : ", result_v)

        
        
        # 2次元
        inputData = pressDimention(inputData, type="PCA")
        result_v = pressDimention(result_v, type="PCA")

        # col = np.linspace(0, 1, num_clusters)

        for c in range(num_clusters):
            col = cm.hsv(float(c) / 10)
            plt.scatter(result_v[c][0], result_v[c][1], c=col)
            for i in range(data.shape[0]):
                if result_select_cluster[c][i] == 1:
                    plt.scatter(inputData[i][0], inputData[i][1], s=4, c=col)
        

        # tf.summary.FileWriter('./logs', s.graph)


        


if __name__ == "__main__":
    
     # sample_data = np.random.randn(120, 24).astype(np.float32) 
    # sample_data = np.concatenate([sample_data, create_data(200, 24, 2, 0.5, 2, 1)], axis=0).astype(np.float32)
    sample_data = create_data(5000, 24, 2, 0.5, 2, 1)
    sample_data = np.concatenate([sample_data, create_data(600, 24, -2, 0.5, 0, 0.5)],axis=0).astype(np.float64)
    test(sample_data, num_clusters=2)
    plt.show()
    
        
    
