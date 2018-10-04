import tensorflow as tf
import numpy as np

# データセットの作成
def create_data(size, dim, x_mean, x_stddiv, y_mean, y_stddiv):
    data = np.random.randn(size, dim)
    for i in data:
        i[0] = i[0] * x_stddiv + x_mean
        i[1] = i[1]  * y_stddiv + y_mean
    return data

# メンバーシップ確率の計算
def initMembership(n_cluster, size):
    m = np.random.random([size, n_cluster])
    sum_row = np.sum(m, axis=1)
    for i in range(size):
        m[i] = m[i] / sum_row[i]
    return m.astype(np.float64)

def updateMembership(X, V, m=2):
    # 距離を求める関数
    dist = lambda a, b: np.power(a - b, 2)
    w = np.zeros([X.shape[0], V.shape[0]])
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            numerator = dist(X[i], V[j])
            w[i][j] = 1 / np.sum([ np.power(numerator / dist(X[i], V[p]), 2 / m - 1) for p in range(w.shape[1])])
    return w.astype(np.float64)
            
        

def centroid(X, w, m=2):
    # 分母
    # denominator = np.zeros([w.shape[1], X.shape[1]])
    # 分子
    # numerator = np.zeros([w.shape[1], X.shape[1]])
    # 更新値
    mu = np.zeros([w.shape[1], X.shape[1]])

    # denominator = np.sum([np.power(w[i], m) * X[i] for for i in range(X.shape[0])], axis=0)
    # numerator = np.sum([np.power(w[i], m) for i in range(X.shape[0])], axis=0)

    for j in range(w.shape[1]):
        numerator = np.sum([np.power(w[i][j], m) * X[i] for i in range(w.shape[1])], axis=0)
        denominator = np.sum([np.power(w[i][j], m) for i in range(w.shape[1])], axis=0)
        mu[j] = numerator / denominator
    
    # mu = np.zeros(w.shape)
    # for j in range(mu.shape[1]):
    #     mu[:, j] = denominator[j] / numerator
    return mu.astype(np.float64)
    

def test(data, max_ittr=30, n_cluster=2, m=2):
    X = tf.placeholder_with_default(data, shape=data.shape, name="input")
    # 各データごとのメンバーシップ確率
    W = tf.Variable(initMembership(n_cluster, data.shape[0]), dtype=tf.float64, name="member_statistics")
    # クラスタ中心
    V = tf.Variable(tf.zeros([n_cluster, X.shape[1]], dtype=tf.float64), dtype=tf.float64, name="center")
    
    # クラスタの更新
    update_V = tf.assign(V, tf.py_func(lambda x, w: centroid(x, w, m), [X, W], tf.float64, name="initCentroid"))

    # メンバーシップ確率の更新
    update_member = tf.py_func(lambda x, v: updateMembership(x, v, m), [X, V], tf.float64, name="updateMember")
    update_member = tf.assign(W, update_member)
    

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        print("初期のメンバーシップ確率")
        print("")
        print(session.run(W))
        print("")
        print("初期のセントロイド")
        print("")
        print(session.run(update_V))
        print("")
        print("-------------------------------更新フェイズ開始-------------------------------------")

        for i in range(max_ittr):
            print("{}回目の更新".format(i))
            print("{}回目の更新されたメンバーシップ確率".format(i))
            print("")
            print(session.run(update_member))
            print("")
            print("{}回目の更新されたセントロイド".format(i))
            print("")
            print(session.run(update_V))
            print("")
            
        print("-------------------------------更新フェイズ終了-------------------------------------")
        print("更新後のメンバーシップ確率")
        print("")
        print(session.run(W))
        print("")
        print("更新後のセントロイド")
        print("")
        print(session.run(V))
        print("")

        tf.summary.FileWriter('./logs', session.graph)


if __name__ == "__main__":
    sample_data = create_data(5000, 2, 2, 0.5, 2, 1)
    sample_data = np.concatenate([sample_data, create_data(800, 2, -2, 0.5, 0, 0.5)],axis=0).astype(np.float64)
    test(sample_data, n_cluster=2)
    
    
