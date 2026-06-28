import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------- 辅助函数：构建邻接距离矩阵 --------------------
def G2D(G):
    """
    将栅格地图转换为节点间距离矩阵。
    输入 G：二维数组，0=自由，1=障碍。
    输出 D：N×N 矩阵，D[i,j] 为节点 i 到 j 的欧氏距离（8邻域连通），不可达则为 0。
    """
    l = G.shape[0]          # 地图尺寸 MM
    N = l * l               # 节点总数
    D = np.zeros((N, N))
    # 遍历所有栅格对
    for i in range(l)
        for j in range(l):
            if G[i, j] == 0:          # 起点必须是自由栅格
                idx1 = i * l + j      # 节点编号 (0-indexed)
                for m in range(l):
                    for n in range(l):
                        if G[m, n] == 0:   # 终点也是自由栅格
                            idx2 = m * l + n
                            di = abs(i - m)
                            dj = abs(j - n)
                            # 8邻域条件：正交或对角相邻
                            if di + dj == 1 or (di == 1 and dj == 1):
                                D[idx1, idx2] = np.sqrt(di + dj)   # 正交=1，对角=√2
    return D

# -------------------- 主程序 --------------------
def main():
    # ========== 1. 地图定义 ==========
    G = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # ========== 2. 参数初始化 ==========
    MM = G.shape[0]               # 地图尺寸 (20)
    N = MM * MM                   # 节点总数 (400)
    Tau = np.ones((N, N)) * 8     # 信息素矩阵，初始值 8
    K = 100                       # 迭代次数
    M = 50                        # 蚂蚁个数
    S = 0                         # 起点节点索引 (左上角，0-indexed)
    E = N - 1                     # 终点节点索引 (右下角，399)
    Alpha = 1                     # 信息素重要程度
    Beta = 7                      # 启发式因子重要程度
    Rho = 0.3                     # 信息素蒸发系数
    Q = 1                         # 信息素增加强度系数
    minkl = np.inf                # 全局最短路径长度
    mink = 0                      # 最短路径对应的迭代序号
    minl = 0                      # 最短路径对应的蚂蚁序号

    # ========== 3. 启发式信息计算 ==========
    a = 1  # 栅格边长
    # 终点坐标（笛卡尔坐标系，原点在左下角）
    Ex = a * ((E % MM) + 0.5)                # 横坐标
    Ey = a * (MM - (E // MM) - 0.5)          # 纵坐标
    Eta = np.zeros(N)                         # 启发式信息向量
    for i in range(N):
        ix = a * ((i % MM) + 0.5)
        iy = a * (MM - (i // MM) - 0.5)
        if i != E:
            Eta[i] = 1.0 / np.sqrt((ix - Ex)**2 + (iy - Ey)**2)
        else:
            Eta[i] = 100.0                    # 终点处启发值极高

    # 构建邻接距离矩阵
    D = G2D(G)

    # ========== 4. 存储结构 ==========
    ROUTES = [[None for _ in range(M)] for _ in range(K)]   # 细胞结构
    PL = np.zeros((K, M))                                   # 路径长度记录

    # ========== 5. 蚁群迭代 ==========
    for k in range(K):
        for m in range(M):
            # --- 蚂蚁状态初始化 ---
            W = S                         # 当前节点
            Path = [S]                    # 路径序列
            PLkm = 0.0                    # 路径总长度
            TABUkm = np.ones(N, dtype=bool)  # True 表示未访问
            TABUkm[S] = False             # 起点标记为已访问
            DD = D.copy()                 # 动态邻接矩阵（每只蚂蚁独立修改）

            # --- 更新当前节点的可选下一节点 ---
            DW = DD[W, :].copy()
            DW[TABUkm == False] = 0       # 排除已访问节点
            LJD = np.where(DW > 0)[0]     # 可选节点列表
            Len_LJD = len(LJD)

            # --- 蚂蚁移动（未到终点且未死锁） ---
            while W != E and Len_LJD >= 1:
                # 轮盘赌选择下一节点
                PP = (Tau[W, LJD] ** Alpha) * (Eta[LJD] ** Beta)
                sumpp = np.sum(PP)
                PP = PP / sumpp
                Pcum = np.cumsum(PP)
                rand_val = random.random()
                select_idx = np.searchsorted(Pcum, rand_val)
                to_visit = LJD[select_idx]

                # 状态更新
                Path.append(to_visit)
                PLkm += DD[W, to_visit]
                W = to_visit
                TABUkm[W] = False

                # 将当前节点与所有已访问节点之间的边删除（防止环路）
                visited = np.where(TABUkm == False)[0]
                DD[W, visited] = 0
                DD[visited, W] = 0

                # 重新计算可选节点
                DW = DD[W, :].copy()
                DW[TABUkm == False] = 0
                LJD = np.where(DW > 0)[0]
                Len_LJD = len(LJD)

            # --- 记录本只蚂蚁的结果 ---
            ROUTES[k][m] = Path
            if Path[-1] == E:
                PL[k, m] = PLkm
                if PLkm < minkl:
                    mink = k
                    minl = m
                    minkl = PLkm
            else:
                PL[k, m] = 0

        # --- 信息素更新 ---
        Delta_Tau = np.zeros((N, N))
        for m in range(M):
            if PL[k, m] > 0:                # 只利用成功到达的蚂蚁
                path = ROUTES[k][m]
                pl_km = PL[k, m]
                for s in range(len(path) - 1):
                    x = path[s]
                    y = path[s + 1]
                    Delta_Tau[x, y] += Q / pl_km
                    Delta_Tau[y, x] += Q / pl_km   # 对称更新
        Tau = (1 - Rho) * Tau + Delta_Tau

    # ========== 6. 绘图 ==========
    plotif = 1
    if plotif == 1:
        # 6.1 收敛曲线
        minPL = np.zeros(K)
        for i in range(K):
            plk = PL[i, :]
            non_zero = plk[plk > 0]
            minPL[i] = np.min(non_zero) if len(non_zero) > 0 else np.inf
        plt.figure(1)
        plt.plot(minPL)
        plt.grid(True)
        plt.title('Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Minimum Path Length')

        # 6.2 最优路径轨迹
        plt.figure(2)
        plt.axis([0, MM, 0, MM])
        # 绘制地图栅格
        for i in range(MM):
            for j in range(MM):
                if G[i, j] == 1:   # 障碍物：黑色
                    color = [0.2, 0.2, 0.2]
                else:               # 自由空间：白色
                    color = [1.0, 1.0, 1.0]
                # 矩形左下角坐标 (j, MM-i-1)，宽高为1
                x_rect = j
                y_rect = MM - i - 1
                plt.fill([x_rect, x_rect+1, x_rect+1, x_rect],
                         [y_rect, y_rect, y_rect+1, y_rect+1], color=color)
        # 绘制最优路径
        best_path = ROUTES[mink][minl]
        Rx = np.zeros(len(best_path))
        Ry = np.zeros(len(best_path))
        for ii, node in enumerate(best_path):
            Rx[ii] = a * ((node % MM) + 0.5)
            Ry[ii] = a * (MM - (node // MM) - 0.5)
        plt.plot(Rx, Ry, 'b-', linewidth=2)
        plt.title('Robot Trajectory (Best Path)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

if __name__ == "__main__":
    main()