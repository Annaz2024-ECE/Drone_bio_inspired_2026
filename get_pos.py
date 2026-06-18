import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# ==========================================
# 校园地图 100x100 坐标交互拾取工具
# 专门为 Member A 提供，用于快速获取建筑物坐标
# ==========================================

# 请确保图片名称与此一致，且与本脚本在同一目录下
# 你可以将刚才下载的海宁校区地图重命名为 campus_map.jpg
IMAGE_PATH = 'haining_map.jpg'

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"错误: 找不到图片文件 '{IMAGE_PATH}'。请确保图片存在且名字正确。")
        return

    # 1. 读取图片
    img = mpimg.imread(IMAGE_PATH)

    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # 3. 将图片映射到 [0, 100] x [0, 100] 的坐标系中
    # extent=[x_min, x_max, y_min, y_max] 强制图片拉伸/缩放到此范围
    ax.imshow(img, extent=[0, 100, 0, 100], aspect='auto')

    # 4. 设置 100x100 的网格
    # 设置主刻度 (每 10 个单位一条粗线)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    # 设置次刻度 (每 1 个单位一条细线)
    ax.set_xticks(np.arange(0, 101, 1), minor=True)
    ax.set_yticks(np.arange(0, 101, 1), minor=True)

    # 绘制网格线
    ax.grid(which='major', color='white', linestyle='-', linewidth=1.2, alpha=0.8)
    ax.grid(which='minor', color='white', linestyle=':', linewidth=0.5, alpha=0.5)

    ax.set_title("ZJU Haining Campus Map Coordinate Picker\n(Click anywhere to get coordinates)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate (0-100)")
    ax.set_ylabel("Y Coordinate (0-100)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # 存储用户点击记录的列表
    clicked_points = []

    # 5. 定义鼠标点击的回调函数
    def onclick(event):
        # 确保点击在坐标轴范围内
        if event.xdata is not None and event.ydata is not None:
            # 提取精确坐标 (浮点数)
            exact_x = event.xdata
            exact_y = event.ydata
            
            # 计算近似到最近的整数坐标
            round_x = int(round(exact_x))
            round_y = int(round(exact_y))
            
            clicked_points.append((round_x, round_y))
            
            # 打印信息到控制台
            print("-" * 40)
            print(f"📍 点击位置近似坐标 (最近整数): X={round_x}, Y={round_y}")
            print(f"🎯 精确坐标为: X={exact_x:.2f}, Y={exact_y:.2f}")
            
            # 给出针对 Member A 的提示建议
            if len(clicked_points) % 2 == 0:
                p1 = clicked_points[-2]
                p2 = clicked_points[-1]
                width = abs(p2[0] - p1[0])
                height = abs(p2[1] - p1[1])
                print(f"💡 [Member A 助手]: 如果这两点是矩形的对角线，您可以这样写代码：")
                print(f"   {{'type': 'rect', 'bottom_left': np.array([{min(p1[0], p2[0])}, {min(p1[1], p2[1])}]), 'width': {width}, 'height': {height}}}")
            
            # 在图上标记点击的位置 (画一个红星)
            ax.plot(exact_x, exact_y, 'r*', markersize=10)
            
            # 标注坐标文本 (改为显示整数坐标)
            ax.text(exact_x + 1, exact_y + 1, f"({round_x}, {round_y})", 
                    color='yellow', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
            
            # 刷新画布显示
            fig.canvas.draw()

    # 绑定鼠标点击事件
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    print("=========================================")
    print("🚀 坐标拾取工具已启动！")
    print("请在弹出的地图窗口中用鼠标点击任意位置。")
    print("点击后，控制台会实时输出对应的 (X, Y) 坐标。")
    print("=========================================")

    plt.show()

if __name__ == '__main__':
    main()