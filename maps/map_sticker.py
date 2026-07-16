import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

# 1. 加载你的紫金港地图图片
img = mpimg.imread('maps/zijingang_map.jpg')
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img, extent=[0, 220, 0, 160]) # 确保 extent 与 JSON bounds 一致

print("==================================================")
print(" 🗺️ 紫金港地图连续取点器已启动！")
print(" 1. 鼠标左键：点击建筑的各个顶点")
print(" 2. 鼠标右键 (或中键)：完成当前建筑，并在终端输出代码")
print(" 3. 退出程序：直接关闭图片窗口，或在没点任何点时按右键")
print("==================================================\n")

polygon_count = 1

# 只要图片窗口没被关闭，就一直循环运行
while plt.fignum_exists(fig.number):
    
    # n=-1 表示不限制点击次数，直到用户按下右键为止
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    
    # 如果用户没取任何点直接按了右键，或者强行关闭了窗口，则退出循环
    if not points:
        print("\n✅ 检测到退出指令，取点器已安全关闭。")
        break
        
    # 强制转换为纯 Python float 并保留 1 位小数
    json_points = [[round(float(x), 1), round(float(y), 1)] for x, y in points]
    
    # 组装 JSON 字符串
    output_str = f"""    
    {{"type": "polygon", 
      "points": {json_points}, 
      "z_min": 0.0, "z_max": 6.0, 
      "color": "#717579"}},"""
    
    print(f"\n🎯 [第 {polygon_count} 个建筑] 取点成功！请复制：")
    print("-" * 50)
    print(output_str)
    print("-" * 50)
    
    # ==========================================
    # 🔥 核心体验升级：在画布上立刻把画好的建筑涂成半透明红色
    # ==========================================
    poly_patch = patches.Polygon(points, closed=True, fill=True, color='#e53935', alpha=0.4, ec='black')
    ax.add_patch(poly_patch)
    plt.draw() # 刷新画布，显示刚才画的多边形
    
    polygon_count += 1