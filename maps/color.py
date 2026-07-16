import tkinter.colorchooser
# 运行后会弹出一个 Mac 系统的原生调色盘，选完颜色关闭，终端就会印出 HEX 代码！
print(tkinter.colorchooser.askcolor(title="给紫金港选个颜色")[1])