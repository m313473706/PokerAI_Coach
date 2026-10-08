# Poker AI Coach — Windows 窗口读取测试工具 0.1

## 用途

测试第三方扑克模拟器是否允许外部程序直接读取：

- 窗口标题
- Win32 子控件
- 控件文字
- UI Automation 控件名称/类型

**本工具不截图、不 OCR、不修改模拟器。**

## Windows 运行

需要 Python 3.11+：

```bat
pip install -r requirements.txt
python app.py
```

启动后：

1. 先打开扑克模拟器。
2. 点击“刷新窗口”。
3. 从左侧选择扑克模拟器窗口。
4. 点击“测试选中窗口”。
5. 查看右侧检测结果。
6. 同目录会生成 `window_test_result.txt`。

## 如何判断

### 🟢 最理想
发现很多有文字的 Win32/UIA 控件。

说明可以继续研究“直接读取控件”的路线，理论上可以做到不截图、不 OCR。

### 🟡 中间情况
能找到窗口和 UI Automation，但没有牌面/筹码文字。

说明模拟器可能是自绘界面，需要进一步测试窗口内部的数据接口或窗口实时捕获。

### 🔴 最差情况
窗口只有一个大画布，没有可访问控件。

这种情况下直接读取文字基本走不通，但仍然可以做“自动窗口捕获 + 图像识别”，你不需要手动截图。

## GitHub Actions

`.github/workflows/build-windows.yml` 可以在 GitHub Windows runner 上打包 `PokerAI_Window_Read_Test.exe`。

