# Poker AI Coach 自研牌面分类器候选

当前工作流从随机初始化训练轻量 52 类卷积分类器，不使用第三方模型权重、Ultralytics 训练代码或识别 API。轮廓算法先定位白色牌框，ONNX 分类器对单张牌框输出 52 类概率；应用运行时只使用本地 CPU ONNX Runtime。

- 输入：RGB 单张牌框，缩放到 128×192。
- 输出类别：按 2C、2D、2H、2S……AC、AD、AH、AS 排列的 52 类。
- 训练素材：CC0 的 52 张标准牌图，固定来源提交 `ae925833a28f587ec6abed561822889eb044a18c`。
- 权重许可：CC0-1.0。训练导出时写入 ONNX 元数据，并报告 SHA-256。
- 训练报告中的验证数据由同一牌面素材经合成变换生成；它不能代表用户扑克模拟器截图的识别准确率。
- 目标模拟器截图回放尚未完成；在回放通过前，不将候选模型描述为 1.0 RC 完成版，也不打包最终安装器。

用 GitHub Actions 的 `Train in-house card classifier` 工作流可复现训练，并下载 `pokerai-card-classifier-candidate` 产物。应用仍要求 `models/playing_cards.onnx` 存在且通过元数据契约。
