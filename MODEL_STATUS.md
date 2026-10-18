# 牌面视觉模型状态

此分支目前**不是可安装交付版**。它移除了 OCR 牌面推断，并让分析引擎只接受来源标记为“牌面视觉模型”、置信度足够、点数与花色完整且无重复牌的输入。仓库尚无合规的牌面模型，因此当前版本会安全地停止在“等待视觉模型”，不会生成行动建议。

## 候选模型审查

- [koolguy06/playing-cards](https://huggingface.co/koolguy06/playing-cards) 的模型卡声明 MIT，并记录 52 类、约 24,000 张训练图像；它以 YOLOv8l 初始化，权重是 87.7 MB 的 PyTorch `.pt`，模型卡没有提供测试集准确率或逐类指标。
- 对应的 [Augmented Startups Playing Cards v4](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/dataset/4) 页面标为 Public Domain，包含 52 类和 1,010 张测试图。这解决了数据集页面的许可声明，但不能证明该模型对本项目模拟器截图的识别效果。
- Ultralytics 的[官方许可说明](https://www.ultralytics.com/license)明确说 YOLOv8 和由其训练的模型默认采用 AGPL-3.0；非 AGPL 整体分发需要 Enterprise 许可。候选模型卡上的 MIT 标记与其 YOLOv8 来源存在许可冲突风险。仓库没有记录该模型获得单独许可的证据，所以此分支不分发或下载该权重。

- [Daveokw/Card-Identifier](https://github.com/Daveokw/Card-Identifier) 仓库内含 53 类 ResNet18 ONNX 文件，适合单张完整卡牌分类；GitHub 仓库没有声明许可证，README 也说明遮挡和训练集差异可能导致错误，因此不能直接随安装器分发或视作本项目的已验证识别器。
- [MODAJ18/Playing_Cards](https://github.com/MODAJ18/Playing_Cards) 声明 MIT，但依赖 YOLOv5，模型权重需另行下载且不在仓库中；项目还包含 YOLOv5 源码及其 GPL-3.0 许可证。尚无完整权重来源链或本项目截图验证，故不采用。
- [The-Pavels/card-reader](https://github.com/The-Pavels/card-reader) 声明 MIT 并包含 52 类 YOLO11 权重。Ultralytics 对 YOLO11 的许可同样要求核实 AGPL-3.0/Enterprise 分发条件；该仓库没有可核实的例外许可，也没有本项目模拟器截图结果，故不采用。

## 仍需完成的验收

1. 选定并核实适用于该应用分发方式的权重许可及完整来源链。
2. 将兼容 ONNX Runtime 的模型固定版本、校验哈希并随安装器离线打包。
3. 使用用户之前提供的模拟器截图做回放预测，覆盖 9♣/3♣ 与 3♥/2♦/7♣、10♣/3♥、K♣/Q♣，检查置信度、手牌重叠、重复牌与牌局状态。
4. Windows CI 必须先验证模型和回放样例，再打包安装器。
5. 由用户在真实模拟器中完成现场验证；CI 无法替代这一步。

上游对话仍有截图附件条目，但这次任务环境无法读取附件文件本身。已知牌值文本不能代替模型的像素级回放测试。
