# Poker AI Coach 自研牌面分类器候选

## 模型概述

- **任务：** 对已定位并裁切的一张扑克牌输出 52 类点数与花色概率。
- **结构：** 自写轻量卷积网络，随机初始化训练；未使用第三方权重、Ultralytics 训练代码或识别 API。
- **输入：** RGB 单张牌框，缩放到 128×192；ONNX 输入张量为 `[1, 3, 192, 128]`。
- **输出：** `[1, 52]` 概率，类别顺序为 2C、2D、2H、2S……AC、AD、AH、AS。
- **推理：** 应用使用 ONNX Runtime CPU provider，在本地运行。
- **权重许可：** CC0-1.0。训练报告会记录模型 SHA-256 和来源元数据。

## 训练来源

52 张牌面素材来自 [safe-acid/Playing-cards-assets](https://github.com/safe-acid/Playing-cards-assets/tree/ae925833a28f587ec6abed561822889eb044a18c) 的固定提交 `ae925833a28f587ec6abed561822889eb044a18c`；仓库声明 CC0-1.0。工作流只取牌面图像与许可文件。训练从随机权重开始，使用程序生成缩放、旋转、亮度、对比度、颜色和轻微模糊变化。

## 验证边界

训练报告里的合成验证集由同一组 52 张素材变换生成。它只能检查训练流程和素材内分类效果，**不能证明**模型能识别用户的扑克模拟器截图。当前没有真实目标截图回放结果，也没有可发布的已验证权重。

## 目标模拟器截图回放

`replay_card_screenshots.py` 会运行当前牌框轮廓定位器，再对定位出的牌框分类。输入 manifest 采用 JSON；截图路径相对 manifest，每张牌的 box 是完整牌框坐标：

```json
{
  "screenshots": [
    {
      "image": "table_01.png",
      "cards": [
        {"label": "9C", "box": [100, 200, 180, 320]},
        {"label": "3H", "box": [250, 100, 330, 220]}
      ]
    }
  ]
}
```

执行方式：

```powershell
python replay_card_screenshots.py --manifest path\to\manifest.json --model models\playing_cards.onnx --report models\real_replay_report.json
```

合格报告需包含至少 52 张标注牌、覆盖 52 类，并且所有预测正确，定位器没有漏牌或多检，分类器没有低置信度拒绝。报告包含模型 SHA-256；Windows 打包会核对该值并拒绝不匹配或未通过的报告。

GitHub Actions 的 `Train in-house card classifier` 工作流生成名为 `pokerai-card-classifier-candidate` 的候选模型 artifact，保留 90 天。该 artifact 不是安装包，也不会自动被打包进发行版。
