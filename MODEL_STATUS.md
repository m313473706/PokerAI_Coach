# 牌面视觉识别状态

截至 2026-10-19，本分支已选定并实现**自研、离线的 52 类牌面分类器候选路线**。本分支仍是 Poker AI Coach 1.0 RC 的工作分支，不是可安装交付版。

## 已落地

- 使用固定提交 `ae925833a28f587ec6abed561822889eb044a18c` 的 [safe-acid/Playing-cards-assets](https://github.com/safe-acid/Playing-cards-assets) 52 张 CC0 牌面素材。
- 从随机权重训练自写轻量 CNN；不使用第三方预训练权重或云端识别 API。
- 导出约定为单张牌框输入、52 类概率输出的 ONNX 模型；应用通过本地 CPU ONNX Runtime 加载。
- 对截图中的白色牌框做轮廓定位，再逐张裁切分类。低置信度、类别差距不足、重复牌、错误牌局阶段或帧间不稳定都会阻止行动建议。
- `MODEL_CARD.md` 记录模型结构、数据来源及边界；`Train in-house card classifier` 工作流生成可下载的**候选模型**和训练报告。
- `replay_card_screenshots.py` 将标注的目标模拟器截图送入实际轮廓定位与分类流程，并生成与模型 SHA-256 绑定的回放报告。
- Windows 打包现在要求该回放报告确认覆盖全部 52 类、所有牌值正确、没有定位漏检/误检和低置信度；报告哈希必须对应待打包权重。

## 当前交付状态

修复文件名映射后的训练 workflow 已成功完成：20 个 epoch 的**合成验证最高准确率为 95.19%**。模型 SHA-256 为 `e72bbb52ad473232f80c699159324fbefa7cd90d141865420ddaab9f25fcebaa`；候选 artifact ZIP SHA-256 为 `09765e904acd21baf0b86dc229a7f7c8b2176c4a1d085b0fef238c4e6b5679d9`。可从 [训练运行记录](https://github.com/m313473706/PokerAI_Coach/actions/runs/39223650642) 下载 `pokerai-card-classifier-candidate`。

仓库仍未提交模型权重或真实截图回放报告。该 artifact 只是待评估候选模型，不是安装包。**95.19% 只来自同一套素材的合成变换，不能说明它能识别目标模拟器画面**；Windows 安装器仍会被安全门槛拦下，直到真实截图回放通过。

### 截图回放清单格式

创建 JSON 清单，截图路径相对该 JSON。简化格式只需写每张截图中可见牌值，顺序按牌框定位器的从上到下、同一行从左到右排列：

```json
{
  "screenshots": [
    {
      "image": "table_01.png",
      "cards": ["3H", "2D", "7C", "9C", "3C"]
    },
    {"image": "10C.png", "cards": ["10C"]}
  ]
}
```

也可以用 `{"label":"9C","box":[x1,y1,x2,y2]}` 指定每张牌的标注框。工具会检查定位器是否覆盖标注框，并将定位框送入分类器。回放至少需要 52 张不同牌的目标模拟器实截图，覆盖 52 类；通过条件是牌值全部正确、无定位漏检/误检和低置信度拒绝。

```powershell
python replay_card_screenshots.py --manifest path\to\manifest.json --model models\playing_cards.onnx --report models\real_replay_report.json
```

Windows workflow 会核对回放报告与待打包 ONNX 权重的 SHA-256。
 
## 运行与 API

- 运行时不调用远程视觉 API，不需要按张付费；识牌在本机 CPU 上完成。
- 训练只通过 GitHub Actions 取固定版本 CC0 素材并导出候选权重；是否需要考虑 Actions 额度取决于 GitHub 账号方案。
- 候选权重目前不会自动进入仓库或安装器，避免把未经真实截图验证的模型当成 RC 发布。

## 尚待完成

1. 运行修复后的训练 workflow，检查模型文件、训练报告和 ONNX 元数据。
2. 获取目标扑克模拟器 52 类实截图及标注，运行像素级回放并按错误结果改进定位或训练增强。
3. 在本机目标电脑运行最终模型和完整 UI；通过 Windows 自检、回放门槛及实际桌面验证后，才形成 1.0 RC 候选交付。
