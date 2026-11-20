# PatchTST 官方代码

- 仓库 [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)
- 论文 [A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730)
- 主分支提交 [204c21efe0b39603ad6e2ca640ef5896646ab1a9](https://github.com/yuqinie98/PatchTST/commit/204c21efe0b39603ad6e2ca640ef5896646ab1a9)
- 下载源 [GitHub codeload 固定提交 ZIP](https://codeload.github.com/yuqinie98/PatchTST/zip/204c21efe0b39603ad6e2ca640ef5896646ab1a9)
- ZIP 的 SHA‑256 为 `1107abb451cd2a244973409421a6856874b4d8c1b6a13d38c5ccf9349d5dcebf`
- 许可见本目录的 `LICENSE`（Apache-2.0）

模型文件取自该提交的 `PatchTST_supervised/`。模型和骨干文件中的四条 `layers` 导入改成相对导入，可与同样含有 `layers` 目录的其他模型共存。模型数学实现使用上游代码。原始文件与本包文件的哈希如下。

| 文件 | 原始 SHA‑256 | 本包 SHA‑256 |
| --- | --- | --- |
| `models/PatchTST.py` | `49d8bb865e1226d6338842f4b33c4bc6992cefb048dc2a4a5a8c410ca708586b` | `f40d1591297e2ade426f475d5342a6db2e74b592adb5222b6d8cfdad1cd05fb6` |
| `layers/PatchTST_backbone.py` | `df67173153787c2356bdfb6491159cd754332ef7382986efe879e1fbea8ebf26` | `fe3820d9bd6d9f7f591aa1421ec47ac27aa90378e456f91c32c74e036635c2ee` |
| `layers/PatchTST_layers.py` | `21c06c70a90c60ee2a269b5c600c702834dea22cdfd72915e6b0f8b4a28db3f6` | 同左 |
| `layers/RevIN.py` | `e64c0ccded9228b347134e7368420d3fb10f70c75145b5ff2d8bdd8c3af59df6` | 同左 |
| `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | 同左 |

`src/patchtst_adapter.py` 选择三层共享的 PatchTST 编码器，本轮回看 336 个交易日，隐藏宽度 32，四个注意力头，前馈宽度 64。主设置用长度 8、步幅 4 的历史片段，并采用官方末端填充。输入已由训练期数据统一预处理，适配器关闭 RevIN 及分解。官方预测头产生每通道一个值，其后接可训练线性层，将所有通道的表示合成为一个待预测收益率。
