# Session: Training-Free On-Policy Distillation (2610.22236)

- 文章：https://arxiv.org/abs/2610.22236（Training-Free On-Policy Distillation, Yang et al., 2026-10-15）
- 日期：2026-10-17｜Track：training｜Verdict：**SUPPORTED**

## 為什麼選這篇（repo fit）

TUNE 高度相關：把每天在 Colab T4 上用 NF4-QLoRA 練出來的 domain specialists
（`soup-daily-finetune` / `hermes-gi-egtkg-finetune` 的 daily-LoRA 模式）
融合成一個 generalist，**不需要任何梯度步驟**——只有 forward pass
收集各層輸入的二階矩，再做一次閉式最小二乘求解。若成立，
每天的 specialist 產出可以直接融合進主模型，省掉多 domain 重訓。

## 測了什麼（核心主張）

1. **精確的層級解**：共享 base 的 specialists，其融合權重是最小二乘問題
   `W* = argmin Σ_d w_d E||Wx − W_d x||²` 的閉式解（Eq. 2），只需要各層
   輸入的 Gram 矩陣 `G_d = XᵀX/n`（一次 forward pass 收集）。
2. **Rollout 內容比求解器重要**（§3.4）：卡在 repetition loop 被截斷的答案
   佔答案數很少、佔 token 很多，且其二階矩集中在少數方向；濾掉它們能
   提升融合模型。

## 做了什麼

- `src/arxiv_lab/training/tf_opd.py`（已畢業到 `src/` 的可重用模組）：
  `accumulate_gram`（§3.2）、`shrink_gram`（Eq. 3, α=0.9）、
  `fuse_linear_layer`（Eq. 2, float64 求解，pseudo-inverse fallback 對應
  Prop. 3.1(iii)）、`unreproduced_share`（Eq. 4 診斷）、
  `weighted_objective`（Eq. 1）、`detect_repetition_loop`（§3.4,
  DEFLATE 壓縮比 < 5% / 20000 字元）、`fuse_layer_dict`
  （bias 按 task arithmetic 合併，§3.3）。
- `src/arxiv_lab/training/test_tf_opd.py`：確定性合成測試（seeded RNG）。
  兩層 MLP（8→16→1，tanh），先在混合任務上訓出共享 base（模擬 pretrain），
  再各用 80 步小 lr 訓出 specialist A（二次函數，dims 0–1）與 B
 （正弦函數，dims 4–5）——小權重移動對應論文 RL specialist 的 regime。
  每個 specialist 讀自己生成的 600×24 條 rollout 行收集 Gram 矩陣。
- `tf_opd_colab_demo.py`：Colab 待跑腳本（已 stage，未執行），見下方。

## 結果（12/12 checks 通過）

- **閉式解精確性**：normal-equation 相對殘差 W1=2.9e-16、W2=7.4e-16
 （solver 精度）；融合權重在蒸餾目標 Eq. 1 上嚴格優於 base 與
  task arithmetic（W1: 0.00046 vs 0.01077 vs 0.01089）。
- **正交輸入範圍的精確重現**（Prop. 3.1(iii)）：fused 層在各自 specialist
  自己的輸入上輸出殘差 4.5e-16，Eq. 4 未重現份額 ε≈1.9e-31——精確成立。
- **端到端技能保留**：fused 在 A 任務 MSE 0.1724（specialist 0.1532，
  1.13×）、B 任務 0.5394（specialist 0.5257，1.03×）；平均
  (0.3559) 優於任一 specialist 單獨的平均（0.3728 / 0.3594）——與論文
  「平均比任一 specialist 好」的 headline 一致。
- **Loop filter**：repetition 文本壓縮比 0.48% 被標記，正常推理文本
  22.1% 通過；loop rollout 的 Gram 最大特徵值佔比 0.99997（集中在
  少數方向），佔 14.3% token 來自僅 15 個答案。
- **Ablation（主張 2 方向性）**：混入 loop rollout 的融合 vs 濾掉的融合，
  乾淨分佈上的 Eq. 1 目標 J 從 0.000314 惡化到 0.000388（+23%），
  端到端 A 任務 MSE 從 0.17242 惡化到 0.17263——方向與論文一致
  （degraded inputs 浪費最小二乘的容量）。

## Verdict：SUPPORTED

數學核心在模擬中被精確驗證（閉式解、正交精確性、loop 機制方向性），
但**論文的真實權重結果（Olmo-3 7B 七個 benchmark 的保留率）沒有重現**；
Colab 真實模型 demo 已寫好但未執行。不是 VERIFIED。

## 誠實的限制

- 合成 MLP 測試，非真實 LLM 權重；specialist 是短步數 SGD 微調，
  非 RL 訓練。
- 端到端保留是近似的（層級精確解經 tanh 複合後有誤差）：本 toy 中
  fused 為 specialist MSE 的 1.03–1.13×。
- Loop ablation 的端到端效應量小（MSE +0.0002），機制層面（J 目標
  +23%）較明顯；論文在 AIME25 上的大幅提升未在此重現。
- 真實模型驗證（Colab 腳本）尚未跑——這是 SUPPORTED 而非 VERIFIED
  的主因。

## Colab 續跑狀態（resume state）

- 腳本：`sessions/2026-10-17-2610.22236-tf-opd/tf_opd_colab_demo.py`
  （self-contained，頂部 pip installs 含 `!pip uninstall -y torchao`）。
- 待做：在 Colab T4 上執行（`hermes-colab-cli` 或貼進 Colab cell）。
  腳本會下載 `Qwen/Qwen2.5-0.5B-Instruct`（公開模型，無需 HF token），
  構造兩個**合成**權重擾動 specialist（誠實標註：非真實訓練），收集
  64×2 條 toy rollout 的 Gram 矩陣，做 Eq. 2 融合，印出每層 Eq. 4
  診斷與 base/specialists/fused 在 held-out toy prompt 上的 perplexity。
- 注意：specialists 是合成擾動，此 demo 驗證的是**融合機制在真實
  transformer 權重形狀上的可跑通性**，不是論文的 benchmark 主張。
- 腳本內無任何 credential/token，金鑰掃描通過（僅 "token" 字樣指
  text tokens / tokenizer）。

## Colab confirmation attempt (2026-10-18)

**未完成（session reclaimed）。** 第一次執行在啟動 3 分鐘後崩潰：
`GramCollector` 在 CPU 上創建 float64 累加器，模型跑在 CUDA —
已修（accumulator 用 `mod.parameters()` 的 device；
`fuse_state_dicts` 與 Eq. 4 診斷改走 CPU float64）。
修後重跑正常推進：math specialist 64/64 rollout 收集完畢，
在 code specialist 的 Gram 收集階段（約第 36 分鐘）被 free tier
回收，`tf_opd_colab_results.log` 未產生。兩個 specialist
各 64 條 greedy rollout（≤96 tokens）在 0.5B 模型上約需 35–40 分鐘，
比預期慢 —— 下次跑建議先縮小 `N_ROLLOUTS` 或縮短
`MAX_NEW_TOKENS`。
結論：Colab 真實權重形狀上的可跑通性仍未確認；本 demo 仍待一次
完整的 session。腳本已是修好的版本，可直接重用。

## Colab retry（2026-10-20）

**FAILED（CUDA OOM）。** Peter 批准三軌同跑：帳號 kyu009009，T4，
`TFOPD_ROLLOUTS=32`（縮半，預估 ~20 分鐘）。兩組 rollout 生成都成功
（math / code specialist 各 kept=32，dropped_loop=0），但在 code
specialist 的 Gram 收集階段崩潰：`collect_grams`
（`tf_opd_colab_demo.py:135`）`torch.OutOfMemoryError`——PyTorch 已佔
14.0 GiB，只剩 99.81 MiB free，再申請 182 MiB 失敗。結果未印出，
`tf_opd_colab_results.log` 為崩潰堆疊（已收回本地）。

誠實的讀法：不是 session 被回收，是實作本身的記憶體佔用在 T4 上
撐不住——Gram 收集階段同時持有模型權重 + rollout 生成的 KV +
float64 累加器。rollout 生成（最重的部分）其實跑得動；瓶頸在
`collect_grams` 的累加器設計。若未來再試，應把 Gram 累加器改成
逐層即時縮減（streaming reduce）或 CPU offload，而不是一次持有
全部。真實權重形狀上的可跑通性仍未確認。

## Colab rerun（2026-10-21，Peter 批准「重跑，下個朝早窗口」）

**未完成（session reclaimed，無結果）。** 帳號 kyu009009，T4 一次即取
（無需 fallback），`TFOPD_ROLLOUTS=32`，上傳的是 OOM-fix 後版本。
啟動後約 65–70 分鐘被 free tier 回收；回收前進程一直存活，
`py-spy` 證實卡在 math specialist 的 `collect_grams`（`model.generate`
→ forward hook `tf_opd_colab_demo.py:115`），從未 OOM、從未印出
`rollouts: kept=…`——收穫為零（reclaimed, no harvest），本地
`tf_opd_colab_results.log` 仍是 10-20 的 OOM 堆疊。

誠實的新發現（fix 的代價）：CPU-float32 累加器消除了 OOM，但每個
generate step 都要把每層 Gram delta 做一次完整 D2H copy
（約 2.5 GB/step 經 PCIe，僅 copy 估計 ~20 秒/prompt）——舊版
（CUDA 累加器）rollout 生成快但最後 OOM；新版慢到在 free T4 的
回收窗口內大概率跑不完 64 條 rollout。10-20 估的「~20 分鐘」是基
於舊版速度，fix 後的真實速度完全是另一回事。

下一步若再試（未批准，僅記錄）：hook 在 generate 階段根本不該累加
——Gram 只需要在 teacher-forced pass 收集；把 `GramCollector` 的
hook 改為只在 `model(full)` 那一下啟用（或 generate 時
`collector.handles` 先 remove），每 prompt 的 D2H copy 從
~96 steps 降到 1，可把速度拉回舊版水平。或者直接縮
`MAX_NEW_TOKENS`。在此之前，呢個 driver 唔值得再燒 free-tier
session。

GPU proof（eps medians/p90、perplexity 表）仍未取得；CPU-float32
fix 喺「唔 OOM」呢一點上 hold 住咗（撐到被回收都無 crash），但完整
run 嘅 proof 依然係零。Verdict 維持 SUPPORTED（數學核心）／未
VERIFIED。

## Fix（2026-10-20，working copy，未 push，未用 GPU）

根因修正（`tf_opd_colab_demo.py`）：`GramCollector` 累加器由 CUDA
float64 改為 **CPU float32**——兩個 specialist 的累加器共 ~10.9 GB，
本來就撐唔住 T4；hook 喺 CUDA 計 outer-product delta（快）之後
`.cpu()` 落 CPU；最後嘅平均改做 in-place `div_`，唔再 allocate
第二份 copy。咁樣 peak CUDA 只剩模型本身（~1.2 GB fp16）。

誠實講：修嘅係 OOM 嗰一下，唔係機制——64 條 rollout（kept=32+32，
dropped_loop=0）本身喺今朝已經成功生成，data-generating 部分係
work 嘅。CPU 驗證（/tmp/tfopd_fix_test.py）：accumulator 喺 CPU
float32、hook 數值同 float64 reference 最大相對誤差 8e-8、in-place
平均無 realloc、全程無 CUDA tensor。未經 GPU 實測——下次重跑
（TFOPD_ROLLOUTS=32，~20 分鐘）先係真正嘅 proof。

## Fix（2026-10-21，working copy，未 push，未用 GPU）

真兇唔止慢，仲計錯。`GramCollector` 嘅 forward hook 喺 `collect_grams`
開頭註冊，之後 `model.generate` 嘅 ~96 個 decode step 每步都觸發——
每層 Linear 每步做一次完整 outer-product delta + D2H copy（168 層，
每步 ~2.7 GB，每 prompt ~260 GB 無謂 D2H；今朝實測 ≥65s/prompt，
65–70 分鐘燒完都未做完 64 條 rollout）。仲 double-count 緊：decode
嗰陣 hook 計 full outer（`X^T X`），teacher-forced 嗰陣又計 average，
兩套語義混埋一齊。

修正（`tf_opd_colab_demo.py`）：`GramCollector` 加 `enabled` flag，
hook 第一行 `if not self.enabled: return`；`collect_grams` 喺
`model.generate` 之前設 `collector.enabled = False`，generate 完、
teacher-forced `model(full)` 之前設返 `True`。Gram 只喺
teacher-forced pass 累積，每 prompt D2H 由 ~96 次降到 1 次。

估計：`TFOPD_ROLLOUTS=32` 全程 ~10–15 分鐘（非 GPU 實測），35 分鐘
free-tier 窗口 fit 到。Caveat：reclamation 依然隨機；fix 後數值同舊
design 唔同（不過 GPU 從來冇出過數，無舊數可比）。
