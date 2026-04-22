# Self-Refine / Critic Loop 筆記

> KB 正本：[self-refine-critic-loop](~/PycharmProjects/llm_knowledge_base/wiki/concepts/self-refine-critic-loop.md)
> 論文：[Madaan et al. 2023](https://arxiv.org/abs/2303.17651)、[Reflexion (Shinn et al. 2023)](https://arxiv.org/abs/2303.11366)

## 一句話

**同一個 model** 先生成 → 自我批評 → 修正。對 reasoning / generation 有用；對 atomic extraction 幫助邊際 —— 抽取場景用 self-consistency 通常更合算。

## 三步驟

1. **Generate** — 初始輸出
2. **Critique** — 依照明確 rubric 給 feedback
3. **Refine** — 修改版輸出，吸收 critique

Reflexion 延伸：critique 存成 persistent memory 跨 trial 累積（verbal RL）

## 程式骨架

```python
output = model.generate(task)
for _ in range(max_iterations):
    critique = model.critique(task, output, rubric)
    if critique.says_acceptable():
        break
    output = model.refine(task, output, critique)
```

## 哪裡強 / 哪裡弱

**強**（Madaan 實驗）：
- Code 生成（錯誤在 re-read 時可見）
- Multi-constraint 生成（逐條 constraint 檢查）
- 數學與多步推理

**弱**：
- Atomic extraction — model 的 initial prior 就決定答案，critic 逃不出去
- 正確性檢查需要外部資訊時（model 自己沒資訊）

## vs LLM-as-Judge

| | LLM-as-Judge | Self-Refine |
|---|---|---|
| 誰來 critique | 不同的（更強）model | 同一個 model |
| 成本 | 1 judge + 1 generate | (1 + 2k) calls for k rounds |
| 逃得出自身盲點 | 可以 | 不可以（同 model 同盲點） |

## 成本與 self-consistency 相仿

(1 + 2k) calls ≈ N=2k+1 的 self-consistency，但 **self-refine 沒有 self-consistency 免費附贈的 confidence 訊號**。所以在 extraction 幾乎永遠輸。

## 業界應用

- **Cursor / Copilot**：內部 critic loop 在 surface suggestion 前抓 syntax / type 錯
- **Bard / ChatGPT long-form writing**：使用者可見的 revision 是 critic loop
- **Agent harness**：LangGraph / CrewAI agent tool 執行失敗後的 replanning step 都是 critic loop 變體
- **Anthropic Claude Code `/review`**：built-in critic loop 加 rubric
- **Patsnap**（邊際）：專利 atomic 參數抽取別用；但 summary / 多專利趨勢報告 / Claim interpretation 推理可用

## 什麼時候用 / 不用

**用**：
- Task 是推理 / generation，不是 atomic extraction
- 錯誤 re-read 可檢（syntax / 單位 / 漏 constraint）
- 成本允許 3-5× 推理

**不用**：
- Atomic extraction（換 self-consistency）
- Critique 需要 model 沒有的知識（換 verifier 或 retrieval verification）
- Latency-bound UI

## 關鍵限制

- **Gain cap at model ceiling**：model 解不出的問題，critic 救不回來。Reflexion 實驗顯示 2-3 iteration 後 diminishing returns
- **Critique prompt design 決定成敗**：泛泛「找問題」差；task-specific rubric（「檢查單位換算」「確認引用 span 存在」）好
- 可與 [retrieval-augmented verification](notes_retrieval_augmented_verification.md) 合用 —— 用 retrieval 把 critique 綁到 source text，跳出 model 自身 prior
