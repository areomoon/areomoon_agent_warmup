# 03 Agent Patterns

**對應 Warmup Week 3 | 預估時數：10–12hr**

## 學習目標

掌握 Generator-Reflector pattern 和 LangGraph；建出能讀論文 → 抽取實驗參數的最小 Agent。

完成後你能：
- [ ] 實作 ACE Generator-Reflector loop
- [ ] 用 LangGraph 建 reflection agent（自我修正）
- [ ] 建出 Orchestrator + Extractor 的雙 agent 系統
- [ ] 理解 Mailbox pattern 和 Plan→Work→Review cycle

## 執行順序

1. `python generator_reflector.py` — 核心 GRC pattern（最重要）
2. `python reflection_agent.py` — LangGraph 自我改進 agent
3. `python multi_agent_basic.py` — Orchestrator 協調多個 specialist
4. 打開 `notebook/agent_patterns_lab.ipynb`

## 關鍵資源

- [Andrew Ng — Reflection Pattern](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/)
- [LangChain Reflection Agents](https://blog.langchain.com/reflection-agents/)
- [ACE Playbook 開源實作](https://github.com/jmanhype/ace-playbook)
- [LangGraph 自我改進 Agent](https://medium.com/@shuv.sdr/langgraph-build-self-improving-agents-8ffefb52d146)
