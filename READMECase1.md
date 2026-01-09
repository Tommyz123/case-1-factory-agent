# Case 1 – Physical Factory Search Agent

## Mission

Build an AI Agent that:
- **Input**: Fortune Global 500 company names
- **Output**: A structured list of their **physical manufacturing facilities**

If needed we can provide any API key for OpenAI / DeepSeek / Claude / Gemini / Qwen / Zhipu (ZLM).

Examples:
- Factories
- Refineries
- Assembly plants
- Manufacturing sites
- R&D centers

---

## Critical Constraint (Hard Rule)

✅ MUST be a place that **produces physical goods**

❌ MUST NOT include:
- Headquarters (HQ)
- Sales offices
- Administrative buildings

Any violation = **automatic failure**.

---

## Technical Requirements

- Agent framework: LangGraph / CrewAI / AutoGen (or equivalent)
- Data: Public web search + optional databases
- Optional bonus: Satellite or map-based validation
- Must include a technical document explaining:
  - Framework choice
  - Data/index choice
  - Extraction & validation logic
  - How non-production sites are excluded

---

## Output Format (Recommended)

Each record should include:
- company_name
- facility_name
- facility_type
- full_address
- country / city
- latitude / longitude (optional)
- evidence_urls (mandatory)
- confidence_score (optional)

## Deliverables

- System design document
- Demo or core implementation
- Sample test cases with expected outputs
- Time spent on the case and tokens used

---

# 案例1 – 实体工厂搜索智能体

## 任务目标

构建一个 AI 智能体，能够：
- **输入**：财富全球500强公司名称
- **输出**：该公司**实体制造设施**的结构化列表

如有需要，我们可提供 OpenAI / DeepSeek / Claude / Gemini / 通义千问 / 智谱（ZLM）的 API 密钥

示例：
- 工厂
- 炼油厂
- 组装车间
- 制造基地
- 研发中心

---

## 核心约束（硬性规则）

✅ 必须是**生产实体产品**的场所

❌ 不得包含：
- 总部（HQ）
- 销售办公室
- 行政办公楼

任何违规 = **自动失败**。

---

## 技术要求

- 智能体框架：LangGraph / CrewAI / AutoGen（或同类框架）
- 数据来源：公开网络搜索 + 可选数据库
- 可选加分项：卫星图像或地图验证
- 必须包含技术文档，说明：
  - 框架选择理由
  - 数据/索引选择
  - 提取与验证逻辑
  - 如何排除非生产性场所

---

## 输出格式（建议）

每条记录应包含：
- company_name（公司名称）
- facility_name（设施名称）
- facility_type（设施类型）
- full_address（完整地址）
- country / city（国家/城市）
- latitude / longitude（经纬度，可选）
- evidence_urls（证据链接，必填）
- confidence_score（置信度，可选）

## 提交方式
- 代码实现
- 技术文档
- 包含预期输出的测试用例
- 完成案例所花费的时间 和 Tokens