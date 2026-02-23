### 第一阶段：引导提取 (Guide Extraction)

**目标**：从论文全文中“沙里淘金”，筛选出所有包含可复现细节（超参数、架构、数据划分）的句子，排除背景描述和废话。

**Prompt:**

```
**Role**: You are a meticulous Software Test Engineer with expertise in machine learning.

**Task**: Analyze the provided research paper sections to create a raw checklist of verifiable facts. These facts will be used to test a code implementation for correctness.

**Goal**: Identify and extract indices of sentences that contain **Code-Level Guidance**. These are specific, actionable claims that must be true in the code.

**Criteria for Selection (Select sentences that describe):**
1. **Data & Task**: Exact dataset names, splits (e.g., "80/10/10 split"), or specific task definitions.
2. **Preprocessing**: Normalization methods, augmentation strategies, tokenization details.
3. **Hyperparameters**: Concrete values for learning rate, batch size, dropout, epochs, optimizer, etc.
4. **Architecture**: Specific layers, dimensions, activation functions, or module connections.
5. **Algorithms**: Specific formulas (loss functions), logical steps, or initialization methods.
6. **Metrics**: Exact evaluation metrics used (e.g., "Top-1 Accuracy", "BLEU-4").

**What to IGNORE**:
- High-level abstract claims (e.g., "Our model is efficient").
- Future work or related work discussions.
- General background information.

**Input Paper Segment**:
{paper_segment}

**Output Format**:
Return ONLY a valid JSON array of integers representing the indices of the selected sentences.
Example:

```

*参考来源: RePro Appendix Figure 6*

---

### 第二阶段：原子化标准化 (Atomicity Standardization)

**目标**：这是核心步骤。将复杂的长句拆解为 `<Fact, Scope>` 对。**Fact** 是具体的数值或操作，**Scope** 是该操作生效的条件（数据集、阶段）。

**Prompt:**

```
**Role**: You are an expert technical writer specializing in creating executable specifications.

**Task**: Decompose the provided "Summary Facts" into **Atomic Verifiable Criteria**. A criterion is atomic if it can be verified by a simple "Pass/Fail" check against code.

**Instructions**:
1. **Decompose Complex Sentences**: Break down multi-part sentences into smallest meaningful units.
   - *Bad*: "We use Adam with lr=1e-3 on CIFAR and SGD with lr=0.1 on ImageNet."
   - *Good*: Split this into 4 separate criteria (Adam for CIFAR, lr=1e-3 for Adam/CIFAR, SGD for ImageNet, etc.).
2. **Preserve Equations**: Treat self-contained mathematical formulas or loss functions as a SINGLE indivisible unit. Do not break a formula into variable definitions.
3. **Structure as Fact-Scope Pairs**:
   - **<fact>**: The core specific value, method, or setting (e.g., "learning rate of 0.001", "ResNet-50 backbone").
   - **<scope>**: The context where this fact applies (e.g., "for the pre-training phase", "on the validation set").

**Input Facts**:
{extracted_sentences}

**Output Format**:
Return a JSON list of objects. Embed XML-style tags `<fact>` and `<scope>` within the criterion string.

**Example Output**:
[
  {
    "criterion": "A <fact>dropout rate of 0.5</fact> is applied <scope>to the output of the attention layer</scope>."
  },
  {
    "criterion": "The <fact>AdamW optimizer</fact> is used <scope>during the fine-tuning stage</scope>."
  },
  {
    "criterion": "The loss function is defined as <fact>L = L_ce + 0.1 * L_reg</fact> <scope>for all training epochs</scope>."
  }
]

```

*参考来源: RePro Appendix Figure 7-8*

---

### 第三阶段：过滤与去重 (Filtering & Deduplication)

**目标**：论文中经常重复提到同一个超参数。此步骤用于合并语义相同的指纹，保留最清晰、最可验证的那一个。

**Prompt:**

```
**Role**: You are a QA Lead Engineer.

**Task**: You are given a list of extracted criteria that are semantically similar. Your goal is to select the **Best Representative** for this group to include in the final verification checklist.

**Selection Principles**:
1. **Directly Verifiable**: Prioritize criteria that map directly to code (e.g., "batch_size = 32" is better than "a moderate batch size").
2. **Unambiguous**: Choose the phrasing that leaves no room for interpretation.
3. **Complete**: Ensure the scope is clearly defined.

**Input Group**:
{cluster_of_similar_criteria}

**Output Format**:
Return a JSON object containing the index of the selected item and the reason.

**Example Input**:
1. "The model uses 12 layers."
2. "We employ a 12-layer Transformer encoder."
3. "Depth is set to 12."

**Example Output**:
{
  "selected_index": 2,
  "reason": "Item 2 is the most specific, specifying both the number of layers and the architecture type (Transformer encoder)."
}

```

*参考来源: RePro Appendix Figure 9-10*

---

### 补充：多模态图表证据提取 (针对包含图表的论文)

如果您的验证涉及图表（如对比 Claim 与 Figure），参考 **MuSciClaims** 的逻辑，增加一个针对图表的提取 Prompt。

**Prompt:**

```
**Role**: You are a Scientific Data Analyst.

**Task**: For the verified claim below, identify the specific visual evidence required from the paper's figures.

**Claim**: "{atomic_claim}"

**Input**: Figure Captions: {all_figure_captions}

**Instructions**:
1. Identify which Figure (e.g., Figure 1, Table 2) contains the data supporting this claim.
2. Specify the exact Panel or Sub-figure (e.g., Panel A, red curve).
3. Extract the trend or value that should be observed (e.g., "The blue line should be higher than the red line").

**Output JSON**:
{
  "figure_id": "Figure 3",
  "panel_id": "b",
  "expected_observation": "Accuracy curve converges after 100 epochs",
  "data_type": "Quantitative Trend"
}

```

### 实施建议

1. **分步执行**：不要试图用一个 Prompt 完成所有工作。先用 Prompt 1 拿到句子，再用 Prompt 2 拆解，最后用 Prompt 3 清洗。
2. **Few-Shot 示例**：在 Prompt 中包含 1-2 个具体领域的例子（如您关注的 AI/ML 领域，使用 "Transformer layers", "Learning rate" 作为示例）能显著提高 LLM 的遵循度。
3. **JSON 强制**：始终要求返回 JSON 格式，以便后续 Python 脚本可以直接解析并存入数据库。

### 📄 科学论文指纹表 (Paper Fingerprint Schema)

这个表格分为四个维度：**元数据**、**复现配置**（给执行智能体用）、**原子主张**（给验证智能体用）和**多模态证据**。

| 维度 (Dimension)                                     | 关键要素 (Key Elements)   | 提取内容说明 (Description)    | 示例 (Example Values)                                                 | 目的 (Purpose) |
| -------------------------------------------------- | --------------------- | ----------------------- | ------------------------------------------------------------------- | ------------ |
| **1. 基础元数据**(Metadata)                             | **Paper ID**          | 论文唯一标识 (DOI/ArXiv ID)   | `10.1145/xxxxx`                                                     | 追踪来源         |
|                                                    | **Repository URL**    | 代码仓库链接                  | `github.com/user/repo`                                              | 代码克隆入口       |
|                                                    | **Venue/Year**        | 发表会议/年份                 | `ICLR 2024`                                                         | 确定SOTA基准     |
| **2. 复现配置**(Reproducibility Config)*供 Coder 智能体使用* | **Dataset Specs**     | 数据集名称、版本、划分方式           | `Name: CIFAR-10`, `Split: 80/20 train/test`                         | 数据准备         |
|                                                    | **Hyperparameters**   | 所有的超参数配置（键值对）           | `{"lr": 0.001, "batch_size": 64, "optimizer": "AdamW", "seed": 42}` | 修改 Config 文件 |
|                                                    | **Model Arch**        | 模型具体架构或层数               | `ResNet-50`, `12-layer Transformer`                                 | 模型实例化        |
|                                                    | **Environment**       | 硬件/软件依赖                 | `PyTorch 2.1`, `CUDA 11.8`, `A100 GPU`                              | Docker 环境配置  |
|                                                    | **Evaluation Metric** | 核心评估指标名称                | `Top-1 Accuracy`, `F1-Score`, `BLEU`                                | 结果解析目标       |
| **3. 原子主张**(Atomic Claims)*供 Verifier 智能体使用*       | **Atomic Fact**       | **核心：** 将复杂结论拆解为单一数值或趋势 | `Accuracy = 92.5%`                                                  | **验证的“金标准”** |
|                                                    | **Scope/Condition**   | 该事实成立的具体条件范围            | `on Test Set (ImageNet)`, `after 100 epochs`                        | 防止张冠李戴       |
|                                                    | **Comparator**        | 对比对象（基线模型）              | `vs. ResNet-18 (89.0%)`                                             | 验证比较优势       |
|                                                    | **Claim Type**        | 主张类型                    | `Empirical` (实证), `Methodological` (方法), `Comparative` (比较)         | 决定验证逻辑       |
| **4. 证据锚点**(Evidence Anchors)*供 Reader 智能体使用*      | **Text Anchor**       | 原文中陈述该主张的句子索引           | `Section 4.2, Paragraph 1`                                          | 文本溯源         |
|                                                    | **Visual Anchor**     | 支持该主张的图表及子图编号           | `Figure 3 (b)`, `Table 2, Row 4`                                    | 多模态验证        |
|                                                    | **Visual Data**       | 从图表中提取的原始数值（如有）         | `{"x":, "y": [0.5, 0.8]}`                                           | 图表趋势比对       |

---

### 💻 JSON 结构示例 (供智能体输出)

在实际开发中，您可以要求您的 **Reader 智能体**（信息提取模块）直接输出如下 JSON 格式。这种格式是 **RePro** 推荐的“Fact-Scope”结构，非常便于后续的代码验证。

```json
{
  "fingerprint_id": "arXiv:2401.xxxxx",
  "configurations": {
    "framework": "PyTorch",
    "hardware_req": "1x NVIDIA A100",
    "hyperparameters": {
      "learning_rate": 3e-4,
      "batch_size": 256,
      "epochs": 100,
      "seed": 42,
      "optimizer": "Adam"
    },
    "dataset": {
      "name": "ImageNet-1K",
      "resolution": "224x224",
      "split_strategy": "standard validation set"
    }
  },
  "claims": [
    {
      "id": "claim_01",
      "type": "Empirical",
      "fact": "Top-1 Accuracy = 82.3%",
      "scope": "on ImageNet-1K validation set after 100 epochs",
      "verification_logic": "exact_match",
      "tolerance": "±0.2%",
      "evidence_source": {
        "type": "Table",
        "location": "Table 1, Row: Ours, Col: Acc"
      }
    },
    {
      "id": "claim_02",
      "type": "Comparative",
      "fact": "Outperforms Baseline X by 2.1%",
      "scope": "in terms of F1-score on Dataset Y",
      "verification_logic": "greater_than_margin",
      "margin": 2.1,
      "evidence_source": {
        "type": "Chart",
        "location": "Figure 4(a) bar chart"
      }
    }
  ]
}
```

### 💡 关键设计点说明：

1. **Fact-Scope 分离 (RePro 模式)**：

   - 不要提取“我们的模型在ImageNet上达到了82.3%的准确率”这种长句。
   - **必须拆解为**：`Fact: "Acc=82.3%"` + `Scope: "ImageNet val set"`。这样您的验证智能体（Verifier）只需要检查代码日志中 `val_acc` 变量是否接近 `0.823`。

2. **原子化 (Atomicity)**：

   - 如果论文说“我们在准确率和推理速度上都优于基线”，这是**两个**指纹条目。验证时必须分别通过才算“Supported”。

3. **容忍度 (Tolerance)**：

   - 由于随机种子和硬件差异，复现结果很难 100% 匹配。在指纹中预设一个 `tolerance`（如 ±0.5% 或 ±1个标准差）是防止智能体误判“验证失败”的关键。

4. **多模态锚点 (Visual Anchors)**：

   - 许多科学结论（尤其是比较优势）仅存在于图表中。提取器必须能够记录“证据在图3的红色曲线”，以便后续的多模态模型（如 GPT-4o）去“看”图验证，或者从生成的代码画出的图中进行比对。

