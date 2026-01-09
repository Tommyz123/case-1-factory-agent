# 实体工厂搜索智能体 - 技术设计文档

**版本**: 1.0
**创建日期**: 2025-01-06
**作者**: Factory Agent Development Team

---

## 目录

1. [系统概述](#1-系统概述)
2. [架构设计](#2-架构设计)
3. [数据模型定义](#3-数据模型定义)
4. [模块接口定义](#4-模块接口定义)
5. [模块间通信协议](#5-模块间通信协议)
6. [错误处理设计](#6-错误处理设计)
7. [日志系统设计](#7-日志系统设计)
8. [API集成规范](#8-api集成规范)
9. [性能优化设计](#9-性能优化设计)
10. [测试策略](#10-测试策略)

---

## 1. 系统概述

### 1.1 项目背景

构建一个AI智能体，能够根据财富全球500强公司名称，自动搜索并输出其**实体制造设施**的结构化列表。

**核心能力**:
- **输入**: 公司名称（如"Toyota Motor Corporation"）
- **输出**: 结构化的制造设施列表（工厂、炼油厂、组装厂等）
- **约束**: 严格排除总部、销售办公室、行政办公楼

### 1.2 核心约束（硬性规则）

#### ✅ 必须包含
生产实体产品的场所:
- 工厂 (Factory)
- 炼油厂 (Refinery)
- 组装车间 (Assembly Plant)
- 制造基地 (Manufacturing Site)
- 铸造厂 (Foundry)
- 制造厂 (Mill)

#### ❌ 严格排除
以下任何一项出现在输出中都会**自动失败**:
- 总部 (Headquarters / HQ)
- 销售办公室 (Sales Office)
- 行政办公楼 (Administrative Building)
- **纯研发中心 (R&D Center，不生产实体产品的)**

### 1.3 技术栈选择

| 组件 | 技术选型 | 理由 |
|------|---------|------|
| **Agent框架** | LangGraph | 状态管理清晰，适合多步骤工作流 |
| **主LLM** | GPT-4o-mini | 成本低，速度快，适合结构化数据提取 |
| **备用LLM** | GPT-4.1 | 更强推理能力，用于复杂验证场景 |
| **搜索API** | Tavily Search | 专为AI Agent设计，返回结构化数据 |
| **Embedding** | text-embedding-3-small | OpenAI，用于去重和相似度计算 |
| **数据验证** | Pydantic | 强类型验证，自动JSON序列化 |
| **并发处理** | ThreadPoolExecutor | Python标准库，轻量级并行 |

---

## 2. 架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Input: Company Name                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │      SearchNode (搜索节点)      │
         │  • 构建搜索查询                  │
         │  • 调用Tavily API               │
         │  • 结果排序和初步去重            │
         └───────────┬───────────────────┘
                     │ SearchResult[]
                     ▼
         ┌───────────────────────────────┐
         │   ExtractionNode (提取节点)     │
         │  • LLM提取结构化数据            │
         │  • JSON Schema验证              │
         │  • 初步关键词过滤               │
         └───────────┬───────────────────┘
                     │ FacilityCandidate[]
                     ▼
         ┌───────────────────────────────┐
         │   ValidationNode (验证节点)     │
         │  • 第1层：关键词过滤            │
         │  • 第2层：地址验证              │
         │  • 第3层：LLM二次验证           │
         └───────────┬───────────────────┘
                     │ ValidatedFacility[]
                     ▼
         ┌───────────────────────────────┐
         │   EvidenceNode (证据收集节点)   │
         │  • 收集evidence_urls            │
         │  • 评估证据质量                 │
         │  • 提取证据文本片段             │
         └───────────┬───────────────────┘
                     │ ValidatedFacility[] (更新)
                     ▼
         ┌───────────────────────────────┐
         │ DeduplicationNode (去重节点)    │
         │  • Embedding相似度计算          │
         │  • 地理位置去重                 │
         │  • 合并重复设施                 │
         └───────────┬───────────────────┘
                     │ ValidatedFacility[] (去重后)
                     ▼
         ┌───────────────────────────────┐
         │     OutputNode (输出节点)       │
         │  • 格式化最终输出               │
         │  • 计算执行指标                 │
         │  • 生成JSON文件                 │
         └───────────┬───────────────────┘
                     │
                     ▼
         ┌───────────────────────────────┐
         │     FinalOutput (最终输出)      │
         │  • company_name                 │
         │  • facilities: List             │
         │  • metrics: ExecutionMetrics    │
         └─────────────────────────────────┘
```

### 2.2 模块划分

| 模块名 | 职责 | 输入 | 输出 |
|--------|------|------|------|
| SearchNode | 搜索相关设施信息 | company_name | List[SearchResult] |
| ExtractionNode | 提取结构化数据 | List[SearchResult] | List[FacilityCandidate] |
| ValidationNode | 多层验证 | List[FacilityCandidate] | List[ValidatedFacility] |
| EvidenceNode | 收集证据链接 | List[ValidatedFacility] | List[ValidatedFacility] |
| DeduplicationNode | 去除重复设施 | List[ValidatedFacility] | List[ValidatedFacility] |
| OutputNode | 格式化输出 | AgentState | FinalOutput |

### 2.3 数据流设计

```python
# 数据流示例
company_name: str
    ↓ SearchNode
search_results: List[SearchResult]
    ↓ ExtractionNode
extracted_candidates: List[FacilityCandidate]
    ↓ ValidationNode
validated_facilities: List[ValidatedFacility]
rejected_facilities: List[Dict]  # 被拒绝的设施
    ↓ EvidenceNode
validated_facilities: List[ValidatedFacility]  # 更新evidence_urls
    ↓ DeduplicationNode
deduplicated_facilities: List[ValidatedFacility]
    ↓ OutputNode
final_output: FinalOutput
```

---

## 3. 数据模型定义

### 3.1 SearchResult（搜索结果模型）

```python
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional
from datetime import datetime

class SearchResult(BaseModel):
    """搜索API返回的单条结果"""

    url: HttpUrl = Field(..., description="搜索结果URL")
    title: str = Field(..., description="页面标题", min_length=1)
    snippet: str = Field(..., description="页面摘要", min_length=10)
    relevance_score: Optional[float] = Field(
        None,
        description="相关性评分(0-1)",
        ge=0.0,
        le=1.0
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="搜索时间戳"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.toyota.com/kentucky-plant",
                "title": "Toyota Kentucky Manufacturing Plant",
                "snippet": "The Georgetown plant produces Camry and Avalon models...",
                "relevance_score": 0.92,
                "timestamp": "2025-01-06T12:00:00Z"
            }
        }
```

**字段说明**:
- `url`: 搜索结果的网页链接（必填，使用Pydantic的HttpUrl自动验证）
- `title`: 网页标题（必填，最少1个字符）
- `snippet`: 网页摘要/描述（必填，最少10个字符）
- `relevance_score`: 搜索引擎返回的相关性评分（可选，0-1之间）
- `timestamp`: 搜索时间戳（自动生成）

---

### 3.2 FacilityCandidate（候选设施模型）

```python
from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional

class FacilityCandidate(BaseModel):
    """LLM提取后的候选设施（未验证）"""

    facility_name: str = Field(..., description="设施名称", min_length=3)
    facility_type: str = Field(
        ...,
        description="设施类型（工厂/炼油厂/组装厂等）"
    )
    full_address: str = Field(..., description="完整地址", min_length=10)
    country: str = Field(..., description="国家", min_length=2)
    city: str = Field(..., description="城市", min_length=2)

    latitude: Optional[float] = Field(
        None,
        description="纬度",
        ge=-90.0,
        le=90.0
    )
    longitude: Optional[float] = Field(
        None,
        description="经度",
        ge=-180.0,
        le=180.0
    )

    evidence_snippet: str = Field(
        ...,
        description="证据文本片段（证明是生产设施）",
        min_length=20
    )
    source_url: HttpUrl = Field(..., description="信息来源URL")

    # 验证前的初始标记
    needs_validation: bool = Field(default=True, description="是否需要验证")

    @validator('facility_type')
    def validate_facility_type(cls, v):
        """验证设施类型合法性"""
        allowed_types = [
            'factory', 'plant', 'refinery', 'assembly',
            'manufacturing', 'mill', 'foundry', 'fabrication'
        ]
        if not any(allowed in v.lower() for allowed in allowed_types):
            raise ValueError(f'Invalid facility_type: {v}. Must contain one of: {allowed_types}')
        return v

    @validator('evidence_snippet')
    def validate_no_excluded_keywords(cls, v):
        """初步检查证据片段中不包含排除关键词"""
        excluded = ['headquarters', 'hq', 'head office', 'sales office', 'r&d only']
        for keyword in excluded:
            if keyword in v.lower():
                raise ValueError(f'Evidence contains excluded keyword: {keyword}')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "facility_name": "Toyota Kentucky Plant",
                "facility_type": "assembly plant",
                "full_address": "1001 Cherry Blossom Way, Georgetown, KY 40324, USA",
                "country": "United States",
                "city": "Georgetown",
                "latitude": 38.2098,
                "longitude": -84.5588,
                "evidence_snippet": "The Georgetown plant produces Camry and Avalon models with over 8,000 employees",
                "source_url": "https://www.toyota.com/kentucky-plant",
                "needs_validation": True
            }
        }
```

**字段说明**:
- `facility_name`: 设施名称（如"Toyota Kentucky Plant"）
- `facility_type`: 设施类型，必须包含合法关键词（factory/plant/refinery等）
- `full_address`: 完整地址
- `country`, `city`: 国家和城市
- `latitude`, `longitude`: 经纬度（可选）
- `evidence_snippet`: 证明是生产设施的文本片段（最少20字符）
- `source_url`: 信息来源

**验证器**:
- `validate_facility_type`: 确保类型包含合法关键词
- `validate_no_excluded_keywords`: 初步检查不包含HQ/Office等关键词

---

### 3.3 ValidatedFacility（验证后的设施模型）

```python
from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Optional
from enum import Enum
from datetime import datetime

class ValidationStatus(str, Enum):
    """验证状态枚举"""
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"

class ValidatedFacility(BaseModel):
    """通过验证的设施"""

    # 基本信息（继承自 FacilityCandidate）
    facility_name: str = Field(..., description="设施名称")
    facility_type: str = Field(..., description="设施类型")
    full_address: str = Field(..., description="完整地址")
    country: str = Field(..., description="国家")
    city: str = Field(..., description="城市")
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # 证据（必须是列表，至少1个URL）
    evidence_urls: List[HttpUrl] = Field(
        ...,
        description="证据链接列表（至少1个）",
        min_length=1
    )
    evidence_snippets: List[str] = Field(
        default_factory=list,
        description="证据文本片段列表"
    )

    # 验证结果
    validation_status: ValidationStatus = Field(
        ...,
        description="验证状态"
    )
    validation_reasons: List[str] = Field(
        default_factory=list,
        description="验证通过/失败的原因"
    )

    # 置信度评分
    confidence_score: float = Field(
        ...,
        description="置信度评分(0-1)",
        ge=0.0,
        le=1.0
    )

    # 元数据
    extracted_at: datetime = Field(
        default_factory=datetime.now,
        description="提取时间"
    )
    validated_at: Optional[datetime] = Field(
        None,
        description="验证时间"
    )

    @validator('evidence_urls')
    def validate_evidence_not_empty(cls, v):
        """确保证据链接不为空"""
        if not v or len(v) == 0:
            raise ValueError('evidence_urls cannot be empty - at least 1 URL required')
        return v

    @validator('confidence_score')
    def validate_confidence_threshold(cls, v, values):
        """如果验证通过，置信度必须>=0.7"""
        if values.get('validation_status') == ValidationStatus.PASSED:
            if v < 0.7:
                raise ValueError('Passed facilities must have confidence >= 0.7')
        return v

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "facility_name": "Toyota Kentucky Plant",
                "facility_type": "assembly plant",
                "full_address": "1001 Cherry Blossom Way, Georgetown, KY 40324, USA",
                "country": "United States",
                "city": "Georgetown",
                "latitude": 38.2098,
                "longitude": -84.5588,
                "evidence_urls": [
                    "https://www.toyota.com/kentucky-plant",
                    "https://en.wikipedia.org/wiki/Toyota_Motor_Manufacturing_Kentucky"
                ],
                "evidence_snippets": [
                    "The Georgetown plant produces Camry and Avalon models"
                ],
                "validation_status": "passed",
                "validation_reasons": [
                    "Contains production keywords: 'produces', 'manufacturing'",
                    "No HQ/office keywords found",
                    "LLM confirmed as production facility"
                ],
                "confidence_score": 0.95,
                "extracted_at": "2025-01-06T12:00:00Z",
                "validated_at": "2025-01-06T12:05:00Z"
            }
        }
```

**新增字段**:
- `evidence_urls`: 证据链接列表（**必填，至少1个URL**）
- `evidence_snippets`: 证据文本片段列表
- `validation_status`: 验证状态（passed/failed/needs_review）
- `validation_reasons`: 验证通过/失败的原因列表
- `confidence_score`: 置信度评分（0-1），通过的设施必须>=0.7
- `extracted_at`, `validated_at`: 时间戳

---

### 3.4 FinalOutput（最终输出模型）

```python
from pydantic import BaseModel, Field, validator
from typing import List
from datetime import datetime

class ExecutionMetrics(BaseModel):
    """执行指标"""
    total_search_results: int = Field(..., description="搜索结果总数")
    extracted_candidates: int = Field(..., description="提取的候选设施数")
    validated_facilities: int = Field(..., description="验证通过的设施数")
    rejected_facilities: int = Field(..., description="被拒绝的设施数")

    total_tokens_used: int = Field(default=0, description="总Token使用量")
    total_api_calls: int = Field(default=0, description="总API调用次数")
    execution_time_seconds: float = Field(..., description="执行时间（秒）")

    errors_count: int = Field(default=0, description="错误数量")

class FinalOutput(BaseModel):
    """最终输出结果"""

    # 基本信息
    company_name: str = Field(..., description="公司名称")

    # 设施列表
    facilities: List[ValidatedFacility] = Field(
        default_factory=list,
        description="验证通过的设施列表"
    )

    # 统计信息
    total_count: int = Field(..., description="设施总数")

    # 执行指标
    metrics: ExecutionMetrics = Field(..., description="执行指标")

    # 时间戳
    search_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="搜索时间戳"
    )

    # 警告和错误
    warnings: List[str] = Field(
        default_factory=list,
        description="警告信息列表"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="错误信息列表"
    )

    @validator('total_count')
    def validate_count_matches_facilities(cls, v, values):
        """确保总数与设施列表长度一致"""
        facilities = values.get('facilities', [])
        if v != len(facilities):
            raise ValueError(f'total_count ({v}) does not match facilities length ({len(facilities)})')
        return v

    def to_json_file(self, filepath: str):
        """导出为JSON文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=2, exclude_none=True))

    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "Toyota Motor Corporation",
                "facilities": [],  # List of ValidatedFacility
                "total_count": 15,
                "metrics": {
                    "total_search_results": 50,
                    "extracted_candidates": 25,
                    "validated_facilities": 15,
                    "rejected_facilities": 10,
                    "total_tokens_used": 15000,
                    "total_api_calls": 30,
                    "execution_time_seconds": 45.5,
                    "errors_count": 0
                },
                "search_timestamp": "2025-01-06T12:00:00Z",
                "warnings": [],
                "errors": []
            }
        }
```

---

### 3.5 AgentState（LangGraph状态）

```python
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

class AgentState(TypedDict):
    """LangGraph 智能体状态"""

    # ========== 输入 ==========
    company_name: str  # 公司名称

    # ========== 搜索阶段 ==========
    search_queries: List[str]  # 搜索查询列表
    search_results: List[SearchResult]  # 搜索结果列表

    # ========== 提取阶段 ==========
    extracted_candidates: List[FacilityCandidate]  # 提取的候选设施

    # ========== 验证阶段 ==========
    validated_facilities: List[ValidatedFacility]  # 验证通过的设施
    rejected_facilities: List[Dict[str, Any]]  # 被拒绝的设施（包含原因）

    # ========== 去重阶段 ==========
    deduplicated_facilities: List[ValidatedFacility]  # 去重后的设施

    # ========== 输出 ==========
    final_output: Optional[FinalOutput]  # 最终输出

    # ========== 执行追踪 ==========
    current_node: str  # 当前执行的节点
    execution_log: List[Dict[str, Any]]  # 执行日志
    errors: List[Dict[str, Any]]  # 错误列表
    warnings: List[str]  # 警告列表

    # ========== 性能指标 ==========
    metrics: Dict[str, Any]  # 性能指标
    start_time: datetime  # 开始时间
    end_time: Optional[datetime]  # 结束时间


def create_initial_state(company_name: str) -> AgentState:
    """创建初始状态"""
    return AgentState(
        # 输入
        company_name=company_name,

        # 中间状态
        search_queries=[],
        search_results=[],
        extracted_candidates=[],
        validated_facilities=[],
        rejected_facilities=[],
        deduplicated_facilities=[],

        # 输出
        final_output=None,

        # 执行追踪
        current_node="init",
        execution_log=[],
        errors=[],
        warnings=[],

        # 性能指标
        metrics={
            "total_tokens_used": 0,
            "total_api_calls": 0,
            "search_api_calls": 0,
            "llm_api_calls": 0,
            "embedding_api_calls": 0
        },
        start_time=datetime.now(),
        end_time=None
    )
```

---

## 4. 模块接口定义

### 4.1 基础节点接口

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, Any
from datetime import datetime

StateT = TypeVar('StateT')

class BaseNode(ABC, Generic[StateT]):
    """节点基类 - 所有节点必须继承此类"""

    def __init__(self, node_name: str):
        self.node_name = node_name

    @abstractmethod
    def execute(self, state: StateT) -> StateT:
        """
        执行节点逻辑（必须实现）

        Args:
            state: 当前状态

        Returns:
            更新后的状态

        Raises:
            NodeExecutionError: 节点执行失败
        """
        pass

    def log_execution(self, state: StateT, message: str, level: str = "INFO"):
        """记录执行日志"""
        log_entry = {
            "node": self.node_name,
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        state["execution_log"].append(log_entry)

    def log_error(self, state: StateT, error: Exception, context: Dict[str, Any] = None):
        """记录错误"""
        error_entry = {
            "node": self.node_name,
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        state["errors"].append(error_entry)
```

---

### 4.2 SearchNode接口

```python
from typing import List, Tuple

class SearchNodeInterface(BaseNode[AgentState]):
    """搜索节点接口"""

    @abstractmethod
    def build_search_queries(self, company_name: str) -> List[str]:
        """
        构建搜索查询

        Args:
            company_name: 公司名称

        Returns:
            搜索查询列表（3-5个查询）

        Example:
            >>> build_search_queries("Toyota Motor Corporation")
            [
                "Toyota Motor Corporation manufacturing facilities locations",
                "Toyota factories plants assembly -headquarters -office",
                "Toyota production sites refineries"
            ]
        """
        pass

    @abstractmethod
    def perform_search(self, query: str) -> List[SearchResult]:
        """
        执行单个搜索查询

        Args:
            query: 搜索查询

        Returns:
            搜索结果列表（最多10条）

        Raises:
            SearchAPIError: 搜索API调用失败
        """
        pass

    @abstractmethod
    def rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        对搜索结果排序

        Args:
            results: 原始搜索结果

        Returns:
            排序后的搜索结果（按相关性降序）
        """
        pass

    def execute(self, state: AgentState) -> AgentState:
        """
        执行搜索节点

        完整流程:
        1. 构建搜索查询
        2. 并行执行所有查询
        3. 合并和排序结果
        4. 更新状态
        """
        self.log_execution(state, f"Searching for: {state['company_name']}")

        # 构建查询
        queries = self.build_search_queries(state['company_name'])
        state['search_queries'] = queries

        # 执行搜索
        all_results = []
        for query in queries:
            try:
                results = self.perform_search(query)
                all_results.extend(results)
                state['metrics']['search_api_calls'] += 1
            except Exception as e:
                self.log_error(state, e, {"query": query})

        # 排序和去重
        ranked_results = self.rank_results(all_results)
        state['search_results'] = ranked_results

        self.log_execution(
            state,
            f"Search complete: {len(ranked_results)} results",
            "INFO"
        )

        return state
```

---

### 4.3 ExtractionNode接口

```python
class ExtractionNodeInterface(BaseNode[AgentState]):
    """提取节点接口"""

    @abstractmethod
    def extract_from_result(
        self,
        search_result: SearchResult,
        company_name: str
    ) -> List[FacilityCandidate]:
        """
        从单个搜索结果中提取设施信息

        Args:
            search_result: 搜索结果
            company_name: 公司名称（用于上下文）

        Returns:
            提取的候选设施列表（可能为空）

        Raises:
            ExtractionError: 提取失败
        """
        pass

    @abstractmethod
    def build_extraction_prompt(
        self,
        content: str,
        company_name: str
    ) -> str:
        """
        构建提取提示词

        Args:
            content: 网页内容/摘要
            company_name: 公司名称

        Returns:
            提示词字符串
        """
        pass

    @abstractmethod
    def parse_llm_response(self, response: str) -> List[FacilityCandidate]:
        """
        解析LLM响应

        Args:
            response: LLM返回的JSON字符串

        Returns:
            候选设施列表

        Raises:
            ParseError: 解析失败
        """
        pass
```

---

### 4.4 ValidationNode接口

```python
from typing import Tuple

class ValidationNodeInterface(BaseNode[AgentState]):
    """验证节点接口 - 三层验证机制"""

    @abstractmethod
    def validate_candidate(
        self,
        candidate: FacilityCandidate
    ) -> Tuple[bool, List[str], float]:
        """
        验证单个候选设施（调用三层验证）

        Args:
            candidate: 候选设施

        Returns:
            (是否通过, 验证原因列表, 置信度评分)

        Example:
            >>> validate_candidate(toyota_ky_plant)
            (True, ["Keyword check passed", "LLM confirmed"], 0.92)
        """
        pass

    @abstractmethod
    def keyword_filter(self, candidate: FacilityCandidate) -> Tuple[bool, str]:
        """
        第1层验证：关键词过滤

        检查点:
        - 是否包含排除关键词（HQ, office, R&D）
        - 是否包含生产关键词（factory, plant, manufacturing）

        Args:
            candidate: 候选设施

        Returns:
            (是否通过, 原因)
        """
        pass

    @abstractmethod
    def address_validation(self, candidate: FacilityCandidate) -> Tuple[bool, str]:
        """
        第2层验证：地址验证

        检查点:
        - 地址是否与已知的总部地址重复
        - 使用Embedding计算相似度

        Args:
            candidate: 候选设施

        Returns:
            (是否通过, 原因)
        """
        pass

    @abstractmethod
    def llm_validation(self, candidate: FacilityCandidate) -> Tuple[bool, str, float]:
        """
        第3层验证：LLM二次验证

        使用LLM判断:
        - 是否生产实体产品？
        - 是否只是办公/管理功能？

        Args:
            candidate: 候选设施

        Returns:
            (是否通过, 原因, 置信度)
        """
        pass
```

**关键词定义**:
```python
# 排除关键词（硬性规则）
EXCLUDE_KEYWORDS = [
    'headquarters', 'hq', 'head office',
    'corporate office', 'administrative',
    'sales office', 'regional office',
    'r&d center', 'research center',
    'innovation center', 'technology center'
]

# 生产关键词（必须包含）
INCLUDE_KEYWORDS = [
    'factory', 'plant', 'manufacturing',
    'assembly', 'production', 'refinery',
    'mill', 'foundry', 'fabrication',
    'produces', 'manufactures'
]
```

---

### 4.5 EvidenceNode接口

```python
class EvidenceNodeInterface(BaseNode[AgentState]):
    """证据收集节点接口"""

    @abstractmethod
    def collect_evidence(
        self,
        facility: FacilityCandidate
    ) -> Tuple[List[str], List[str]]:
        """
        收集设施的证据

        Args:
            facility: 候选设施

        Returns:
            (证据URL列表, 证据文本片段列表)
        """
        pass

    @abstractmethod
    def validate_evidence_quality(self, url: str) -> float:
        """
        验证证据来源的质量

        评分标准:
        - 官方网站（公司域名）: 1.0
        - 维基百科: 0.9
        - 新闻网站: 0.8
        - 其他: 0.6

        Args:
            url: 证据URL

        Returns:
            质量评分(0-1)
        """
        pass
```

---

### 4.6 DeduplicationNode接口

```python
class DeduplicationNodeInterface(BaseNode[AgentState]):
    """去重节点接口"""

    @abstractmethod
    def deduplicate(
        self,
        facilities: List[ValidatedFacility]
    ) -> List[ValidatedFacility]:
        """
        去除重复设施

        策略:
        1. 地址完全相同 → 重复
        2. Embedding相似度 > 0.95 → 重复
        3. 经纬度距离 < 1km → 重复

        Args:
            facilities: 设施列表

        Returns:
            去重后的设施列表
        """
        pass

    @abstractmethod
    def calculate_similarity(
        self,
        facility1: ValidatedFacility,
        facility2: ValidatedFacility
    ) -> float:
        """
        计算两个设施的相似度

        使用Embedding计算 facility_name + full_address 的相似度

        Args:
            facility1: 设施1
            facility2: 设施2

        Returns:
            相似度(0-1)，>0.95视为重复
        """
        pass

    @abstractmethod
    def merge_duplicates(
        self,
        duplicates: List[ValidatedFacility]
    ) -> ValidatedFacility:
        """
        合并重复设施

        合并策略:
        - 保留证据最多的记录
        - 合并所有 evidence_urls
        - 取最高的 confidence_score

        Args:
            duplicates: 重复设施列表

        Returns:
            合并后的设施
        """
        pass
```

---

### 4.7 OutputNode接口

```python
class OutputNodeInterface(BaseNode[AgentState]):
    """输出节点接口"""

    @abstractmethod
    def format_output(self, state: AgentState) -> FinalOutput:
        """
        格式化最终输出

        Args:
            state: 当前状态

        Returns:
            最终输出对象
        """
        pass

    @abstractmethod
    def calculate_metrics(self, state: AgentState) -> ExecutionMetrics:
        """
        计算执行指标

        Args:
            state: 当前状态

        Returns:
            执行指标
        """
        pass
```

---

## 5. 模块间通信协议

### 5.1 数据传递格式

**所有节点遵循统一的接口**:
```python
def node_function(state: AgentState) -> AgentState:
    """
    节点函数签名

    Args:
        state: 当前AgentState

    Returns:
        更新后的AgentState
    """
    pass
```

**状态更新规则**:
1. **追加式更新**: 每个节点只更新与自己相关的字段
2. **不删除数据**: 保留前序节点的数据，便于调试
3. **记录日志**: 每个节点执行前后都记录日志

### 5.2 错误传递机制

```python
# 错误记录格式
error_entry = {
    "node": "search",                    # 发生错误的节点
    "timestamp": "2025-01-06T12:00:00Z", # 时间戳
    "error_type": "SearchAPIError",      # 异常类型
    "error_message": "API timeout",      # 错误信息
    "context": {                          # 上下文信息
        "query": "Toyota factories",
        "retry_count": 3
    }
}

# 添加到状态
state["errors"].append(error_entry)
```

**错误处理原则**:
- 错误不中断流程（除非是致命错误）
- 记录错误到 `state["errors"]`
- 节点可以选择性跳过错误项继续处理

### 5.3 节点调用示例

```python
# SearchNode 完整实现示例
def search_node(state: AgentState) -> AgentState:
    """搜索节点实现"""
    state["current_node"] = "search"
    state["execution_log"].append({
        "node": "search",
        "action": "start",
        "timestamp": datetime.now().isoformat()
    })

    try:
        # 构建查询
        queries = build_search_queries(state["company_name"])
        state["search_queries"] = queries

        # 执行搜索
        all_results = []
        for query in queries:
            try:
                results = perform_search(query)
                all_results.extend(results)
                state["metrics"]["search_api_calls"] += 1
            except SearchAPIError as e:
                state["errors"].append({
                    "node": "search",
                    "error": str(e),
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                })

        # 排序和去重
        state["search_results"] = rank_and_deduplicate(all_results)

        state["execution_log"].append({
            "node": "search",
            "action": "complete",
            "results_count": len(state["search_results"]),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        state["errors"].append({
            "node": "search",
            "error": str(e),
            "fatal": True,
            "timestamp": datetime.now().isoformat()
        })

    return state
```

---

## 6. 错误处理设计

### 6.1 自定义异常类型

```python
class FactoryAgentError(Exception):
    """基础异常类"""
    pass

class SearchAPIError(FactoryAgentError):
    """搜索API调用失败"""
    def __init__(self, message: str, query: str = None):
        self.message = message
        self.query = query
        super().__init__(self.message)

class LLMAPIError(FactoryAgentError):
    """LLM API调用失败"""
    def __init__(self, message: str, model: str = None):
        self.message = message
        self.model = model
        super().__init__(self.message)

class ExtractionError(FactoryAgentError):
    """数据提取失败"""
    pass

class ValidationError(FactoryAgentError):
    """验证失败"""
    pass

class ParseError(FactoryAgentError):
    """JSON解析失败"""
    pass

class NodeExecutionError(FactoryAgentError):
    """节点执行失败"""
    pass
```

### 6.2 错误恢复策略

| 错误类型 | 恢复策略 | 重试次数 | 说明 |
|---------|---------|---------|------|
| `SearchAPIError` | 重试 + 指数退避 | 3次 | 网络临时故障 |
| `LLMAPIError` | 切换备用模型 | 1次 | GPT-4o-mini→GPT-4.1 |
| `ParseError` | 跳过该条结果 | 0次 | 个别结果格式错误 |
| `ValidationError` | 记录并拒绝 | 0次 | 不符合约束 |
| `NodeExecutionError` | 终止流程 | 0次 | 致命错误 |

### 6.3 降级方案

```python
# 搜索API降级
def perform_search_with_fallback(query: str) -> List[SearchResult]:
    """搜索API降级策略"""
    try:
        return tavily_search(query)
    except SearchAPIError as e:
        logger.warning(f"Tavily API failed: {e}, falling back to Google")
        try:
            return google_search(query)
        except SearchAPIError:
            logger.error("All search APIs failed")
            return []

# LLM API降级
def call_llm_with_fallback(prompt: str) -> str:
    """LLM API降级策略"""
    try:
        return call_gpt4o_mini(prompt)
    except LLMAPIError as e:
        logger.warning(f"GPT-4o-mini failed: {e}, falling back to GPT-4.1")
        return call_gpt4(prompt)
```

---

## 7. 日志系统设计

### 7.1 日志配置

```python
import logging
import sys
from datetime import datetime

# 日志格式
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(
            f'logs/factory_agent_{datetime.now().strftime("%Y%m%d")}.log'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('factory_agent')
```

### 7.2 日志级别使用

| 级别 | 用途 | 示例 |
|------|------|------|
| `DEBUG` | 详细调试信息 | LLM提示词内容、API响应 |
| `INFO` | 常规信息 | 节点开始/结束、API调用 |
| `WARNING` | 警告信息 | API降级、部分结果失败 |
| `ERROR` | 错误信息 | API调用失败、验证失败 |
| `CRITICAL` | 致命错误 | 系统崩溃 |

### 7.3 关键日志点

```python
# 1. 节点开始
logger.info(f"[{node_name}] Starting execution for company: {company_name}")

# 2. API调用
logger.info(f"[API] Calling {api_name} with query: {query}")
logger.debug(f"[API] Request: {request_data}")
logger.debug(f"[API] Response: {response_data}")

# 3. 验证结果
logger.info(f"[Validation] {facility_name}: PASSED (confidence={score})")
logger.warning(f"[Validation] {facility_name}: REJECTED - {reason}")

# 4. 性能指标
logger.info(f"[Metrics] Tokens used: {tokens}, API calls: {calls}, Time: {time}s")

# 5. 错误
logger.error(f"[Error] {error_type}: {error_message}", exc_info=True)
```

---

## 8. API集成规范

### 8.1 LLM API调用封装

```python
from typing import Optional, Dict, Any
import time

class LLMClient:
    """LLM API客户端封装"""

    def __init__(
        self,
        primary_model: str = "gpt-4o-mini",
        fallback_model: str = "gpt-4.1",
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.timeout = timeout

    def call(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        调用LLM API with retry and fallback

        Args:
            prompt: 提示词
            temperature: 温度参数（0.0=确定性）
            max_tokens: 最大token数
            response_format: 响应格式（用于结构化输出）

        Returns:
            LLM响应文本

        Raises:
            LLMAPIError: 所有API都失败
        """
        for attempt in range(self.max_retries):
            try:
                response = self._call_primary(
                    prompt, temperature, max_tokens, response_format
                )
                return response
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.warning(f"Primary model failed after {self.max_retries} retries, trying fallback")
                    return self._call_fallback(
                        prompt, temperature, max_tokens, response_format
                    )
                wait_time = 2 ** attempt  # 指数退避
                logger.warning(f"Retry {attempt+1}/{self.max_retries} after {wait_time}s")
                time.sleep(wait_time)

        raise LLMAPIError("All LLM APIs failed")

    def _call_primary(self, prompt: str, **kwargs) -> str:
        """调用主模型（GPT-4o-mini）"""
        # Implementation with Azure OpenAI SDK
        pass

    def _call_fallback(self, prompt: str, **kwargs) -> str:
        """调用备用模型（GPT-4.1）"""
        # Implementation with Azure OpenAI SDK
        pass
```

### 8.2 搜索API调用封装

```python
class SearchClient:
    """搜索API客户端封装"""

    def __init__(
        self,
        api_key: str,
        api_type: str = "tavily",
        max_results: int = 10
    ):
        self.api_key = api_key
        self.api_type = api_type
        self.max_results = max_results

    def search(
        self,
        query: str,
        **kwargs
    ) -> List[SearchResult]:
        """
        执行搜索

        Args:
            query: 搜索查询

        Returns:
            搜索结果列表

        Raises:
            SearchAPIError: 搜索失败
        """
        try:
            if self.api_type == "tavily":
                return self._tavily_search(query, **kwargs)
            elif self.api_type == "google":
                return self._google_search(query, **kwargs)
            else:
                raise ValueError(f"Unsupported API type: {self.api_type}")
        except Exception as e:
            raise SearchAPIError(f"Search failed: {e}", query=query)

    def _tavily_search(self, query: str, **kwargs) -> List[SearchResult]:
        """Tavily搜索实现"""
        from tavily import TavilyClient
        client = TavilyClient(api_key=self.api_key)
        response = client.search(query, max_results=self.max_results)

        return [
            SearchResult(
                url=result['url'],
                title=result.get('title', ''),
                snippet=result.get('content', ''),
                relevance_score=result.get('score', 0.5)
            )
            for result in response.get('results', [])
        ]

    def _google_search(self, query: str, **kwargs) -> List[SearchResult]:
        """Google搜索实现（备用）"""
        # Implementation
        pass
```

### 8.3 重试和超时策略

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError))
)
def call_api_with_retry(api_func, *args, **kwargs):
    """带重试的API调用装饰器"""
    return api_func(*args, **kwargs)
```

---

## 9. 性能优化设计

### 9.1 并行处理

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any

def parallel_process(
    items: List[Any],
    process_func: Callable,
    max_workers: int = 5
) -> List[Any]:
    """
    并行处理列表项

    Args:
        items: 待处理项列表
        process_func: 处理函数
        max_workers: 最大并发数

    Returns:
        处理结果列表
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_func, item): item
            for item in items
        }

        for future in as_completed(future_to_item):
            try:
                result = future.result()
                if result:  # 过滤None结果
                    results.append(result)
            except Exception as e:
                logger.error(f"Parallel processing error: {e}")

    return results

# 使用示例：并行提取
extracted = parallel_process(
    search_results[:20],  # 限制前20条结果
    lambda r: extract_from_result(r, company_name),
    max_workers=5
)
```

### 9.2 缓存机制

```python
import hashlib
import json
import os
from typing import Optional, Any, Dict

class ResultCache:
    """结果缓存类"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "results.json")
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """加载缓存"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        """设置缓存"""
        self.cache[key] = value
        self._save_cache()

    def _save_cache(self):
        """保存缓存到文件"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """生成缓存键（MD5哈希）"""
        content = json.dumps(
            {"args": args, "kwargs": kwargs},
            sort_keys=True,
            default=str
        )
        return hashlib.md5(content.encode()).hexdigest()

# 使用示例
cache = ResultCache()
cache_key = ResultCache.make_key("Toyota Motor Corporation")
cached_result = cache.get(cache_key)

if cached_result:
    logger.info("Using cached result")
    return cached_result
else:
    result = expensive_search_operation()
    cache.set(cache_key, result)
    return result
```

### 9.3 Token优化

```python
def optimize_prompt_length(content: str, max_length: int = 4000) -> str:
    """
    优化提示词长度（截断过长内容）

    Args:
        content: 原始内容
        max_length: 最大长度（字符数）

    Returns:
        优化后的内容
    """
    if len(content) <= max_length:
        return content

    # 智能截断：保留开头和结尾
    half = max_length // 2
    return content[:half] + "\n\n...[content truncated]...\n\n" + content[-half:]


# Token计数
try:
    import tiktoken

    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """计算文本的token数"""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

except ImportError:
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """简单估算（1 token ≈ 4 characters）"""
        return len(text) // 4
```

---

## 10. 测试策略

### 10.1 单元测试

```python
import pytest
from unittest.mock import patch, MagicMock
from models import SearchResult, FacilityCandidate
from nodes.search import SearchNode
from nodes.validation import ValidationNode

class TestSearchNode:
    """搜索节点单元测试"""

    @pytest.fixture
    def search_node(self):
        return SearchNode(api_key="test_key")

    def test_build_search_queries(self, search_node):
        """测试查询构建"""
        queries = search_node.build_search_queries("Toyota Motor Corporation")

        # 验证查询数量
        assert len(queries) >= 3

        # 验证包含公司名
        assert all("Toyota" in q for q in queries)

        # 验证包含排除词
        has_exclude = any("-headquarters" in q or "-office" in q for q in queries)
        assert has_exclude

    @patch('nodes.search.tavily_search')
    def test_perform_search(self, mock_tavily, search_node):
        """测试搜索执行（Mock API）"""
        # Mock返回值
        mock_tavily.return_value = [
            SearchResult(
                url="https://example.com/factory",
                title="Test Factory",
                snippet="A manufacturing plant producing cars",
                relevance_score=0.9
            )
        ]

        results = search_node.perform_search("Toyota factory")

        assert len(results) > 0
        assert results[0].url == "https://example.com/factory"
        mock_tavily.assert_called_once()


class TestValidationNode:
    """验证节点单元测试"""

    @pytest.fixture
    def validation_node(self):
        return ValidationNode()

    def test_keyword_filter_reject_hq(self, validation_node):
        """测试关键词过滤拒绝HQ"""
        candidate = FacilityCandidate(
            facility_name="Toyota Headquarters",
            facility_type="office building",
            full_address="1 Toyota Way, Tokyo, Japan",
            country="Japan",
            city="Tokyo",
            evidence_snippet="Toyota's corporate headquarters in Tokyo",
            source_url="https://example.com"
        )

        passed, reason = validation_node.keyword_filter(candidate)

        assert not passed
        assert "headquarters" in reason.lower()

    def test_keyword_filter_accept_factory(self, validation_node):
        """测试关键词过滤接受工厂"""
        candidate = FacilityCandidate(
            facility_name="Toyota Kentucky Plant",
            facility_type="assembly plant",
            full_address="Georgetown, KY, USA",
            country="USA",
            city="Georgetown",
            evidence_snippet="Assembly plant producing Camry vehicles with 8000 employees",
            source_url="https://example.com"
        )

        passed, reason = validation_node.keyword_filter(candidate)

        assert passed
        assert "production" in reason.lower() or "manufacturing" in reason.lower()

    def test_validate_candidate_full_pipeline(self, validation_node):
        """测试完整验证流程"""
        candidate = FacilityCandidate(
            facility_name="Shell Pernis Refinery",
            facility_type="refinery",
            full_address="Pernis, Rotterdam, Netherlands",
            country="Netherlands",
            city="Rotterdam",
            evidence_snippet="Europe's largest refinery, processing crude oil into fuels",
            source_url="https://shell.com/pernis"
        )

        passed, reasons, confidence = validation_node.validate_candidate(candidate)

        assert passed
        assert confidence >= 0.7
        assert len(reasons) > 0
```

### 10.2 集成测试

```python
import pytest
from agent import FactoryAgent

class TestEndToEnd:
    """端到端集成测试"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_toyota(self):
        """测试完整流程：Toyota"""
        agent = FactoryAgent()
        result = agent.run("Toyota Motor Corporation")

        # 验证输出结构
        assert result.company_name == "Toyota Motor Corporation"
        assert result.total_count > 0
        assert len(result.facilities) > 0

        # 验证约束合规性
        for facility in result.facilities:
            # 不包含HQ关键词
            assert "headquarters" not in facility.facility_name.lower()
            assert "hq" not in facility.facility_name.lower()

            # 必须有证据链接
            assert len(facility.evidence_urls) > 0

            # 置信度>=0.7
            assert facility.confidence_score >= 0.7

            # 验证状态必须是PASSED
            assert facility.validation_status == "passed"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_shell(self):
        """测试完整流程：Shell"""
        agent = FactoryAgent()
        result = agent.run("Shell")

        assert result.company_name == "Shell"
        assert result.total_count > 0

        # 验证有炼油厂
        has_refinery = any(
            "refinery" in f.facility_type.lower()
            for f in result.facilities
        )
        assert has_refinery

    @pytest.mark.integration
    def test_empty_company(self):
        """测试空公司名"""
        agent = FactoryAgent()

        with pytest.raises(ValueError):
            agent.run("")

    @pytest.mark.integration
    def test_nonexistent_company(self):
        """测试不存在的公司"""
        agent = FactoryAgent()
        result = agent.run("NonexistentCompanyXYZ123")

        # 应该返回空结果，不应该崩溃
        assert result.total_count == 0
        assert len(result.facilities) == 0
```

### 10.3 测试用例定义

| 测试ID | 输入 | 预期输出 | 验证点 |
|--------|------|---------|--------|
| TC001 | Toyota Motor Corporation | ≥10个设施 | 无HQ，有证据URL，置信度≥0.7 |
| TC002 | Shell | ≥5个炼油厂 | facility_type包含refinery |
| TC003 | 空字符串 | ValueError异常 | 输入验证 |
| TC004 | 不存在的公司 | 0个设施 | 优雅处理，不崩溃 |
| TC005 | "Apple Inc." | 0个设施或很少 | 软件公司，很少制造设施 |

---

## 附录

### A. 配置文件示例

```python
# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """配置类"""

    # LLM配置
    primary_llm_model: str = "gpt-4o-mini"
    fallback_llm_model: str = "gpt-4.1"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000

    # 搜索配置
    search_api_type: str = "tavily"
    search_max_results: int = 10
    max_search_queries: int = 5

    # 验证配置
    min_confidence_score: float = 0.7
    similarity_threshold: float = 0.95

    # 性能配置
    max_parallel_workers: int = 5
    enable_cache: bool = True
    cache_dir: str = ".cache"

    # API密钥（从环境变量读取）
    tavily_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    @classmethod
    def from_env(cls):
        """从环境变量创建配置"""
        import os
        return cls(
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
```

### B. 项目文件结构

```
factory-agent/
├── src/
│   ├── __init__.py
│   ├── agent.py                 # 主Agent类
│   ├── models.py                # 数据模型（Pydantic）
│   ├── state.py                 # AgentState定义
│   ├── config.py                # 配置
│   ├── nodes/                   # 节点实现
│   │   ├── __init__.py
│   │   ├── base.py              # BaseNode
│   │   ├── search.py            # SearchNode
│   │   ├── extract.py           # ExtractionNode
│   │   ├── validate.py          # ValidationNode
│   │   ├── evidence.py          # EvidenceNode
│   │   ├── dedup.py             # DeduplicationNode
│   │   └── output.py            # OutputNode
│   ├── utils/                   # 工具函数
│   │   ├── __init__.py
│   │   ├── llm_client.py        # LLM API客户端
│   │   ├── search_client.py     # 搜索API客户端
│   │   ├── cache.py             # 缓存工具
│   │   └── helpers.py           # 辅助函数
│   └── prompts/                 # 提示词模板
│       ├── __init__.py
│       ├── extraction.py        # 提取提示词
│       └── validation.py        # 验证提示词
├── tests/                       # 测试文件
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_nodes.py
│   ├── test_integration.py
│   └── conftest.py              # Pytest配置
├── docs/                        # 文档
│   └── TECHNICAL_DESIGN.md      # 本文档
├── examples/                    # 示例
│   ├── input_example.json
│   └── output_example.json
├── logs/                        # 日志目录
├── .cache/                      # 缓存目录
├── .env.example                 # 环境变量示例
├── requirements.txt             # 依赖
├── main.py                      # 入口文件
└── README.md                    # 项目说明
```

---

## 文档版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|---------|
| 1.0 | 2025-01-06 | Factory Agent Team | 初始版本：完整技术设计文档 |

---

**文档结束**
