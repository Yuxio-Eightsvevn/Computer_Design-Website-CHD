# 大模型配置指南

本文档详细说明如何配置大模型API，以便在教育模式中生成AI分析报告。

---

## 一、参数说明

### 1. 必需参数

#### 显示名称 (display_name)

- **说明**: 在界面上显示的模型名称，用于识别不同的模型配置
- **格式**: 字符串，建议格式："服务商名 模型名"
- **限制**: 最多50个字符
- **示例**: 
  - "智谱AI GLM-4"
  - "DeepSeek Chat"
  - "Moonshot V1 8K"
  - "OpenAI GPT-4"

#### Base URL (base_url)

- **说明**: API服务的基础URL地址，- **格式**: 完整的HTTP/HTTPS URL，- **重要**: 
  - **智谱AI** 必须使用: `https://open.bigmodel.cn/api/paas/v4` (注意：不是 `https://open.bigmodel.cn/api`)
  - 其他服务通常不需要 `/paas` 路径
- **示例**:
  - 智谱AI: `https://open.bigmodel.cn/api/paas/v4`
  - OpenAI: `https://api.openai.com`
  - DeepSeek: `https://api.deepseek.com`
  - Moonshot: `https://api.moonshot.cn`

#### API Key (api_key)

- **说明**: 访问API的密钥，用于身份验证
- **格式**: 字符串，由服务提供商提供
- **安全提示**: 
  - 请妥善保管API Key，不要泄露给他人
  - 系统会自动脱敏显示（只显示前4位和后4位）
- **示例**:
  - 智谱AI: `abcdefghij1234567890.abcdefghij-1234567890`
  - OpenAI: `sk-1234567890abcdefghijklmnop`
  - DeepSeek: `sk-1234567890abcdefghijklmnop`

#### 模型名称 (model)

- **说明**: 调用API时使用的模型标识符，不同的模型有不同的能力和价格
- **格式**: 字符串，由服务提供商定义
- **常见选项**:
  - 智谱AI: `glm-4`, `glm-4-flash`, `glm-3-turbo`
  - OpenAI: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
  - DeepSeek: `deepseek-chat`, `deepseek-coder`
  - Moonshot: `moonshot-v1-8k`, `moonshot-v1-32k`

---

### 2. 可选参数

#### 描述 (description)

- **说明**: 对模型的简要描述，帮助识别模型特点
- **格式**: 字符串，最多200字符
- **示例**: 
  - "智谱AI最新GLM-4模型，支持128K上下文，新用户送30元"
  - "DeepSeek对话模型，免费1元额度"
  - "OpenAI GPT-4，最强性能，按使用量计费"

---

## 二、主流服务商配置示例

### 1. 智谱AI (BigModel) - 推荐

**特点**：
- 🎁 新用户送30元额度
- 🇨🇳 国内服务，响应速度快
- 📊 GLM-4性能优秀，支持128K上下文
- 💰 按使用量计费，价格合理

**配置示例**：
```json
{
  "display_name": "智谱AI GLM-4",
  "base_url": "https://open.bigmodel.cn/api/paas/v4",
  "api_key": "你的API Key",
  "model": "glm-4",
  "description": "智谱AI最新GLM-4模型，新用户送30元额度"
}
```

**⚠️ 重要提示 - 智谱AI的Base URL**：
- ✅ **正确**: `https://open.bigmodel.cn/api/paas/v4`
- ❌ **错误**: `https://open.bigmodel.cn/api` (缺少 `/paas/v4`)
- ❌ **错误**: `https://open.bigmodel.cn` (缺少完整路径)

**为什么？**
- 智谱AI的API需要完整的路径 `/api/paas/v4`
- 如果只输入 `https://open.bigmodel.cn/api`，系统会自动补全为 `https://open.bigmodel.cn/api/paas/v4`
- 但为了避免混淆，建议直接输入完整路径

**⚠️ 重要提示**：
- **智谱AI的Base URL必须是**: `https://open.bigmodel.cn/api/paas/v4`
- **不要使用**: `https://open.bigmodel.cn/api` （这会导致404错误）

**⚠️ 重要提示**：
- **智谱AI的Base URL必须是**: `https://open.bigmodel.cn/api/paas/v4`
- **不要使用**: `https://open.bigmodel.cn/api` （这会导致404错误）

**重要提示**：
- ⚠️ **Base URL必须是**: `https://open.bigmodel.cn/api/paas/v4` （注意：包含 `/api/paas/v4`）
- ❌ **错误的Base URL**: `https://open.bigmodel.cn/api` （这会导致404错误）

**获取API Key步骤**：
1. 访问 https://open.bigmodel.cn/
2. 点击右上角"注册/登录"
3. 完成注册并登录
4. 进入控制台，点击"API密钥"
5. 点击"创建新的API Key"
6. 复制生成的API Key

**推荐模型**：
- `glm-4`: 最新版本，性能最佳
- `glm-4-flash`: 速度更快，成本更低

---

### 2. DeepSeek - 性价比之选

**特点**：
- 🎁 新用户送1元免费额度
- 💰 价格便宜，性价比高
- 🚀 响应速度快
- 🔧 支持代码和对话

**配置示例**：
```json
{
  "display_name": "DeepSeek Chat",
  "base_url": "https://api.deepseek.com",
  "api_key": "你的API Key",
  "model": "deepseek-chat",
  "description": "DeepSeek对话模型，免费1元额度"
}
```

**获取API Key步骤**：
1. 访问 https://platform.deepseek.com/
2. 点击"注册"
3. 完成注册并登录
4. 进入"API Keys"页面
5. 点击"创建新的API Key"
6. 复制生成的API Key

**推荐模型**：
- `deepseek-chat`: 通用对话模型
- `deepseek-coder`: 代码专用模型

---

### 3. Moonshot AI (Kimi) - 长文本专家

**特点**：
- 🎁 新用户送15元额度
- 📚 支持超长文本（最多128K tokens）
- 🇨🇳 国内服务
- 💡 适合需要处理大量文本的场景

**配置示例**：
```json
{
  "display_name": "Moonshot V1 8K",
  "base_url": "https://api.moonshot.cn",
  "api_key": "你的API Key",
  "model": "moonshot-v1-8k",
  "description": "Moonshot Kimi模型，新用户送15元"
}
```

**获取API Key步骤**：
1. 访问 https://platform.moonshot.cn/
2. 点击"注册/登录"
3. 完成注册并登录
4. 进入"API Key管理"
5. 点击"创建新的API Key"
6. 复制生成的API Key

**推荐模型**：
- `moonshot-v1-8k`: 8K上下文，通用场景
- `moonshot-v1-32k`: 32K上下文，中等长度文本
- `moonshot-v1-128k`: 128K上下文，超长文本

---

### 4. OpenAI - 行业标杆

**特点**：
- 🏆 GPT-4性能最强
- 🌍 全球服务
- 💡 功能最全面
- ⚠️ 无免费额度，需要付费

**配置示例**：
```json
{
  "display_name": "OpenAI GPT-4",
  "base_url": "https://api.openai.com",
  "api_key": "你的API Key",
  "model": "gpt-4",
  "description": "OpenAI GPT-4，最强性能"
}
```

**获取API Key步骤**：
1. 访问 https://platform.openai.com/
2. 点击"Sign up"注册
3. 完成注册并登录（需要国外手机号或邮箱）
4. 进入"API keys"页面
5. 点击"Create new secret key"
6. 复制生成的API Key（只显示一次，请妥善保管）

**推荐模型**：
- `gpt-4`: 最强性能
- `gpt-4-turbo`: 更快速度，更低成本
- `gpt-3.5-turbo`: 最便宜，适合简单任务

---

## 三、配置步骤

### 步骤1：获取API Key

1. 选择一个服务商（推荐智谱AI）
2. 访问对应的服务商网站
3. 注册并登录
4. 在控制台找到"API Key"或"API密钥"
5. 创建新的API Key并复制

### 步骤2：在教育管理后台添加模型

1. 以管理员身份登录系统
2. 进入"教育管理后台"（edu_admin.html）
3. 滚动到"大模型配置"卡片
4. 点击"增加模型"按钮
5. 填写模型信息：
   - **显示名称**：例如"智谱AI GLM-4"
   - **Base URL**：例如`https://open.bigmodel.cn/api`
   - **API Key**：粘贴从服务商获取的密钥
   - **模型名称**：例如`glm-4`
   - **描述**（可选）：例如"智谱AI最新GLM-4模型"
6. 点击"保存"

### 步骤3：测试连接

1. 在模型下拉框中选择刚添加的模型
2. 点击"测试连接"按钮
3. 等待测试结果
4. 如果显示"✅ 连接测试成功"，说明配置正确
5. 如果显示"❌ 连接失败"，请检查参数是否正确

### 步骤4：使用模型

配置完成后，当用户提交教育判读练习时：
1. 系统会自动使用选中的模型
2. 在后台异步调用大模型API
3. 生成详细的AI分析报告
4. 用户可以在报告页面查看大模型评价

---

## 四、常见问题

### Q1: API Key输入后看不到完整内容？

**A**: 这是正常的安全设计。系统会自动脱敏显示API Key，只显示前4位和后4位，中间用星号替代。例如：`abcd****wxyz`

### Q2: 测试连接失败怎么办？

**A**: 请按以下步骤检查：
1. **检查Base URL**: 确保URL正确，不要有多余的斜杠
2. **检查API Key**: 确保复制完整，没有多余空格
3. **检查模型名称**: 确保模型名称正确（区分大小写）
4. **检查网络**: 确保服务器能访问对应的API地址
5. **检查余额**: 确保账户有足够的余额或免费额度

### Q3: 如何选择合适的模型？

**A**: 建议：
- **首选**：智谱AI GLM-4（免费额度多，国内服务快）
- **备选**：DeepSeek Chat（价格便宜）
- **特殊需求**：Moonshot（需要处理长文本）或OpenAI（追求最强性能）

### Q4: 配置多个模型有什么用？

**A**: 
- 可以在不同模型间切换，对比效果
- 一个模型余额用完后切换到另一个
- 根据不同场景选择最合适的模型

### Q5: 如何删除或修改已配置的模型？

**A**: 
- **删除**: 在模型下拉框中选择要删除的模型，点击"删除模型"按钮
- **修改**: 选择模型后，点击"编辑模型"按钮，修改后保存

### Q6: API Key泄露了怎么办？

**A**: 
1. 立即登录服务商网站
2. 找到对应的API Key
3. 删除或禁用该Key
4. 创建新的API Key
5. 在系统中更新配置

### Q7: 系统支持哪些大模型？

**A**: 理论上支持所有兼容OpenAI API格式的大模型，包括：
- 智谱AI (BigModel)
- OpenAI
- DeepSeek
- Moonshot AI
- 通义千问 (Qwen)
- 百度文心一言
- 其他兼容OpenAI格式的API

---

## 五、费用说明

### 智谱AI (BigModel)
- **免费额度**: 新用户30元
- **GLM-4**: 约0.1元/千tokens
- **GLM-4-Flash**: 约0.01元/千tokens

### DeepSeek
- **免费额度**: 新用户1元
- **DeepSeek Chat**: 约0.001元/千tokens
- **DeepSeek Coder**: 约0.001元/千tokens

### Moonshot AI
- **免费额度**: 新用户15元
- **8K模型**: 约0.012元/千tokens
- **32K模型**: 约0.024元/千tokens
- **128K模型**: 约0.06元/千tokens

### OpenAI
- **免费额度**: 无
- **GPT-4**: 约0.3元/千tokens（输入）+ 0.6元/千tokens（输出）
- **GPT-3.5 Turbo**: 约0.01元/千tokens

---

## 六、最佳实践

### 1. 成本控制
- 优先使用有免费额度的服务（智谱AI、DeepSeek、Moonshot）
- 对于简单任务，使用更便宜的模型
- 定期检查使用量和余额

### 2. 性能优化
- 根据任务复杂度选择合适的模型
- 对于教育报告，中等复杂度的模型即可满足需求
- 不必一味追求最强性能

### 3. 安全管理
- 定期更换API Key
- 不要在公共场合分享API Key
- 为不同的应用创建不同的API Key

### 4. 备份策略
- 配置多个模型作为备份
- 一个模型失败时可以快速切换到另一个

---

## 七、技术支持

如有问题，请检查：
1. 本文档的常见问题部分
2. 服务商的官方文档
3. 系统的错误日志

---

*文档版本: 2026-04-04*
*最后更新: 创建配置指南*
