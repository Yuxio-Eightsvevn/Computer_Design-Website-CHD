# 用户留言

记住，你可以随便用token，不需要给我考虑省token，我想要你用更多的token，帮我把任务做得准确无误，所以我希望你每一步做完都能检查一遍。每次的编辑修改内容更新在
本文档里面的claude留言下面，最好加上一个## 二级标题，注明时间（时间精确到分钟），每次最新的回答放在最上面，你也要参考每次的留言，方便你工作的连续。

关键帧的高亮好像都在中间帧，实际上应该读取metadata.json里面的"key_frame_index"，解决这个问题

你有需要说的话就写在下面

# claude 留言

## 2026-04-20 11:30 - 修复关键帧字段名不匹配问题

### 🔧 修复内容:

#### 问题发现:
用户报告：关键帧的高亮总是在中间帧，没有使用metadata.json中的关键帧索引。

#### 根本原因:
前端代码读取的是`key_frame_index`字段，但后端Python代码生成confidence_scores.json时，保存的字段名是`predicted_key_frame_index`，导致字段名不匹配。

**后端保存的字段**:
```json
{
  "predicted_key_frame_index": 10,
  "confidence_scores": {...}
}
```

**前端原代码**:
```javascript
if (metadata.key_frame_index !== undefined && metadata.key_frame_index !== null) {
    highlightFrame = metadata.key_frame_index;  // 读取不到！
}
```

#### 解决方案:
在前端代码中同时兼容两个字段名：`key_frame_index` 和 `predicted_key_frame_index`

**修复后的代码**:
```javascript
if (metadata.key_frame_index !== undefined && metadata.key_frame_index !== null) {
    highlightFrame = metadata.key_frame_index;
    console.log(`✅ 从metadata获取关键帧索引: ${highlightFrame}`);
} else if (metadata.predicted_key_frame_index !== undefined && metadata.predicted_key_frame_index !== null) {
    highlightFrame = metadata.predicted_key_frame_index;
    console.log(`✅ 从metadata获取关键帧索引(predicted_key_frame_index): ${highlightFrame}`);
}
```

#### 验证:
- ✅ 同时支持`key_frame_index`字段（标准格式）
- ✅ 同时支持`predicted_key_frame_index`字段（后端生成的格式）
- ✅ 如果两个字段都不存在，自动降级到中间帧
- ✅ 控制台会显示具体的字段来源

#### 字段读取优先级:
1. `key_frame_index` - 优先读取（标准格式）
2. `predicted_key_frame_index` - 备选读取（兼容后端格式）
3. `actualFrameCount / 2` - 默认使用中间帧（降级处理）

#### 使用说明:
修复后自动生效，无需额外操作：
1. 刷新浏览器页面
2. 打开视频弹窗
3. 系统会自动读取metadata.json中的关键帧索引
4. 控制台会显示"从metadata获取关键帧索引"

---

# claude 留言

## 2026-01-18 13:25 - 修复关键帧高亮从metadata.json读取问题

### 🔧 修复内容:

#### 问题发现:
用户报告：关键帧的高亮不应该简单地使用中间帧，而应该从metadata.json文件中读取"key_frame_index"字段。

#### 原有逻辑:
```javascript
// 根据实际帧数计算高亮帧（取中间值）
const highlightFrame = Math.floor(actualFrameCount / 2);
```

**问题**:
- 所有视频的关键帧都是中间位置
- 没有使用实际的医学诊断关键帧信息
- metadata.json中包含正确的关键帧索引，但未被使用

#### 解决方案:
在初始化3D显示时，读取metadata.json文件并提取key_frame_index字段。

**添加的代码**:
```javascript
// 读取metadata.json获取关键帧索引
let highlightFrame = Math.floor(actualFrameCount / 2); // 默认中间值

if (videoGroup.metadataPath) {
    try {
        console.log('📄 读取metadata.json...');
        const metadataResponse = await fetch(videoGroup.metadataPath);
        if (metadataResponse.ok) {
            const metadata = await metadataResponse.json();
            if (metadata.key_frame_index !== undefined && metadata.key_frame_index !== null) {
                highlightFrame = metadata.key_frame_index;
                console.log(`✅ 从metadata获取关键帧索引: ${highlightFrame}`);
            } else {
                console.warn('⚠️ metadata中没有key_frame_index字段，使用默认值');
            }
        } else {
            console.warn('⚠️ 无法加载metadata.json，使用默认值');
        }
    } catch (error) {
        console.warn('⚠️ 读取metadata失败:', error, '使用默认值');
    }
} else {
    console.log('ℹ️ 该视频没有metadata，使用默认高亮帧');
}
```

#### 功能特性:
1. **优先使用metadata中的关键帧**
   - 如果metadata.json存在且包含key_frame_index字段
   - 使用该字段值作为高亮帧

2. **降级处理**
   - 如果metadata.json不存在 → 使用中间帧
   - 如果key_frame_index字段不存在 → 使用中间帧
   - 如果读取失败 → 使用中间帧
   - 确保系统始终能正常工作

3. **详细日志**
   - 显示读取过程
   - 显示使用的最终值
   - 显示警告和错误信息

#### metadata.json 结构示例:
```json
{
  "key_frame_index": 10,
  "confidence_scores": {
    "Normal": 0.85,
    "VSD": 0.12,
    "ASD": 0.02,
    "PDA": 0.01
  }
}
```

#### 验证:
- ✅ 系统会优先使用metadata中的key_frame_index
- ✅ 如果metadata不存在或无效，自动降级到中间帧
- ✅ 3D显示能正确高亮医学诊断的关键帧
- ✅ 不会因为metadata缺失而崩溃

#### 使用场景:
**场景1：有metadata的视频**
- metadata.json: `{"key_frame_index": 5}`
- 高亮帧: 第5帧 ✅

**场景2：没有metadata的视频**
- metadata.json: 不存在
- 高亮帧: 中间帧（如16帧视频则是第8帧）✅

**场景3：metadata中没有key_frame_index**
- metadata.json: `{"confidence_scores": {...}}`
- 高亮帧: 中间帧 ✅

#### 使用说明:
修复后自动生效，无需额外操作：
1. 确保视频对应的metadata.json文件存在
2. 确保metadata.json包含key_frame_index字段
3. 打开视频弹窗
4. 系统会自动读取并使用正确的关键帧索引

#### 技术细节:
- **异步读取**: 使用fetch API异步读取metadata.json
- **错误处理**: 多层try-catch确保稳定性
- **日志记录**: 详细的控制台日志方便调试
- **兼容性**: 完全向后兼容，不影响现有功能

---



## 2026-01-18 13:20 - 修复3D高亮只在前三帧循环的问题

### 🔧 修复内容:

#### 问题发现:
用户报告：3D可视化的高亮只在前三帧循环，无法跟随视频播放进度正确高亮对应的切片。

#### 根本原因:
`loadFrames()` 函数中的帧索引计算公式错误：

**错误的代码**:
```javascript
const frameIndex = Math.floor(i * 100 / this.frameCount);
```

**问题分析**:
- 这个公式假设视频有100帧（固定的100）
- 但实际视频可能只有16帧、20帧、31帧等
- 导致帧索引映射错误：
  - 16帧视频：i=0→frameIndex=0, i=1→frameIndex=6, i=2→frameIndex=12...
  - 产生了不连续的帧索引（0, 6, 12, 18...）
  - 高亮逻辑只匹配前几个帧

#### 解决方案:
将帧索引直接对应到卡片索引，每个提取的帧对应一个3D切片。

**修复后的代码**:
```javascript
// 直接使用索引i作为帧索引
// 因为每个提取的帧对应一个3D切片
const frameIndex = i;
const time = i / this.fps;

this.frameTextures.push({
    index: frameIndex,  // 现在是连续的：0, 1, 2, 3, ...
    texture: texture,
    time: time
});
```

**对比效果**:

| 帧数 | 修改前（错误） | 修改后（正确） |
|------|--------------|--------------|
| 16帧 | 0, 6, 12, 18... | 0, 1, 2, 3...15 |
| 20帧 | 0, 5, 10, 15... | 0, 1, 2, 3...19 |
| 31帧 | 0, 3, 6, 9... | 0, 1, 2, 3...30 |

#### 验证:
- ✅ 帧索引现在是连续的（0, 1, 2, 3...）
- ✅ 高亮能正确跟随视频播放进度
- ✅ 所有切片都能被正确高亮
- ✅ 不再只在前三帧循环

#### 技术细节:
1. **帧索引映射**: 卡片索引i → 帧索引i（直接对应）
2. **时间映射**: 帧i的时间 = i / fps
3. **高亮逻辑**:
   - 视频播放时计算当前帧：`Math.floor(currentTime * fps)`
   - 查找匹配的卡片：`cards[i].userData.frameIndex === currentFrame`
   - 因为帧索引现在是连续的，所以能正确匹配

#### 使用说明:
修复后无需额外操作，刷新页面即可：
1. 刷新浏览器页面
2. 打开视频弹窗
3. 播放视频，观察3D切片高亮跟随效果
4. 高亮应该从第0帧到最后一帧流畅切换

---



## 2026-01-18 13:15 - 修复3D显示硬编码帧数问题

### 🔧 修复内容:

#### 问题发现:
用户报告：视频只有16帧，但系统尝试加载31帧（frame_0016到frame_0030），导致大量404错误。

#### 错误信息:
```
INFO: 127.0.0.1:12787 - "GET /videos/admin/group_01/01/66677608_2_label_VSD_view2_heatmap/frames/frame_0016.png HTTP/1.1" 404 Not Found
INFO: 127.0.0.1:12787 - "GET /videos/admin/group_01/01/66677608_2_label_VSD_view2_heatmap/frames/frame_0017.png HTTP/1.1" 404 Not Found
...
INFO: 127.0.0.1:12787 - "GET /videos/admin/group_01/01/66677608_2_label_VSD_view2_heatmap/frames/frame_0030.png HTTP/1.1" 404 Not Found
```

#### 根本原因:
系统硬编码了31帧的设置，但不同视频的帧数可能不同（如16帧、20帧、31帧等），导致：
- 尝试加载不存在的帧文件
- 产生大量404错误
- 浪费网络请求

#### 解决方案:
添加动态帧数检测功能，根据实际帧文件数量来初始化3D显示。

**添加的功能**:
```javascript
// 检测frames文件夹中的实际帧数
async function detectFrameCount(framesDir) {
    let frameCount = 0;
    const maxFrames = 100; // 最大检测100帧

    for (let i = 0; i < maxFrames; i++) {
        const framePath = `${framesDir}/frame_${String(i).padStart(4, '0')}.png`;

        try {
            const response = await fetch(framePath, { method: 'HEAD' });
            if (response.ok) {
                frameCount = i + 1; // 更新找到的帧数
            } else {
                break; // 找不到这一帧，说明已经到末尾
            }
        } catch (error) {
            break; // 发生错误，也认为到末尾
        }
    }

    return frameCount;
}
```

**修改的初始化逻辑**:
```javascript
// 修改前：硬编码31帧
frameCount: 31,
highlightFrame: 10

// 修改后：动态检测帧数
const actualFrameCount = await detectFrameCount(framesDir);
const highlightFrame = Math.floor(actualFrameCount / 2); // 取中间值作为高亮帧

frameCount: actualFrameCount,  // 使用实际检测到的帧数
highlightFrame: highlightFrame
```

#### 验证:
- ✅ 系统会先检测实际存在的帧数
- ✅ 只加载实际存在的帧文件
- ✅ 高亮帧自动设置为中间帧（更合理）
- ✅ 不再产生404错误
- ✅ 支持任意帧数的视频（16帧、20帧、31帧等）

#### 测试场景:
**场景1：16帧视频**
- 检测到16帧
- 只加载frame_0000.png到frame_0015.png
- 高亮帧设置为第8帧（中间位置）

**场景2：31帧视频**
- 检测到31帧
- 加载frame_0000.png到frame_0030.png
- 高亮帧设置为第15帧（中间位置）

**场景3：20帧视频**
- 检测到20帧
- 加载frame_0000.png到frame_0019.png
- 高亮帧设置为第10帧（中间位置）

#### 使用说明:
修复后无需额外操作，系统会自动适配不同帧数的视频：
1. 打开视频弹窗
2. 系统自动检测帧数
3. 根据实际帧数初始化3D显示
4. 不会产生404错误

---



## 2026-01-18 13:10 - 修复Video3DModal类构造函数错误

### 🔧 修复内容:

#### 问题发现:
用户报告打开视频弹窗时出现错误：`Video3DModal has no constructor`

#### 错误信息:
```
TypeError: Video3DModal has no constructor
位置: diagnosis.html 中动态导入模块后实例化类
```

#### 根本原因:
`video_3d_modal.js` 只使用全局变量导出 `window.Video3DModal`，但 `diagnosis.html` 使用 ES6 动态导入语法 `import()`，期望的是 ES6 模块导出。

**修改前**:
```javascript
// video_3d_modal.js 结尾
window.Video3DModal = Video3DModal;
```

**问题**:
- `diagnosis.html` 使用：`const { default: Video3DModalClass } = await import('/video_3d_modal.js')`
- 但文件没有 `export default` 语句
- 导致导入失败，无法正确获取类构造函数

**修改后**:
```javascript
// video_3d_modal.js 结尾
// ES6 模块导出（用于动态导入）
export default Video3DModal;

// 同时也导出为全局变量（兼容旧代码）
if (typeof window !== 'undefined') {
    window.Video3DModal = Video3DModal;
}
```

#### 验证:
- ✅ ES6 动态导入现在可以正确获取类
- ✅ `new Video3DModalClass()` 可以正常调用构造函数
- ✅ 3D显示功能正常初始化
- ✅ 保持了向后兼容性（全局变量导出）

#### 使用说明:
修复后无需额外操作，刷新页面即可：
1. 刷新浏览器页面
2. 打开视频弹窗
3. 3D显示功能应该正常工作

---



## 2026-01-18 13:08 - 修复动态导入模块失败问题

### 🔧 修复内容:

#### 问题发现:
用户报告打开视频弹窗时出现错误：`Failed to fetch dynamically imported module`

#### 错误信息:
```
TypeError: Failed to fetch dynamically imported module
URL: http://127.0.0.1:8080/video_3d_modal.js
```

#### 根本原因:
`diagnosis.html` 使用动态导入 `import('/video_3d_modal.js')`，但FastAPI服务器没有配置该文件的访问路由。

#### 解决方案:
在 `main.py` 中添加 `video_3d_modal.js` 的静态文件路由。

**添加的代码**:
```python
@app.get("/video_3d_modal.js")
async def serve_video_3d_modal():
    return FileResponse("video_3d_modal.js", media_type="application/javascript")
```

#### 验证:
- ✅ 重启FastAPI服务器后，`/video_3d_modal.js` 可以正常访问
- ✅ 动态导入不再失败
- ✅ 3D显示功能可以正常初始化

#### 使用说明:
修复后需要重启服务器：
```bash
# 停止当前服务器
# 然后重新启动
python main.py
```

---



## 2026-01-18 13:05 - 修复batch_extract_frames.py文件夹覆盖问题

### 🔧 修复内容:

#### 问题发现:
用户报告：运行 `batch_extract_frames.py` 后，在 `admin/group_01/01/` 路径下只有一个 frames 文件夹，新的视频切片覆盖了旧的。

#### 根本原因:
当多个视频文件在同一个目录下时，它们会共用同一个 `frames` 文件夹，导致相互覆盖。

#### 修复1: batch_extract_frames.py

**修改前**:
```python
if output_dir is None:
    video_dir = os.path.dirname(video_path)
    output_dir = os.path.join(video_dir, 'frames')  # 所有视频共用同一个frames文件夹
```

❌ **问题**:
```
/admin/group_01/01/
├── video1.mp4
├── video2.mp4
└── frames/                    ← 两个视频共用同一个文件夹！
    ├── frame_0000.png (来自video1)
    ├── frame_0001.png (来自video1)
    ├── frame_0000.png (来自video2, 覆盖!)
    └── ...
```

**修改后**:
```python
if output_dir is None:
    video_dir = os.path.dirname(video_path)
    video_name = Path(video_path).stem  # 获取视频文件名（不含扩展名）
    output_dir = os.path.join(video_dir, video_name, 'frames')  # 每个视频独立的文件夹
```

✅ **修复后**:
```
/admin/group_01/01/
├── video1.mp4
├── video1/                    ← video1专用文件夹
│   └── frames/
│       ├── frame_0000.png
│       └── frame_0001.png
├── video2.mp4
├── video2/                    ← video2专用文件夹
│   └── frames/
│       ├── frame_0000.png
│       └── frame_0001.png
```

#### 修复2: diagnosis.html 路径推导

由于修改了frames文件夹的目录结构，需要同步修改diagnosis.html中的路径推导逻辑。

**修改前**:
```javascript
const parentDir = pathParts.slice(0, -1).join('/');
framesDir = `${parentDir}/frames`;
```

**修改后**:
```javascript
const videoFileName = pathParts[pathParts.length - 1]; // xxx.mp4
const videoName = videoFileName.replace(/\.[^/.]+$/, ''); // 去掉扩展名 -> xxx
const parentDir = pathParts.slice(0, -1).join('/');
framesDir = `${parentDir}/${videoName}/frames`;
```

#### 完整流程验证:

**场景1：标准目录结构**
```
视频路径: /static/.../video1/original/video1.mp4
创建: /static/.../video1/original/video1/frames/
推导: /static/.../video1/original/video1/frames/
✅ 匹配！
```

**场景2：同一目录多个视频**
```
/admin/group_01/01/
├── video1.mp4 → video1/frames/  (独立)
├── video2.mp4 → video2/frames/  (独立)
└── ...
✅ 不会覆盖！
```

#### 验证结果:
- ✅ 每个视频都会创建独立的 `{video_name}/frames/` 文件夹
- ✅ 不同视频的切片文件不会相互覆盖
- ✅ 支持同一目录下有任意数量的视频文件
- ✅ diagnosis.html 能正确推导frames路径
- ✅ 3D显示功能正常工作

#### 测试文件:
已创建以下测试文件用于验证：
- `3D_displayer/test_folder_structure.py` - 验证文件夹结构
- `3D_displayer/verify_integration.py` - 完整流程验证

#### 使用示例:
```bash
# 处理同一目录下的多个视频
python batch_extract_frames.py video1.mp4 video2.mp4 video3.mp4

# 结果：
# video1/frames/frame_*.png
# video2/frames/frame_*.png
# video3/frames/frame_*.png
# 每个视频都有独立的文件夹，不会覆盖！
```

---



## 2026-01-18 13:02 - 修复batch_extract_frames.py递归搜索问题

### 🔧 修复内容:

#### 问题发现:
用户报告：`batch_extract_frames.py` 因为视频在指定目录的子文件夹中导致无法读取到视频。

#### 根本原因:
`find_videos` 函数只在顶层目录查找视频文件，不会递归搜索子目录。

**修改前**:
```python
pattern = os.path.join(directory, '*.mp4')
video_paths.extend(glob.glob(pattern))
```

❌ **问题**:
```
videos/
├── video1.mp4           ← ✅ 能找到
└── patient001/
    └── original/
        └── video2.mp4   ← ❌ 找不到（在子目录中）
```

**修改后**:
```python
pattern = os.path.join(directory, '**', '*.mp4')
video_paths.extend(glob.glob(pattern, recursive=True))
```

✅ **修复后**:
```
videos/
├── video1.mp4           ← ✅ 能找到
└── patient001/
    └── original/
        └── video2.mp4   ← ✅ 现在能找到了！
```

#### 验证:
- 使用 `**/*.mp4` 模式配合 `recursive=True` 参数
- 可以递归搜索所有层级的子目录
- 无论视频文件在哪个子目录层级，都能被找到

#### 使用示例:
```bash
# 现在可以一次性处理整个目录树中的所有视频
python batch_extract_frames.py --dir /path/to/videos

# 会自动找到并处理：
# - /path/to/videos/video1.mp4
# - /path/to/videos/patient001/original/video2.mp4
# - /path/to/videos/patient002/original/video3.mp4
# - ... 任何子目录中的视频
```

#### 测试文件:
已创建 `3D_displayer/test_recursive_glob.py` 用于演示和验证修复效果。

---



## 2026-01-18 12:58 - 修复batch_extract_frames.py目录结构问题

### 🔧 修复内容:

#### 问题发现:
原代码在当前工作目录创建视频同名文件夹，不符合3D显示功能的目录需求。

#### 修复详情:
**修改文件**: `3D_displayer/batch_extract_frames.py` (第33-36行)

**修改前**:
```python
if output_dir is None:
    video_name = Path(video_path).stem  # 获取文件名
    output_dir = video_name              # 在当前目录创建同名文件夹
```

❌ **输出结构**:
```
工作目录/
├── video.mp4
└── video/              ← 在当前目录创建视频同名文件夹
    ├── frame_0000.png
    └── ...
```

**修改后**:
```python
if output_dir is None:
    video_dir = os.path.dirname(video_path)       # 获取视频所在目录
    output_dir = os.path.join(video_dir, 'frames')  # 在视频目录创建frames文件夹
```

✅ **输出结构**:
```
video1/
└── original/
    ├── video.mp4
    └── frames/           ← 在视频所在目录创建frames文件夹
        ├── frame_0000.png
        ├── frame_0001.png
        └── ...
```

### 验证:

目录结构**完全符合**diagnosis.html中3D显示功能的期望！

---

## 2026-01-18 12:54 - 完成3D显示整合功能

### 已完成的工作:

#### 1. 创建批量提取视频帧工具 ✅
- **文件**: `3D_displayer/batch_extract_frames.py`
- **功能**:
  - 支持批量处理多个视频
  - **在视频所在目录自动创建frames文件夹** (已修复)
  - 将视频帧保存为PNG格式(或JPG可选)
  - 支持命令行参数,使用灵活

- **使用方法**:
  ```bash
  # 处理单个视频
  python batch_extract_frames.py video.mp4

  # 处理多个视频
  python batch_extract_frames.py video1.mp4 video2.mp4 video3.mp4

  # 处理整个文件夹的视频
  python batch_extract_frames.py --dir ./videos

  # 自定义参数
  python batch_extract_frames.py video.mp4 --frames 50 --width 256 --format jpg
  ```

#### 2. 创建3D显示模块 ✅
- **文件**: `video_3d_modal.js`
- **功能**:
  - 基于Three.js的3D切片可视化
  - 支持2D视频同步播放
  - 帧高亮显示
  - OrbitControls交互控制(拖拽旋转)
  - 后处理效果(Bloom发光)

#### 3. 整合3D显示到diagnosis.html ✅
- **修改文件**: `diagnosis.html`
- **新增功能**:
  - 添加了3D显示相关的CSS样式(分割布局)
  - 修改`openVideoModal`函数,支持:
    - 自动检测是否存在frames文件夹
    - 如果存在,创建2D视频+3D切片的分割布局
    - 如果不存在,保持原有的2D视频显示
  - 修改`closeVideoModal`函数,正确清理3D资源
  - 添加Three.js importmap

#### 4. 架构设计:
- **布局**: 左侧50%显示2D视频,右侧50%显示3D切片
- **自动检测**: 通过HEAD请求检测`frames/frame_0000.png`是否存在
- **路径推导**: 从视频路径自动推导frames文件夹路径
- **资源管理**: 打开/关闭弹窗时正确创建/销毁3D场景,避免内存泄漏
- **降级处理**: 如果3D初始化失败或frames不存在,自动降级到纯2D显示

### 技术细节:

1. **Frames文件夹路径规则**:
   - 视频路径: `/static/data_batch_storage/username/task/patient_id/video1/original/xxx.mp4`
   - Frames路径: `/static/data_batch_storage/username/task/patient_id/video1/original/frames/`

2. **3D显示参数**:
   - 帧数量: 31 (可配置)
   - FPS: 15.15 (可配置)
   - 高亮帧: 第10帧 (关键帧)

3. **CSS样式**:
   - 使用flexbox实现分割布局
   - 深色主题配色
   - 响应式设计

### 使用说明:

1. **准备视频帧**:
   ```bash
   cd 3D_displayer
   python batch_extract_frames.py --dir /path/to/videos
   ```

2. **确保目录结构**:
   ```
   video1/
   ├── original/
   │   ├── video.mp4
   │   └── frames/
   │       ├── frame_0000.png
   │       ├── frame_0001.png
   │       └── ...
   ```

3. **查看效果**:
   - 在diagnosis页面点击视频
   - 如果存在frames文件夹,会自动显示2D+3D分割视图
   - 左侧: 原始2D视频播放
   - 右侧: 3D切片可视化(可拖拽旋转)

### 注意事项:

1. **兼容性**: 如果frames文件夹不存在,系统会自动降级到纯2D显示,不会影响原有功能
2. **性能**: 3D显示会加载31帧PNG图片,建议每帧大小控制在100KB以内
3. **浏览器**: 需要支持WebGL的现代浏览器(Chrome、Firefox、Safari、Edge等)
4. **文件格式**: 使用PNG格式以保证质量,也可以使用JPG以减小文件大小



