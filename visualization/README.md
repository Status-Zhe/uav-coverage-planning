# UAV Coverage Planner - 可视化界面

基于真实数据驱动的 UAV 路径规划 3D 可视化界面。

## ✨ 特性

- **真实数据驱动**: 从 `trajectory.json` 读取 Python 后端生成的真实路径数据
- **四种算法可视化**:
  - Boustrophedon (S形扫描)
  - Oblique (五向倾斜摄影)
  - Spiral (螺旋环绕)
  - Viewpoint (Box包裹视点)
- **实时动画**: UAV 沿真实路径飞行演示
- **交互式 3D**: 鼠标旋转/缩放视角

## 🚀 快速启动

### 方式 1: Python HTTP 服务器

```bash
cd visualization
python3 -m http.server 8080
# 浏览器访问 http://localhost:8080
```

### 方式 2: Node.js (若已安装)

```bash
cd visualization
npx serve
```

## 📂 数据结构

### trajectory.json 格式

```json
{
  "building": {
    "points": [[x, y, z], ...],
    "bounds": {
      "min": [x, y, z],
      "max": [x, y, z]
    }
  },
  "trajectories": {
    "boustrophedon": {
      "waypoints": [{"x": 0, "y": 0, "z": 35, "heading": 0, "gimbal_pitch": -90}, ...],
      "stats": {"waypoint_count": 48, "total_distance_m": 324.5, "estimated_time_min": 1.1}
    },
    "oblique": { ... },
    "spiral": { ... },
    "viewpoint": { ... }
  }
}
```

## 🔧 从 Python 生成数据

```python
from uav_planners import CoveragePipeline, MissionConfig
from uav_planners.models import Camera, PointCloud

# 创建点云和配置
building = PointCloud(...)
config = MissionConfig(...)

# 运行规划
pipeline = CoveragePipeline(config)
result = pipeline.run(building)

# 导出 JSON
result.export_json("visualization/trajectory.json")
```

## 📋 界面操作

1. 点击算法按钮切换不同路径
2. 点击"加载路径"加载选中算法的数据
3. 点击"播放"观看 UAV 沿路径飞行动画

## ⚠️ 注意事项

- 必须通过 HTTP 服务器访问（不能直接用 file:// 打开）
- 确保 `trajectory.json` 与 `index.html` 在同一目录
