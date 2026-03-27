<template>
  <div class="dashboard-container">
    <div class="header">
      <div class="header-content">
        <div class="logo">
          <h1>基于DeepSeek的数据自动化分类分级系统</h1>
          
        </div>
        <div class="nav-menu">
          <el-menu mode="horizontal" :default-active="currentRoute" router>
            <el-menu-item index="/dashboard">
              <el-icon><House /></el-icon>
              <span>首页</span>
            </el-menu-item>
            <el-menu-item index="/classify">
              <el-icon><Upload /></el-icon>
              <span>文件分析</span>
            </el-menu-item>
            <el-menu-item index="/knowledge">
              <el-icon><Box /></el-icon>
              <span>知识库</span>
            </el-menu-item>
            <el-menu-item index="/visualization">
              <el-icon><DataAnalysis /></el-icon>
              <span>可视化</span>
            </el-menu-item>
          </el-menu>
        </div>
      </div>
    </div>

    <div class="content">
      <el-row :gutter="20">
        <el-col :span="24">
          <div class="welcome-card">
            <h2>欢迎使用数据分类分级系统</h2>
            <p>本系统基于 DeepSeek-llm-7B-chat 大语言模型，能够自动对数据字段进行分类和分级，帮助您快速识别敏感数据。</p>
          </div>
        </el-col>
      </el-row>

      <el-row :gutter="20" class="stats-row">
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon blue"><el-icon><Document /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.totalProcessed.toLocaleString() }}</div>
              <div class="stat-label">已处理字段</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon teal"><el-icon><Folder /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.totalFiles }}</div>
              <div class="stat-label">已分析文件</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon orange"><el-icon><Clock /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.avgTime }}</div>
              <div class="stat-label">平均耗时</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon red"><el-icon><Warning /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.sensitiveCount }}</div>
              <div class="stat-label">敏感字段</div>
            </div>
          </el-card>
        </el-col>
      </el-row>

      <el-row :gutter="20">
        <el-col :span="12">
          <el-card class="chart-card">
            <template #header>
              <span>分类分布</span>
            </template>
            <div id="categoryChart" style="width: 100%; height: 300px"></div>
          </el-card>
        </el-col>
        <el-col :span="12">
          <el-card class="chart-card">
            <template #header>
              <span>分级分布</span>
            </template>
            <div id="gradingChart" style="width: 100%; height: 300px"></div>
          </el-card>
        </el-col>
      </el-row>

      <el-row :gutter="20" class="features-row">
        <el-col :span="8">
          <el-card class="feature-card">
            <el-icon class="feature-icon"><Upload /></el-icon>
            <h3>文件上传分析</h3>
            <p>支持上传 CSV、Excel 文件，自动分析表头和数据样本</p>
            <el-button type="primary" @click="$router.push('/classify')">开始分析</el-button>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card class="feature-card">
            <el-icon class="feature-icon"><Box /></el-icon>
            <h3>知识库管理</h3>
            <p>管理分类分级规则，支持行业特定规则配置</p>
            <el-button type="primary" @click="$router.push('/knowledge')">查看规则</el-button>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card class="feature-card">
            <el-icon class="feature-icon"><DataAnalysis /></el-icon>
            <h3>结果可视化</h3>
            <p>图表展示分类分级结果，支持导出和详情查看</p>
            <el-button type="primary" @click="$router.push('/visualization')">查看报告</el-button>
          </el-card>
        </el-col>
      </el-row>

      <el-row :gutter="20">
        <el-col :span="24">
          <el-card class="steps-card">
            <template #header>
              <span>快速开始指南</span>
            </template>
            <el-steps :active="0" align-center>
              <el-step title="上传文件" description="上传 CSV 或 Excel 文件" />
              <el-step title="选择行业" description="选择对应的行业类型" />
              <el-step title="开始分析" description="系统自动进行分类分级" />
              <el-step title="查看结果" description="查看和导出分析结果" />
            </el-steps>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onActivated } from 'vue'
import { useRouter } from 'vue-router'
import { House, Upload, Box, Document, Folder, Clock, Warning, DataAnalysis } from '@element-plus/icons-vue'
import * as echarts from 'echarts'

const router = useRouter()
const currentRoute = ref('/dashboard')

// 分类标签列表
const classificationLabels = [
  'ID类/主键ID', '结构类/分类代码', '结构类/产品代码', '结构类/企业代码', '结构类/标准代码',
  '属性类/名称标题', '属性类/类别标签', '属性类/描述文本', '属性类/技能标签', '属性类/地址位置',
  '度量类/计量数值', '度量类/计数统计', '度量类/比率比例', '度量类/时间度量', '度量类/序号排序',
  '身份类/人口统计', '身份类/联系方式', '身份类/教育背景', '身份类/职业信息',
  '状态类/二元标志', '状态类/状态枚举', '状态类/时间标记',
  '扩展类/扩展代码', '扩展类/其他字段'
]

// 动态统计数据
const stats = ref({
  totalProcessed: 0,
  totalFiles: 0,
  sensitiveCount: 0,
  avgTime: '0s/字段'
})

// 从 localStorage 加载统计
const loadStats = () => {
  try {
    const stored = localStorage.getItem('classification_stats')
    if (stored) {
      const data = JSON.parse(stored)
      stats.value = {
        totalProcessed: data.totalFields || 0,
        totalFiles: data.totalFiles || 0,
        sensitiveCount: data.sensitiveFields || 0,
        avgTime: data.totalTime > 0 && data.totalFields > 0
          ? `${(data.totalTime / data.totalFields / 1000).toFixed(1)}s/字段`
          : '0.5s/字段'
      }
    }
  } catch (e) {
    console.error('加载统计失败:', e)
  }
}

// 加载图表数据
const loadChartData = () => {
  try {
    const stored = localStorage.getItem('classification_stats')
    if (stored) {
      const data = JSON.parse(stored)
      // 更新分类图表 - 按大类分组
      const categoryChart = echarts.init(document.getElementById('categoryChart'))

      // 按大类聚合分类数据
      const categoryMap = {}
      for (const [name, value] of Object.entries(data.classificationDistribution || {})) {
        const mainCategory = name.split('/')[0] // 取大类名称
        categoryMap[mainCategory] = (categoryMap[mainCategory] || 0) + value
      }

      const classData = Object.entries(categoryMap).map(([name, value]) => ({ name, value }))
      if (classData.length > 0) {
        categoryChart.setOption({
          tooltip: { trigger: 'item' },
          legend: { orient: 'vertical', left: 'left' },
          color: ['#5a8ab0', '#68a67d', '#c4a35a', '#b05a5a', '#7a6ab0', '#5ab08a'],
          series: [{
            name: '分类统计',
            type: 'pie',
            radius: '70%',
            data: classData,
            emphasis: {
              itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0, 0, 0, 0.5)' }
            }
          }]
        })
      }

      // 更新分级图表
      const gradingChart = echarts.init(document.getElementById('gradingChart'))
      const gradeData = ['第1级/公开', '第2级/内部', '第3级/敏感', '第4级/机密'].map(
        grade => ({ grade, count: data.gradingDistribution?.[grade] || 0 })
      )
      gradingChart.setOption({
        tooltip: { trigger: 'axis' },
        xAxis: {
          type: 'category',
          data: gradeData.map(d => d.grade)
        },
        yAxis: { type: 'value', name: '字段数' },
        color: ['#68a67d', '#c4a35a', '#b05a5a', '#b05a5a'],
        series: [{
          data: gradeData.map(d => d.count),
          type: 'bar',
          itemStyle: {
            color: (params) => {
              const colors = ['#68a67d', '#c4a35a', '#b05a5a', '#8b0000']
              return colors[params.dataIndex]
            }
          }
        }]
      })
    }
  } catch (e) {
    console.error('加载图表失败:', e)
  }
}

onMounted(() => {
  loadStats()
  initCharts()
  // 监听 storage 变化
  window.addEventListener('storage', loadStats)
})

onActivated(() => {
  // 每次返回首页时刷新统计
  loadStats()
  loadChartData()
})

const initCharts = () => {
  loadChartData()

  // 如果没有真实数据，使用默认图表
  const categoryChart = echarts.getInstanceByDom(document.getElementById('categoryChart'))
  const gradingChart = echarts.getInstanceByDom(document.getElementById('gradingChart'))

  if (!categoryChart) {
    const catChart = echarts.init(document.getElementById('categoryChart'))
    catChart.setOption({
      tooltip: { trigger: 'item' },
      legend: { orient: 'vertical', left: 'left' },
      color: ['#5a8ab0', '#68a67d', '#c4a35a', '#b05a5a', '#7a6ab0'],
      series: [{
        name: '分类统计',
        type: 'pie',
        radius: '70%',
        data: [
          { value: 35, name: '属性类' },
          { value: 25, name: '度量类' },
          { value: 20, name: '状态类' },
          { value: 12, name: '身份类' },
          { value: 8, name: 'ID类' }
        ],
        emphasis: {
          itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0, 0, 0, 0.5)' }
        }
      }]
    })
  }

  if (!gradingChart) {
    const gradChart = echarts.init(document.getElementById('gradingChart'))
    gradChart.setOption({
      tooltip: { trigger: 'axis' },
      xAxis: {
        type: 'category',
        data: ['第1级/公开', '第2级/内部', '第3级/敏感', '第4级/机密']
      },
      yAxis: { type: 'value' },
      color: ['#68a67d', '#c4a35a', '#b05a5a', '#b05a5a'],
      series: [{
        data: [45, 30, 18, 7],
        type: 'bar',
        itemStyle: {
          color: (params) => {
            const colors = ['#68a67d', '#c4a35a', '#b05a5a', '#b05a5a']
            return colors[params.dataIndex]
          }
        }
      }]
    })
  }
}
</script>

<style lang="scss" scoped>
.dashboard-container {
  min-height: 100vh;
  background: #f0f4f8;
}

.header {
  background: linear-gradient(135deg, #2c5282 0%, #1a365d 100%);
  color: white;
  padding: 0 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.header-content {
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo {
  h1 {
    font-size: 24px;
    margin: 0;
  }
  p {
    font-size: 12px;
    margin: 5px 0 0;
    opacity: 0.85;
  }
}

.nav-menu {
  :deep(.el-menu) {
    background: transparent;
    border: none;
  }
  :deep(.el-menu-item) {
    color: rgba(255, 255, 255, 0.9);
    &:hover {
      background: rgba(255, 255, 255, 0.1);
      color: white;
    }
    &.is-active {
      background: rgba(255, 255, 255, 0.15);
      color: white;
      border-bottom-color: rgba(255, 255, 255, 0.8);
    }
  }
}

.content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

.welcome-card {
  background: linear-gradient(135deg, #2c5282 0%, #1a365d 100%);
  color: white;
  padding: 40px;
  border-radius: 12px;
  margin-bottom: 20px;

  h2 {
    margin: 0 0 10px;
  }
  p {
    margin: 0;
    opacity: 0.9;
  }
}

.stats-row {
  margin-bottom: 20px;

  &.secondary {
    .el-col {
      margin-bottom: 10px;
    }
  }
}

.stat-card {
  display: flex;
  align-items: center;
  padding: 20px;
  border-radius: 12px;
  transition: transform 0.3s, box-shadow 0.3s;
  background: white;
  border: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }

  &.small {
    padding: 15px 20px;
  }
}

.stat-icon {
  width: 60px;
  height: 60px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  color: white;
  margin-right: 15px;

  &.blue { background: linear-gradient(135deg, #4a7ab0 0%, #2c5282 100%); }
  &.teal { background: linear-gradient(135deg, #319795 0%, #285e61 100%); }
  &.orange { background: linear-gradient(135deg, #c4a35a 0%, #b08930 100%); }
  &.red { background: linear-gradient(135deg, #b05a5a 0%, #904040 100%); }
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #2d3748;
}

.stat-value-small {
  font-size: 16px;
  font-weight: 600;
  color: #4a5568;
  margin-top: 5px;
}

.stat-label {
  font-size: 14px;
  color: #718096;
  margin-top: 4px;
}

.features-row {
  margin: 20px 0;
}

.feature-card {
  text-align: center;
  padding: 30px 20px;
  border-radius: 12px;
  transition: transform 0.3s, box-shadow 0.3s;
  background: white;
  border: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);

  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  }

  .feature-icon {
    font-size: 48px;
    color: #2c5282;
    margin-bottom: 15px;
  }

  h3 {
    margin: 0 0 10px;
    color: #2d3748;
  }

  p {
    color: #718096;
    margin-bottom: 20px;
    font-size: 14px;
  }
}

.chart-card {
  border-radius: 12px;
  border: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);

  :deep(.el-card__header) {
    border-bottom: 1px solid #e2e8f0;
    color: #2d3748;
    font-weight: 500;
  }
}

.steps-card {
  border-radius: 12px;
  border: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);

  :deep(.el-card__header) {
    border-bottom: 1px solid #e2e8f0;
    color: #2d3748;
    font-weight: 500;
  }
}
</style>
