<template>
  <div class="visualization-container">
    <div class="header">
      <div class="header-content">
        <div class="logo">
          <h1>分析结果可视化</h1>
          <p class="subtitle">查看历史分析记录和数据分布统计</p>
        </div>
        <div class="nav-menu">
          <el-menu mode="horizontal" :default-active="currentRoute" :ellipsis="false" router>
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

    <!-- 主内容区域 -->
    <div class="main-content">
      <!-- 统计概览 -->
      <div class="stats-row">
      <el-row :gutter="20">
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon blue"><el-icon><Document /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ historyList.length }}</div>
              <div class="stat-label">分析记录</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon green"><el-icon><Files /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ totalFields }}</div>
              <div class="stat-label">总字段数</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon orange"><el-icon><Warning /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ totalSensitive }}</div>
              <div class="stat-label">敏感字段</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <el-icon class="stat-icon purple"><Delete /></el-icon>
            <div class="stat-info">
              <el-button type="danger" link @click="clearAllHistory">清空全部</el-button>
              <div class="stat-label">谨慎操作</div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 历史记录列表 -->
    <div class="history-section">
      <el-card class="history-card">
        <template #header>
          <div class="card-header">
            <span>历史分析记录</span>
            <el-button type="primary" plain @click="loadHistory">
              <el-icon><Refresh /></el-icon> 刷新
            </el-button>
          </div>
        </template>

        <div v-if="historyList.length === 0" class="empty-state">
          <el-empty description="暂无分析记录" />
          <el-button type="primary" @click="$router.push('/classify')">去分析文件</el-button>
        </div>

        <el-table v-else :data="historyList" stripe style="width: 100%">
          <el-table-column prop="filename" label="文件名" min-width="180" />
          <el-table-column prop="industry" label="行业" width="100">
            <template #default="{ row }">
              <el-tag size="small">{{ row.industry }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="fieldCount" label="字段数" width="100" align="center" />
          <el-table-column prop="sensitiveCount" label="敏感字段" width="100" align="center">
            <template #default="{ row }">
              <el-tag v-if="row.sensitiveCount > 0" type="danger" size="small">
                {{ row.sensitiveCount }}
              </el-tag>
              <span v-else class="text-muted">0</span>
            </template>
          </el-table-column>
          <el-table-column prop="timestamp" label="分析时间" width="180">
            <template #default="{ row }">
              {{ formatDate(row.timestamp) }}
            </template>
          </el-table-column>
          <el-table-column label="操作" width="200" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" link @click="viewDetail(row)">查看详情</el-button>
              <el-button type="success" link @click="exportRecord(row)">导出</el-button>
              <el-button type="danger" link @click="deleteRecord(row.id)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </div>

    <!-- 详情对话框 -->
    <el-dialog v-model="detailVisible" title="分析详情" width="90%" top="5vh">
      <div v-if="currentRecord" class="detail-container">
        <!-- 文件信息 -->
        <el-card class="info-card">
          <el-descriptions :column="4" border>
            <el-descriptions-item label="文件名">{{ currentRecord.filename }}</el-descriptions-item>
            <el-descriptions-item label="行业">{{ currentRecord.industry }}</el-descriptions-item>
            <el-descriptions-item label="字段数">{{ currentRecord.fieldCount }}</el-descriptions-item>
            <el-descriptions-item label="分析时间">{{ formatDate(currentRecord.timestamp) }}</el-descriptions-item>
          </el-descriptions>
        </el-card>

        <!-- 图表区域 -->
        <el-row :gutter="20">
          <!-- 分类分布饼图 -->
          <el-col :span="12">
            <el-card class="chart-card">
              <template #header>
                <span>分类分布</span>
              </template>
              <div ref="classificationChart" class="chart-container"></div>
            </el-card>
          </el-col>
          <!-- 分级分布饼图 -->
          <el-col :span="12">
            <el-card class="chart-card">
              <template #header>
                <span>分级分布</span>
              </template>
              <div ref="gradingChart" class="chart-container"></div>
            </el-card>
          </el-col>
        </el-row>

        <el-row :gutter="20">
          <!-- 分类统计柱状图 -->
          <el-col :span="24">
            <el-card class="chart-card">
              <template #header>
                <span>各分类字段数量</span>
              </template>
              <div ref="categoryBarChart" class="chart-container-wide"></div>
            </el-card>
          </el-col>
        </el-row>

        <!-- 敏感字段列表 -->
        <el-card v-if="sensitiveFields.length > 0" class="sensitive-card">
          <template #header>
            <span>敏感字段 ({{ sensitiveFields.length }})</span>
          </template>
          <el-table :data="sensitiveFields" stripe size="small">
            <el-table-column prop="field_name" label="字段名" width="200" />
            <el-table-column prop="classification" label="分类" width="200">
              <template #default="{ row }">
                <el-tag type="success" size="small">{{ row.classification }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="grading" label="分级" width="150">
              <template #default="{ row }">
                <el-tag :type="row.grading === '第4级/机密' ? 'danger' : 'warning'" size="small">
                  {{ row.grading }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="meaning" label="业务含义" />
          </el-table>
        </el-card>

        <!-- 完整字段列表 -->
        <el-card class="all-fields-card">
          <template #header>
            <div class="card-header">
              <span>完整字段列表 ({{ currentRecord.fieldCount }})</span>
              <el-button type="primary" size="small" @click="exportRecord(currentRecord)">
                导出结果
              </el-button>
            </div>
          </template>
          <el-table :data="currentRecord.results" stripe size="small" max-height="400">
            <el-table-column prop="field_name" label="字段名" width="180" />
            <el-table-column prop="classification" label="分类" width="200">
              <template #default="{ row }">
                <el-tag type="success" size="small">{{ row.classification }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="grading" label="分级" width="150">
              <template #default="{ row }">
                <el-tag :type="getGradingType(row.grading)" size="small">
                  {{ row.grading }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="meaning" label="业务含义" min-width="200" />
            <el-table-column prop="data_type" label="数据类型" width="120" />
          </el-table>
        </el-card>
      </div>
    </el-dialog>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Document, Files, Warning, Refresh, Delete, House, Box } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import * as XLSX from 'xlsx'

const API_BASE_URL = 'http://localhost:8001'

const router = useRouter()
const currentRoute = ref('/visualization')

// 历史记录
const historyList = ref([])
const detailVisible = ref(false)
const currentRecord = ref(null)

// 图表 DOM refs
const classificationChart = ref(null)
const gradingChart = ref(null)
const categoryBarChart = ref(null)

// 图表实例
let classificationChartInstance = null
let gradingChartInstance = null
let categoryBarChartInstance = null

// 监听对话框打开
watch(detailVisible, (newVal) => {
  if (newVal) {
    setTimeout(() => {
      nextTick(() => {
        initCharts()
      })
    }, 200)
  }
})

// 计算属性
const totalFields = computed(() => {
  return historyList.value.reduce((sum, r) => sum + (r.fieldCount || 0), 0)
})

const totalSensitive = computed(() => {
  return historyList.value.reduce((sum, r) => sum + (r.sensitiveCount || 0), 0)
})

const sensitiveFields = computed(() => {
  if (!currentRecord.value || !currentRecord.value.results) return []
  return currentRecord.value.results.filter(r =>
    r.grading === '第3级/敏感' || r.grading === '第4级/机密'
  )
})

// 加载历史记录
const loadHistory = () => {
  const saved = localStorage.getItem('analysis_history')
  if (saved) {
    try {
      historyList.value = JSON.parse(saved)
    } catch (e) {
      historyList.value = []
    }
  }
}

// 保存历史记录
const saveHistory = () => {
  localStorage.setItem('analysis_history', JSON.stringify(historyList.value))
}

// 删除单条记录
const deleteRecord = async (id) => {
  try {
    await ElMessageBox.confirm('确定要删除这条记录吗?', '提示', { type: 'warning' })
    historyList.value = historyList.value.filter(r => r.id !== id)
    saveHistory()
    ElMessage.success('删除成功')
  } catch {
    // 用户取消
  }
}

// 清空全部
const clearAllHistory = async () => {
  try {
    await ElMessageBox.confirm('确定要清空全部历史记录吗？此操作不可恢复！', '警告', {
      type: 'warning',
      confirmButtonText: '确定清空',
      cancelButtonText: '取消'
    })
    historyList.value = []
    saveHistory()
    ElMessage.success('已清空全部历史记录')
  } catch {
    // 用户取消
  }
}

// 查看详情
const viewDetail = async (record) => {
  currentRecord.value = record
  detailVisible.value = true
}

// 导出记录
const exportRecord = (record) => {
  if (!record || !record.results) {
    ElMessage.warning('没有可导出的数据')
    return
  }

  const ws = XLSX.utils.json_to_sheet(record.results.map(r => ({
    '字段名': r.field_name,
    '分类': r.classification,
    '分级': r.grading,
    '数据类型': r.data_type,
    '业务含义': r.meaning
  })))
  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, ws, '分类结果')
  XLSX.writeFile(wb, `分析结果_${record.filename}_${formatDate(record.timestamp).replace(/[/:]/g, '-')}.xlsx`)
  ElMessage.success('导出成功')
}

// 格式化日期
const formatDate = (timestamp) => {
  if (!timestamp) return '-'
  const date = new Date(timestamp)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 获取分级标签类型
const getGradingType = (grading) => {
  if (grading === '第1级/公开') return 'info'
  if (grading === '第2级/内部') return ''
  if (grading === '第3级/敏感') return 'warning'
  if (grading === '第4级/机密') return 'danger'
  return 'info'
}

// 预定义颜色数组
const chartColors = [
  '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
  '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#73c0de',
  '#fc8452', '#91cc75', '#5470c6', '#ee6666', '#fac858'
]

// 初始化图表
const initCharts = () => {
  if (!currentRecord.value || !currentRecord.value.results) return

  const results = currentRecord.value.results

  // 分类分布饼图
  const classificationData = {}
  results.forEach(r => {
    classificationData[r.classification] = (classificationData[r.classification] || 0) + 1
  })

  if (classificationChartInstance) {
    classificationChartInstance.dispose()
  }
  if (classificationChart.value) {
    classificationChartInstance = echarts.init(classificationChart.value)
    const dataEntries = Object.entries(classificationData)
    classificationChartInstance.setOption({
      tooltip: { trigger: 'item' },
      legend: { orient: 'vertical', left: 'left', textStyle: { fontSize: 11 } },
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
        label: { show: false },
        emphasis: { label: { show: true, fontSize: 14, fontWeight: 'bold' } },
        data: dataEntries.map(([name, value], i) => ({
          name, value,
          itemStyle: { color: chartColors[i % chartColors.length] }
        }))
      }]
    })
  }

  // 分级分布饼图
  const gradingData = {}
  results.forEach(r => {
    gradingData[r.grading] = (gradingData[r.grading] || 0) + 1
  })

  const gradingColors = {
    '第1级/公开': '#909399',
    '第2级/内部': '#409EFF',
    '第3级/敏感': '#E6A23C',
    '第4级/机密': '#F56C6C'
  }

  if (gradingChartInstance) {
    gradingChartInstance.dispose()
  }
  if (gradingChart.value) {
    gradingChartInstance = echarts.init(gradingChart.value)
    gradingChartInstance.setOption({
      tooltip: { trigger: 'item' },
      legend: { orient: 'vertical', left: 'left' },
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
        label: { show: false },
        emphasis: { label: { show: true, fontSize: 14, fontWeight: 'bold' } },
        data: Object.entries(gradingData).map(([name, value]) => ({
          name, value,
          itemStyle: { color: gradingColors[name] || '#909399' }
        }))
      }]
    })
  }

  // 分类柱状图
  const categorySorted = Object.entries(classificationData)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)

  if (categoryBarChartInstance) {
    categoryBarChartInstance.dispose()
  }
  if (categoryBarChart.value) {
    categoryBarChartInstance = echarts.init(categoryBarChart.value)
    categoryBarChartInstance.setOption({
      tooltip: { trigger: 'axis' },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'value' },
      yAxis: {
        type: 'category',
        data: categorySorted.map(([name]) => name.split('/')[1] || name),
        axisLabel: { interval: 0, fontSize: 10 }
      },
      series: [{
        type: 'bar',
        data: categorySorted.map(([, value]) => value),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#409EFF' },
            { offset: 1, color: '#79BBFF' }
          ]),
          borderRadius: [0, 4, 4, 0]
        },
        label: { show: true, position: 'right' }
      }]
    })
  }
}

// 窗口调整时重绘图表
const handleResize = () => {
  classificationChartInstance?.resize()
  gradingChartInstance?.resize()
  categoryBarChartInstance?.resize()
}

onMounted(() => {
  loadHistory()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  classificationChartInstance?.dispose()
  gradingChartInstance?.dispose()
  categoryBarChartInstance?.dispose()
})
</script>

<style lang="scss" scoped>
.visualization-container {
  padding: 20px;
  min-height: 100vh;
  background: #f0f4f8;
}

.header {
  padding: 0 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  background: linear-gradient(135deg, #2c5282 0%, #1a365d 100%);
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
    color: white;
  }
  .subtitle {
    font-size: 12px;
    margin: 5px 0 0;
    opacity: 0.85;
    color: white;
  }
}

.main-content {
  padding: 20px;
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

.stats-row {
  max-width: 1200px;
  margin: 0 auto 30px;
}

.stat-card {
  display: flex;
  align-items: center;
  padding: 20px;
  border-radius: 12px;

  :deep(.el-card__body) {
    display: flex;
    align-items: center;
    width: 100%;
  }
}

.stat-icon {
  width: 60px;
  height: 60px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: white;
  margin-right: 15px;

  &.blue { background: linear-gradient(135deg, #4a7ab0 0%, #2c5282 100%); }
  &.green { background: linear-gradient(135deg, #68a67d 0%, #4a8a5f 100%); }
  &.orange { background: linear-gradient(135deg, #e6a23c 0%, #c07820 100%); }
  &.purple { background: linear-gradient(135deg, #9c6ade 0%, #7c4dbe 100%); }
}

.stat-info {
  flex: 1;
  text-align: center;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #2d3748;
}

.stat-label {
  font-size: 14px;
  color: #718096;
  margin-top: 4px;
}

.history-section {
  max-width: 1200px;
  margin: 0 auto;
}

.history-card {
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);

  :deep(.el-card__header) {
    border-bottom: 1px solid #e2e8f0;
  }
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.empty-state {
  text-align: center;
  padding: 40px 0;

  .el-button {
    margin-top: 20px;
  }
}

.text-muted {
  color: #909399;
}

.detail-container {
  max-height: 75vh;
  overflow-y: auto;
  padding-right: 10px;
}

.info-card {
  margin-bottom: 20px;
}

.chart-card {
  margin-bottom: 20px;

  :deep(.el-card__header) {
    font-weight: 500;
  }
}

.chart-container {
  height: 300px;
}

.chart-container-wide {
  height: 350px;
}

.sensitive-card,
.all-fields-card {
  margin-bottom: 20px;

  :deep(.el-card__header) {
    font-weight: 500;
  }
}

:deep(.el-descriptions__label) {
  width: 120px;
}
</style>
