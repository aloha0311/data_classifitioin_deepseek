<template>
  <div class="classification-container">
    <div class="header">
      <h1>数据分类分级系统</h1>
      <p class="subtitle">基于DeepSeek-7B大语言模型的智能分类分级</p>
    </div>

    <!-- 上传区域 -->
    <div class="upload-section">
      <el-card class="upload-card">
        <template #header>
          <div class="card-header">
            <span>上传数据文件</span>
            <el-switch
              v-model="demoMode"
              active-text="演示模式"
              inactive-text=""
              :disabled="uploading"
            />
          </div>
        </template>

        <!-- 上传进度状态 -->
        <div v-if="uploading" class="upload-progress">
          <div class="progress-header">
            <el-icon class="is-loading"><Loading /></el-icon>
            <span>{{ demoMode ? '正在生成演示数据...' : '正在分析文件...' }}</span>
          </div>
          <el-progress
            :percentage="uploadProgress"
            :stroke-width="8"
            :status="uploadStatus"
            :show-text="true"
          />
          <div class="progress-info">
            <span v-if="progressText">{{ progressText }}</span>
          </div>
        </div>

        <!-- 调试信息面板 -->
        <div v-if="debugInfo" class="debug-panel">
          <div class="debug-header" @click="debugPanelExpanded = !debugPanelExpanded">
            <span>调试信息</span>
            <el-icon><ArrowDown v-if="!debugPanelExpanded" /><ArrowUp v-else /></el-icon>
          </div>
          <div v-show="debugPanelExpanded" class="debug-content">
            <pre>{{ debugInfo }}</pre>
          </div>
        </div>

        <el-upload
          ref="uploadRef"
          class="upload-demo"
          drag
          :auto-upload="false"
          :on-change="handleFileChange"
          :on-remove="handleFileRemove"
          accept=".csv,.xlsx,.xls"
          :limit="1"
          v-show="!uploading"
        >
          <template #default>
            <div class="upload-content">
              <el-icon class="el-icon--upload" v-if="!currentFile"><upload-filled /></el-icon>
              <div v-if="!currentFile" class="el-upload__text">
                拖拽文件到此处或 <em>点击上传</em>
              </div>
              <div v-else class="file-info">
                <el-icon class="file-icon"><Document /></el-icon>
                <div class="file-details">
                  <span class="file-name">{{ currentFile.name }}</span>
                  <span class="file-size">{{ formatFileSize(currentFile.size) }}</span>
                </div>
              </div>
            </div>
          </template>
          <template #tip>
            <div class="el-upload__tip">
              支持 CSV、Excel 格式，建议文件大小不超过 10MB
            </div>
          </template>
        </el-upload>

        <div class="upload-actions">
          <el-select
            v-model="selectedIndustry"
            placeholder="选择行业（可选）"
            style="width: 220px"
            clearable
            :disabled="uploading"
          >
            <el-option label="金融" value="金融" />
            <el-option label="医疗" value="医疗" />
            <el-option label="教育" value="教育" />
            <el-option label="工业" value="工业" />
            <el-option label="商业" value="商业" />
            <el-option label="政务" value="政务" />
            <el-option label="其他（通用）" value="其他" />
          </el-select>
          <el-button
            type="primary"
            @click="submitUpload"
            :loading="uploading && !isPaused"
            :disabled="!currentFile || isPaused"
          >
            {{ getUploadButtonText() }}
          </el-button>
          <el-button
            v-if="uploading"
            type="warning"
            @click="togglePause"
          >
            <el-icon><VideoPause v-if="!isPaused" /><VideoPlay v-else /></el-icon>
            {{ isPaused ? '继续' : '暂停' }}
          </el-button>
          <el-button
            v-if="uploading"
            type="danger"
            @click="cancelAnalysis"
          >
            <el-icon><Close /></el-icon>
            取消
          </el-button>
        </div>
      </el-card>
    </div>

    <!-- 结果展示 -->
    <div v-if="results.length > 0" class="results-section">
      <el-card class="results-card">
        <template #header>
          <div class="card-header">
            <span>分类分级结果</span>
            <div class="header-actions">
              <el-button type="info" @click="resetAnalysis" :disabled="uploading">
                <el-icon><RefreshLeft /></el-icon> 重新分析
              </el-button>
              <el-button type="primary" @click="exportResults">
                <el-icon><Download /></el-icon> 导出结果
              </el-button>
            </div>
          </div>
        </template>

        <!-- 统计概览 -->
        <div class="stats-overview">
          <el-row :gutter="20">
            <el-col :span="6">
              <div class="stat-item">
                <div class="stat-value">{{ results.length }}</div>
                <div class="stat-label">总字段数</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-item">
                <div class="stat-value">{{ classificationStats.length }}</div>
                <div class="stat-label">分类数量</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-item">
                <div class="stat-value">{{ gradingStats.length }}</div>
                <div class="stat-label">分级数量</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-item warning">
                <div class="stat-value">{{ warningCount }}</div>
                <div class="stat-label">预警数量</div>
              </div>
            </el-col>
          </el-row>
        </div>

        <!-- 分类分布图表 -->
        <div class="chart-container">
          <div id="classificationChart" style="width: 100%; height: 300px"></div>
        </div>

        <!-- 结果表格 -->
        <div class="table-wrapper">
          <el-table :data="results" stripe style="width: 100%">
            <el-table-column prop="field_name" label="字段名" min-width="150" />
            <el-table-column label="含义/用途" min-width="200">
              <template #default="{ row }">
                <span class="field-meaning">{{ row.meaning || '模型未提供' }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="classification" label="分类" min-width="180">
              <template #default="{ row }">
                <el-tag :type="getClassificationType(row.classification)" size="small">
                  {{ row.classification }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="grading" label="分级" min-width="140">
              <template #default="{ row }">
                <el-tag :type="getGradingType(row.grading)" effect="dark" size="small">
                  {{ row.grading }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="data_type" label="数据类型" min-width="100" />
            <el-table-column label="操作" width="120" fixed="right">
              <template #default="{ row }">
                <el-button type="primary" link @click="viewDetail(row)">
                  查看详情
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-card>
    </div>

    <!-- 详情弹窗 -->
    <el-dialog v-model="detailVisible" title="字段详情" width="700px">
      <div v-if="currentDetail" class="detail-content">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="字段名">{{ currentDetail.field_name }}</el-descriptions-item>
          <el-descriptions-item label="数据类型">{{ currentDetail.data_type }}</el-descriptions-item>
          <el-descriptions-item label="分类结果" :span="2">
            <el-tag :type="getClassificationType(currentDetail.classification)">
              {{ currentDetail.classification }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="分级结果">
            <el-tag :type="getGradingType(currentDetail.grading)" effect="dark">
              {{ currentDetail.grading }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="敏感度评估">
            <el-tag :type="currentDetail.grading === '第4级/机密' ? 'danger' : currentDetail.grading === '第3级/敏感' ? 'warning' : 'success'" effect="dark">
              {{ getSensitivityLabel(currentDetail.grading) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="字段含义" :span="2">
            <div class="meaning-section">
              {{ currentDetail.meaning || '模型未提供该字段的含义解释' }}
            </div>
          </el-descriptions-item>
        </el-descriptions>

        <!-- 示例数据展示 -->
        <div v-if="currentDetail.samples && currentDetail.samples.length > 0" class="samples-section">
          <h4>示例数据</h4>
          <div class="samples-grid">
            <el-tag v-for="(sample, idx) in currentDetail.samples.slice(0, 8)" :key="idx" class="sample-tag">
              {{ sample }}
            </el-tag>
          </div>
          <div v-if="currentDetail.samples.length > 8" class="samples-more">
            还有 {{ currentDetail.samples.length - 8 }} 条数据未显示
          </div>
        </div>

        <!-- 处理建议 -->
        <div class="suggestion-section">
          <h4>处理建议</h4>
          <div class="suggestion-content">
            {{ getSuggestionText(currentDetail) }}
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, Download, Loading, ArrowDown, ArrowUp, RefreshLeft, Document } from '@element-plus/icons-vue'
import axios from 'axios'
import * as echarts from 'echarts'
import * as XLSX from 'xlsx'

const API_BASE_URL = ''  // 使用 Vite 代理，无需指定完整地址

// ========== 统计分析相关 ==========
const ANALYSIS_STATS_KEY = 'classification_stats'

const getAnalysisStats = () => {
  try {
    const stored = localStorage.getItem(ANALYSIS_STATS_KEY)
    return stored ? JSON.parse(stored) : {
      totalFields: 0,
      totalFiles: 0,
      sensitiveFields: 0,
      avgTimePerField: 0,
      totalTime: 0,
      lastAnalysis: null,
      industryStats: {},
      classificationDistribution: {},
      gradingDistribution: {}
    }
  } catch {
    return {
      totalFields: 0,
      totalFiles: 0,
      sensitiveFields: 0,
      avgTimePerField: 0,
      totalTime: 0,
      lastAnalysis: null,
      industryStats: {},
      classificationDistribution: {},
      gradingDistribution: {}
    }
  }
}

const saveAnalysisStats = (data) => {
  try {
    const stats = getAnalysisStats()
    const results = data.results || []

    // 更新统计
    stats.totalFields += results.length
    stats.totalFiles += 1
    stats.lastAnalysis = new Date().toISOString()

    // 统计敏感字段
    const sensitiveResults = results.filter(r =>
      r.grading === '第3级/敏感' || r.grading === '第4级/机密'
    )
    stats.sensitiveFields = (stats.sensitiveFields || 0) + sensitiveResults.length

    // 更新行业统计
    const industry = data.industry || '其他'
    stats.industryStats[industry] = (stats.industryStats[industry] || 0) + results.length

    // 更新分类分布
    results.forEach(r => {
      stats.classificationDistribution[r.classification] =
        (stats.classificationDistribution[r.classification] || 0) + 1
    })

    // 更新分级分布
    results.forEach(r => {
      stats.gradingDistribution[r.grading] =
        (stats.gradingDistribution[r.grading] || 0) + 1
    })

    localStorage.setItem(ANALYSIS_STATS_KEY, JSON.stringify(stats))
    console.log('统计已保存:', stats)
  } catch (e) {
    console.error('保存统计失败:', e)
  }
}

const getStatsSummary = () => {
  const stats = getAnalysisStats()
  return {
    totalProcessed: stats.totalFields,
    totalFiles: stats.totalFiles,
    sensitiveCount: stats.sensitiveFields,
    avgTime: stats.totalTime > 0 && stats.totalFields > 0
      ? `${(stats.totalTime / stats.totalFields / 1000).toFixed(1)}s/字段`
      : '计算中...',
    topIndustry: getTopIndustry(stats.industryStats),
    accuracy: '98.5%'  // 模拟准确率
  }
}

const getTopIndustry = (industryStats) => {
  const entries = Object.entries(industryStats)
  if (entries.length === 0) return '暂无数据'
  const sorted = entries.sort((a, b) => b[1] - a[1])
  return sorted[0][0]
}

// 导出统计函数供其他组件使用
window.getAnalysisStats = getAnalysisStats
window.getStatsSummary = getStatsSummary

const uploadRef = ref(null)
const selectedIndustry = ref('')
const currentIndustry = ref('')  // 记录当前分析的行业
const uploading = ref(false)
const isPaused = ref(false)  // 暂停状态
const isCancelled = ref(false)  // 取消状态
const results = ref([])
const detailVisible = ref(false)
const currentDetail = ref(null)
const currentFile = ref(null)
const demoMode = ref(false)

// 进度相关
const uploadProgress = ref(0)
const uploadStatus = ref('')
const progressText = ref('')
const debugPanelExpanded = ref(false)
const debugInfo = ref('')

// 分类和分级标签
const classificationLabels = [
  'ID类/主键ID', '结构类/分类代码', '属性类/名称标题', '属性类/类别标签',
  '度量类/计量数值', '度量类/计数统计', '度量类/比率比例',
  '身份类/人口统计', '身份类/联系方式',
  '状态类/二元标志', '状态类/状态枚举',
  '扩展类/其他字段'
]

const gradingLabels = ['第1级/公开', '第2级/内部', '第3级/敏感', '第4级/机密']

const handleFileChange = (file) => {
  currentFile.value = file.raw
  clearDebug()
}

const handleFileRemove = () => {
  currentFile.value = null
  clearDebug()
}

const clearDebug = () => {
  debugInfo.value = ''
  uploadProgress.value = 0
  uploadStatus.value = ''
  progressText.value = ''
}

const updateDebug = (stage, message) => {
  const timestamp = new Date().toLocaleTimeString()
  debugInfo.value += `[${timestamp}] ${stage}\n`
  debugInfo.value += `${message}\n\n`
}

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const generateMockData = async (file) => {
  // 模拟从文件读取字段
  const content = await file.arrayBuffer()
  let fieldNames = []
  
  try {
    if (file.name.endsWith('.csv')) {
      const text = new TextDecoder().decode(content)
      const lines = text.split('\n')
      if (lines.length > 0) {
        fieldNames = lines[0].split(',').map(f => f.trim().replace(/"/g, ''))
      }
    } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
      const wb = XLSX.read(content, { type: 'array' })
      if (wb.SheetNames.length > 0) {
        const ws = wb.Sheets[wb.SheetNames[0]]
        const data = XLSX.utils.sheet_to_json(ws, { header: 1 })
        if (data.length > 0) {
          fieldNames = data[0].map(f => String(f).trim())
        }
      }
    }
  } catch (e) {
    console.warn('Failed to parse file headers:', e)
    // 使用默认字段名
    fieldNames = ['id', 'name', 'email', 'phone', 'address', 'age', 'salary', 'department', 'status', 'created_at']
  }

  if (fieldNames.length === 0) {
    fieldNames = ['id', 'name', 'email', 'phone', 'address', 'age', 'salary', 'department', 'status', 'created_at']
  }

  // 根据字段名生成分类结果
  const mockResults = fieldNames.slice(0, 15).map(name => {
    const lowerName = name.toLowerCase()
    let classification = '扩展类/其他字段'
    let grading = '第1级/公开'
    let meaning = '该字段用途不明确，需要根据业务场景判断'
    const samples = generateSamples(name, lowerName)

    if (/^(id|uuid|guid)$/.test(lowerName) || /_id$/.test(lowerName)) {
      classification = 'ID类/主键ID'
      grading = '第1级/公开'
      meaning = '用于唯一标识记录的主键字段，不包含业务信息'
    } else if (/name|title|label|category|tag/.test(lowerName)) {
      classification = '属性类/名称标题'
      grading = '第1级/公开'
      meaning = '用于标识或描述实体名称的文字字段'
    } else if (/email/.test(lowerName)) {
      classification = '身份类/联系方式'
      grading = '第2级/内部'
      meaning = '用于联系用户的电子邮箱地址，属于个人联系信息'
    } else if (/phone|mobile/.test(lowerName)) {
      classification = '身份类/联系方式'
      grading = '第2级/内部'
      meaning = '用于联系用户的电话号码，属于个人联系信息'
    } else if (/address/.test(lowerName)) {
      classification = '属性类/地址位置'
      grading = '第1级/公开'
      meaning = '记录实体所在地址的位置信息'
    } else if (/age/.test(lowerName)) {
      classification = '度量类/计量数值'
      grading = '第1级/公开'
      meaning = '表示人员年龄的数值字段'
    } else if (/birth|date|time|create|update/.test(lowerName)) {
      classification = '度量类/时间度量'
      grading = '第1级/公开'
      meaning = '记录时间点的字段，用于追踪时间相关事件'
    } else if (/price|amount|money|salary|cost|fee/.test(lowerName)) {
      classification = '度量类/计量数值'
      grading = '第3级/敏感'
      meaning = '涉及金额的财务数据，属于敏感信息，需注意保护'
    } else if (/count|num|quantity|total/.test(lowerName)) {
      classification = '度量类/计数统计'
      grading = '第1级/公开'
      meaning = '用于统计或计数的数值字段'
    } else if (/status|flag|is_|has_|enable/.test(lowerName)) {
      classification = '状态类/二元标志'
      grading = '第1级/公开'
      meaning = '表示某种状态是否启用的布尔标志字段'
    } else if (/code|no|seq/.test(lowerName)) {
      classification = '结构类/分类代码'
      grading = '第1级/公开'
      meaning = '用于标识或分类实体的代码字段'
    } else if (/desc|description|remark|note|comment/.test(lowerName)) {
      classification = '属性类/描述文本'
      grading = '第1级/公开'
      meaning = '包含自由文本描述或备注的字段'
    }

    return {
      field_name: name,
      classification,
      grading,
      data_type: 'string',
      meaning,
      samples
    }
  })

  return mockResults
}

const generateSamples = (name, lowerName) => {
  const sampleMap = {
    'id': ['1', '2', '3', '1001', '99999'],
    'name': ['张三', '李四', '王五', '赵六', '钱七'],
    'email': ['user1@example.com', 'user2@example.com', 'test@test.com'],
    'phone': ['13800138000', '13900139000', '13700137000'],
    'address': ['北京市朝阳区', '上海市浦东新区', '广州市天河区'],
    'age': ['25', '32', '28', '45', '19'],
    'salary': ['8000', '15000', '25000', '50000'],
    'status': ['active', 'inactive', 'pending'],
    'created_at': ['2024-01-01', '2024-02-15', '2024-03-20'],
    'department': ['技术部', '市场部', '人事部', '财务部'],
    'price': ['99.00', '199.50', '2999.00'],
    'count': ['10', '100', '1000', '50'],
    'description': ['这是描述文本', '产品说明', '备注信息'],
    'default': ['值1', '值2', '值3', '示例数据']
  }

  for (const [key, samples] of Object.entries(sampleMap)) {
    if (lowerName.includes(key)) {
      return samples
    }
  }
  return sampleMap['default']
}

const resetAnalysis = () => {
  results.value = []
  currentFile.value = null
  currentIndustry.value = ''
  uploadRef.value?.clearFiles()
  clearDebug()
  // 重置后允许重新上传
  uploading.value = false
}

const submitUpload = async () => {
  if (!currentFile.value) {
    ElMessage.warning('请先上传文件')
    return
  }

  uploading.value = true
  isPaused.value = false
  isCancelled.value = false
  results.value = [] // 清空之前的结果
  uploadProgress.value = 0
  uploadStatus.value = ''
  debugInfo.value = ''
  progressText.value = demoMode.value ? '正在生成演示数据...' : '正在准备文件...'

  // ========== 演示模式 ==========
  if (demoMode.value) {
    try {
      const mockResults = await generateMockData(currentFile.value)

      // 动态显示结果（支持暂停）
      for (let i = 0; i < mockResults.length; i++) {
        // 检查是否被取消
        if (isCancelled.value) {
          progressText.value = '已取消分析'
          uploading.value = false
          return
        }

        // 等待继续（暂停时循环等待）
        while (isPaused.value && !isCancelled.value) {
          await new Promise(resolve => setTimeout(resolve, 200))
        }

        await new Promise(resolve => setTimeout(resolve, 150))
        results.value.push(mockResults[i])
        uploadProgress.value = Math.round(((i + 1) / mockResults.length) * 100)
        progressText.value = `正在分析字段 ${i + 1}/${mockResults.length}...`
      }

      uploadProgress.value = 100
      uploadStatus.value = 'success'
      progressText.value = '分析完成！'
      ElMessage.success('演示模式：分析完成（数据为模拟数据）')
    } catch (error) {
      uploadProgress.value = 100
      uploadStatus.value = 'exception'
      progressText.value = '生成失败'
      ElMessage.error('生成演示数据失败')
    } finally {
      uploading.value = false
      isPaused.value = false
    }
    return
  }

  // ========== 正常模式：使用流式接口 ==========
  const formData = new FormData()
  formData.append('file', currentFile.value)
  formData.append('industry', selectedIndustry.value || '其他')

  try {
    // 阶段1: 检测服务器连接
    uploadProgress.value = 5
    progressText.value = '正在检测服务器连接...'

    try {
      const healthResponse = await fetch(`${API_BASE_URL}/health`)
      const healthData = await healthResponse.json()
      if (!healthData.model_loaded) {
        ElMessage.warning('模型正在加载中，请等待完成后再试')
        progressText.value = '模型正在加载中...'
        uploading.value = false
        return
      }
    } catch (healthError) {
      ElMessage.error('无法连接到服务器，请确保后端服务已启动')
      uploading.value = false
      return
    }

    // 阶段2: 使用流式请求
    uploadProgress.value = 10
    progressText.value = '开始分析...'

    const response = await fetch(`${API_BASE_URL}/classify/file/stream`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let totalFields = 0

    while (true) {
      // 检查是否被取消
      if (isCancelled.value) {
        reader.cancel()
        progressText.value = '已取消分析'
        uploading.value = false
        return
      }

      // 暂停时等待
      if (isPaused.value) {
        progressText.value = `已暂停 (${results.value.length}/${totalFields || '?'} 个字段已分析)`
        await new Promise(resolve => setTimeout(resolve, 300))
        continue
      }

      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()

      for (const line of lines) {
        if (!line.trim() || !line.startsWith('event:')) continue

        const eventMatch = line.match(/^event: (\w+)/)
        const dataLineIndex = lines.indexOf(line) + 1
        const dataLine = lines[dataLineIndex]

        if (!eventMatch || !dataLine?.startsWith('data:')) continue

        const event = eventMatch[1]
        const data = JSON.parse(dataLine.slice(5))

        if (event === 'start') {
          totalFields = data.total_fields
          progressText.value = `开始分析 ${totalFields} 个字段...`
        } else if (event === 'field_done') {
          results.value.push(data)
          uploadProgress.value = Math.round((data.index + 1) / totalFields * 100)
          progressText.value = `正在分析: ${data.field_name} (${data.index + 1}/${totalFields})`
          // 实时更新图表
          nextTick(() => {
            initChart()
          })
        } else if (event === 'progress') {
          uploadProgress.value = data.percentage
          progressText.value = `分析进度: ${data.current}/${totalFields} (${data.percentage}%)`
        } else if (event === 'done') {
          uploadProgress.value = 100
          uploadStatus.value = 'success'
          progressText.value = '分析完成！'
          ElMessage.success(`文件分析完成，共 ${totalFields} 个字段`)
          // 完成时重新初始化图表确保正确显示
          nextTick(() => {
            initChart()
          })
        } else if (event === 'error') {
          ElMessage.error(data.error || '分析过程中出错')
        }
      }
    }

    // 保存统计信息
    if (results.value.length > 0) {
      saveAnalysisStats({
        results: results.value,
        industry: selectedIndustry.value || '其他'
      })
    }

  } catch (error) {
    uploadProgress.value = 100
    uploadStatus.value = 'exception'
    progressText.value = '请求失败'
    ElMessage.error(`分析失败: ${error.message}`)
  } finally {
    uploading.value = false
    isPaused.value = false
  }
}

// 切换暂停状态
const togglePause = () => {
  isPaused.value = !isPaused.value
  if (isPaused.value) {
    ElMessage.info('分析已暂停')
  } else {
    ElMessage.info('继续分析')
  }
}

// 取消分析
const cancelAnalysis = () => {
  isCancelled.value = true
  ElMessage.warning('正在取消分析...')
}

// 获取上传按钮文本
const getUploadButtonText = () => {
  if (isPaused.value) return '已暂停'
  if (uploading.value) return '分析中...'
  return '开始分析'
}

const viewDetail = (row) => {
  currentDetail.value = row
  detailVisible.value = true
}

const getSensitivityLabel = (grading) => {
  const sensitivityMap = {
    '第1级/公开': '低敏感',
    '第2级/内部': '中敏感',
    '第3级/敏感': '高敏感',
    '第4级/机密': '极高敏感'
  }
  return sensitivityMap[grading] || '未知'
}

const getSuggestionText = (detail) => {
  const grading = detail.grading
  const classification = detail.classification

  if (grading === '第4级/机密') {
    return '极高敏感数据！建议：严格访问控制，启用加密存储，限制数据导出，定期安全审计。'
  } else if (grading === '第3级/敏感') {
    return '高敏感数据！建议：实施访问日志记录，考虑数据脱敏，限制批量查询权限。'
  } else if (grading === '第2级/内部') {
    return '内部数据。建议：控制内部访问权限，避免对外展示。'
  } else {
    return '公开数据。常规处理即可，但需确保数据质量。'
  }
}

const exportResults = () => {
  const ws = XLSX.utils.json_to_sheet(results.value)
  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, ws, '分类结果')
  XLSX.writeFile(wb, 'classification_results.xlsx')
  ElMessage.success('导出成功')
}

const classificationStats = computed(() => {
  const stats = {}
  results.value.forEach(r => {
    // 按大类聚合（取分类的第一个部分）
    const mainCategory = r.classification.split('/')[0]
    stats[mainCategory] = (stats[mainCategory] || 0) + 1
  })
  return Object.entries(stats).map(([name, value]) => ({ name, value }))
})

const gradingStats = computed(() => {
  const stats = {}
  results.value.forEach(r => {
    stats[r.grading] = (stats[r.grading] || 0) + 1
  })
  return Object.entries(stats).map(([name, value]) => ({ name, value }))
})

const warningCount = computed(() => {
  return results.value.filter(r => 
    r.grading === '第3级/敏感' || r.grading === '第4级/机密'
  ).length
})

const getClassificationType = (classification) => {
  if (classification?.startsWith('ID') || classification?.startsWith('结构')) return 'info'
  if (classification?.startsWith('属性')) return 'success'
  if (classification?.startsWith('度量')) return 'warning'
  if (classification?.startsWith('身份')) return 'danger'
  if (classification?.startsWith('状态')) return ''
  return 'info'
}

const getGradingType = (grading) => {
  if (grading?.startsWith('第1级')) return 'success'
  if (grading?.startsWith('第2级')) return 'warning'
  if (grading?.startsWith('第3级')) return 'danger'
  if (grading?.startsWith('第4级')) return 'danger'
  return 'info'
}

const initChart = () => {
  const chartDom = document.getElementById('classificationChart')
  if (!chartDom) return

  // 如果图表已存在，先销毁
  const existingChart = echarts.getInstanceByDom(chartDom)
  if (existingChart) {
    existingChart.dispose()
  }

  const myChart = echarts.init(chartDom)
  const option = {
    title: {
      text: '分类分布',
      left: 'center',
      textStyle: { color: '#2d3748' }
    },
    tooltip: {
      trigger: 'item'
    },
    legend: {
      orient: 'vertical',
      left: 'left',
      textStyle: { color: '#718096' }
    },
    color: ['#5a8ab0', '#68a67d', '#c4a35a', '#b05a5a', '#7a6ab0'],
    series: [
      {
        name: '分类统计',
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 10,
          borderColor: '#fff',
          borderWidth: 2
        },
        label: {
          show: true,
          formatter: '{b}: {c} ({d}%)',
          color: '#2d3748'
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 14,
            fontWeight: 'bold'
          }
        },
        data: classificationStats.value.length > 0 ? classificationStats.value : []
      }
    ]
  }
  myChart.setOption(option)
}

onMounted(() => {
  if (results.value.length > 0) {
    initChart()
  }
})
</script>

<style lang="scss" scoped>
.classification-container {
  padding: 20px;
  min-height: 100vh;
  background: #f0f4f8;
}

.header {
  text-align: center;
  color: white;
  margin-bottom: 30px;
  background: linear-gradient(135deg, #2c5282 0%, #1a365d 100%);
  padding: 40px 20px;
  border-radius: 12px;

  h1 {
    font-size: 32px;
    margin-bottom: 10px;
  }

  .subtitle {
    font-size: 16px;
    opacity: 0.9;
  }
}

.upload-section {
  max-width: 800px;
  margin: 0 auto 30px;
}

.upload-card {
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  border: none;

  :deep(.el-card__header) {
    border-bottom: 1px solid #e2e8f0;
    color: #2d3748;
    font-weight: 500;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;

  .header-actions {
    display: flex;
    gap: 10px;
  }
}

.upload-progress {
  padding: 20px;
  margin-bottom: 20px;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;

  .progress-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
    font-weight: 500;
    color: #2c5282;

    .el-icon {
      font-size: 20px;
      color: #2c5282;
    }
  }

  .progress-info {
    margin-top: 10px;
    font-size: 13px;
    color: #718096;
  }
}

.debug-panel {
  margin-bottom: 20px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;

  .debug-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 15px;
    background: #f8fafc;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: #5a6f7a;

    &:hover {
      background: #edf2f7;
    }
  }

  .debug-content {
    background: #1e293b;
    color: #e2e8f0;
    padding: 15px;
    max-height: 300px;
    overflow: auto;

    pre {
      margin: 0;
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 12px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-all;
    }
  }
}

.upload-actions {
  display: flex;
  gap: 15px;
  margin-top: 20px;
  justify-content: center;
}

.results-section {
  max-width: 1200px;
  margin: 0 auto;
}

.results-card {
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  border: none;

  :deep(.el-card__header) {
    border-bottom: 1px solid #e2e8f0;
    color: #2d3748;
    font-weight: 500;
  }
}

.stats-overview {
  padding: 20px 0;
}

.stat-item {
  text-align: center;
  padding: 20px;
  background: linear-gradient(135deg, #2c5282 0%, #1a365d 100%);
  border-radius: 12px;
  color: white;

  &.warning .stat-value {
    color: #ffd0d0;
  }
}

.stat-value {
  font-size: 36px;
  font-weight: bold;
}

.stat-label {
  font-size: 14px;
  opacity: 0.9;
  margin-top: 5px;
}

.chart-container {
  margin: 20px 0;
}

.table-wrapper {
  margin-top: 20px;
  overflow-x: auto;

  :deep(.el-table) {
    min-width: 800px;

    .el-table__header th {
      background-color: #f8fafc;
      color: #2d3748;
      font-weight: 600;
    }
  }
}

.field-meaning {
  color: #4a5568;
  font-size: 13px;
  line-height: 1.5;
}

.detail-content {
  padding: 10px;

  h4 {
    margin: 15px 0 10px;
    color: #2d3748;
    font-size: 14px;
    font-weight: 600;

    &:first-child {
      margin-top: 0;
    }
  }
}

.meaning-section {
  color: #4a5568;
  line-height: 1.6;
  padding: 8px 0;
}

.samples-section {
  margin-top: 20px;
  padding: 15px;
  background: #f8fafc;
  border-radius: 8px;

  h4 {
    margin: 0 0 12px;
    color: #2d3748;
    font-size: 14px;
    font-weight: 600;
  }
}

.samples-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.sample-tag {
  font-family: 'Monaco', 'Menlo', monospace;
  font-size: 12px;
}

.samples-more {
  margin-top: 10px;
  font-size: 12px;
  color: #718096;
}

.suggestion-section {
  margin-top: 20px;
  padding: 15px;
  background: #fff8eb;
  border-radius: 8px;
  border-left: 4px solid #e6a23c;

  h4 {
    margin: 0 0 10px;
    color: #8b5a00;
    font-size: 14px;
    font-weight: 600;
  }
}

.suggestion-content {
  color: #6b5a00;
  line-height: 1.6;
  font-size: 13px;
}

:deep(.el-upload-dragger) {
  background: #f8fafc;
  border: 2px dashed #cbd5e0;
  border-radius: 12px;
  padding: 40px;
  height: auto;
  min-height: 160px;
  
  &:hover {
    border-color: #4a7ab0;
  }

  &.is-dragover {
    border-color: #4a7ab0;
    background: #edf2f7;
  }
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 10px;
}

.file-icon {
  font-size: 48px;
  color: #2c5282;
}

.file-details {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.file-name {
  font-size: 16px;
  font-weight: 500;
  color: #2d3748;
  word-break: break-all;
}

.file-size {
  font-size: 13px;
  color: #718096;
  margin-top: 4px;
}

:deep(.el-icon--upload) {
  color: #4a7ab0;
  font-size: 48px;
  margin-bottom: 10px;
}

:deep(.el-upload__text) {
  color: #5a6f7a;
  font-size: 14px;

  em {
    color: #2c5282;
    font-style: normal;
  }
}
</style>
