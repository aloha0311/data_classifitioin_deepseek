<template>
  <div class="knowledge-container">
    <div class="header">
      <div class="header-content">
        <div class="logo">
          <h1>知识库管理</h1>
          <p class="subtitle">管理分类分级规则与模式库</p>
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
            <div class="stat-icon blue"><el-icon><Setting /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ totalRules }}</div>
              <div class="stat-label">通用规则</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon teal"><el-icon><Briefcase /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ totalIndustryRules }}</div>
              <div class="stat-label">行业规则</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon green"><el-icon><Document /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ cachedFields }}</div>
              <div class="stat-label">总规则数</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon red"><el-icon><Warning /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ conflictCount }}</div>
              <div class="stat-label">冲突预警</div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 规则列表 -->
    <div class="rules-section">
      <el-card class="rules-card">
        <template #header>
          <div class="card-header">
            <span>规则列表</span>
            <div class="header-actions">
              <el-dropdown @command="handleExport" trigger="click">
                <el-button type="success">
                  <el-icon><Download /></el-icon> 导出规则
                </el-button>
                <template #dropdown>
                  <el-dropdown-menu>
                    <el-dropdown-item command="general">导出通用规则</el-dropdown-item>
                    <el-dropdown-item command="industry">导出行业规则</el-dropdown-item>
                    <el-dropdown-item command="all">导出全部规则</el-dropdown-item>
                  </el-dropdown-menu>
                </template>
              </el-dropdown>
              <el-upload
                ref="uploadRef"
                :auto-upload="false"
                :show-file-list="false"
                accept=".json"
                :on-change="handleFileImport"
                style="display: inline-block; margin-left: 10px"
              >
                <el-button type="warning">
                  <el-icon><Upload /></el-icon> 导入规则
                </el-button>
              </el-upload>
              <el-button type="primary" @click="showAddDialog" style="margin-left: 10px">
                <el-icon><Plus /></el-icon> 添加规则
              </el-button>
            </div>
          </div>
        </template>

        <el-tabs v-model="activeTab">
          <el-tab-pane label="通用规则" name="general">
            <div class="tab-toolbar">
              <el-button type="info" plain size="small" @click="downloadTemplate('general')">
                <el-icon><Download /></el-icon> 下载模板
              </el-button>
            </div>
            <el-table :data="generalRules" stripe style="width: 100%">
              <el-table-column prop="id" label="规则ID" width="100" />
              <el-table-column prop="patterns" label="匹配模式" width="200">
                <template #default="{ row }">
                  <el-tag v-for="p in row.patterns" :key="p" size="small" style="margin-right: 5px">
                    {{ p }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="category" label="分类" width="180">
                <template #default="{ row }">
                  <el-tag type="success">{{ row.category }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="grading" label="分级" width="120">
                <template #default="{ row }">
                  <el-tag type="warning">{{ row.grading }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="weight" label="权重" width="80">
                <template #default="{ row }">
                  {{ (row.weight * 100).toFixed(0) }}%
                </template>
              </el-table-column>
              <el-table-column label="操作" width="150">
                <template #default="{ row }">
                  <el-button type="primary" link @click="editRule(row)">编辑</el-button>
                  <el-button type="danger" link @click="deleteRule(row)">删除</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>

          <el-tab-pane label="行业规则" name="industry">
            <div class="tab-toolbar">
              <el-select v-model="selectedIndustry" placeholder="选择行业" size="small" style="width: 120px; margin-right: 10px">
                <el-option v-for="ind in industries" :key="ind" :label="ind" :value="ind" />
              </el-select>
              <el-button type="info" plain size="small" @click="downloadTemplate('industry')">
                <el-icon><Download /></el-icon> 下载模板
              </el-button>
            </div>
            <el-table :data="industryRules" stripe style="width: 100%">
              <el-table-column prop="field" label="字段名" width="180" />
              <el-table-column prop="category" label="分类" width="180">
                <template #default="{ row }">
                  <el-tag type="success">{{ row.category }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="grading" label="分级" width="120">
                <template #default="{ row }">
                  <el-tag type="warning">{{ row.grading }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="description" label="描述" />
              <el-table-column label="操作" width="150">
                <template #default="{ row }">
                  <el-button type="primary" link @click="editRule(row, true)">编辑</el-button>
                  <el-button type="danger" link @click="deleteRule(row, true)">删除</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>

          <el-tab-pane label="冲突检测" name="conflicts">
            <div class="conflict-toolbar">
              <el-button type="info" plain size="small" @click="detectConflicts">
                <el-icon><Refresh /></el-icon> 重新检测
              </el-button>
            </div>
            <el-alert v-if="conflicts.length === 0" type="success" :closable="false">
              未检测到规则冲突，所有规则都协调一致
            </el-alert>
            <el-table v-else :data="conflicts" stripe style="width: 100%">
              <el-table-column prop="field" label="字段/模式" width="180" />
              <el-table-column prop="issue" label="冲突说明" />
              <el-table-column label="相关规则" width="300">
                <template #default="{ row }">
                  <el-tag
                    v-for="(entry, idx) in row.entries"
                    :key="idx"
                    :type="entry.type.includes('通用') ? 'primary' : 'success'"
                    style="margin-right: 5px; margin-bottom: 3px"
                  >
                    {{ entry.type }}: {{ entry.category }} / {{ entry.grading }}
                  </el-tag>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>
        </el-tabs>
      </el-card>
    </div>

    <!-- 添加/编辑规则对话框 -->
    <el-dialog v-model="dialogVisible" :title="isEdit ? '编辑规则' : '添加规则'" width="500px">
      <el-form :model="ruleForm" label-width="100px">
        <el-form-item label="字段名/模式">
          <el-input v-model="ruleForm.field" placeholder="如: customer_id 或正则表达式" />
        </el-form-item>
        <el-form-item label="分类">
          <el-select v-model="ruleForm.category" placeholder="选择分类">
            <el-option v-for="label in classificationLabels" :key="label" :label="label" :value="label" />
          </el-select>
        </el-form-item>
        <el-form-item label="分级">
          <el-select v-model="ruleForm.grading" placeholder="选择分级">
            <el-option v-for="label in gradingLabels" :key="label" :label="label" :value="label" />
          </el-select>
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="ruleForm.description" type="textarea" />
        </el-form-item>
        <el-form-item v-if="activeTab === 'industry'" label="行业">
          <el-select v-model="ruleForm.industry" placeholder="选择行业">
            <el-option v-for="ind in industries" :key="ind" :label="ind" :value="ind" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="saveRule">确定</el-button>
      </template>
    </el-dialog>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Setting, Briefcase, Document, Warning, Download, Upload, Refresh, House, DataAnalysis } from '@element-plus/icons-vue'
import { getKnowledgeStats, getRules, addRule, deleteRule as apiDeleteRule } from '../api/knowledge'

const API_BASE_URL = 'http://localhost:8001'

const activeTab = ref('general')
const selectedIndustry = ref('金融')
const dialogVisible = ref(false)
const isEdit = ref(false)

const router = useRouter()
const currentRoute = ref('/knowledge')

// 原始行业规则数据（按行业存储）
const allIndustryRules = ref({})
const generalRules = ref([])

// 冲突列表
const conflicts = ref([])

// 统计相关
const totalRules = computed(() => generalRules.value.length)
const totalIndustryRules = computed(() => {
  return Object.values(allIndustryRules.value).reduce((sum, rules) => sum + (rules?.length || 0), 0)
})
const cachedFields = computed(() => totalRules.value + totalIndustryRules.value)
const conflictCount = computed(() => conflicts.value.length)

// 行业规则：根据当前选择的行业动态计算
const industryRules = computed(() => {
  return allIndustryRules.value[selectedIndustry.value] || []
})

// 监听行业切换，切换后重新加载该行业的规则
watch(selectedIndustry, () => {
  // industryRules 是 computed，会自动更新
})

const ruleForm = ref({
  field: '',
  category: '',
  grading: '',
  description: '',
  industry: ''
})

const industries = ['金融', '医疗', '教育', '工业', '商业', '政务', '其他']

const classificationLabels = [
  'ID类/主键ID', '结构类/分类代码', '结构类/产品代码', '结构类/企业代码', '结构类/标准代码',
  '属性类/名称标题', '属性类/类别标签', '属性类/描述文本', '属性类/技能标签', '属性类/地址位置',
  '度量类/计量数值', '度量类/计数统计', '度量类/比率比例', '度量类/时间度量', '度量类/序号排序',
  '身份类/人口统计', '身份类/联系方式', '身份类/教育背景', '身份类/职业信息',
  '状态类/二元标志', '状态类/状态枚举', '状态类/时间标记',
  '扩展类/扩展代码', '扩展类/其他字段'
]

const gradingLabels = ['第1级/公开', '第2级/内部', '第3级/敏感', '第4级/机密']

const loadData = async () => {
  // 先尝试从后端加载
  await loadFromStorage()

  // 如果后端和 localStorage 都没有数据，使用默认数据
  if (generalRules.value.length === 0) {
    generalRules.value = [
      { id: 'rule_001', patterns: ['^id$', '_id$'], category: 'ID类/主键ID', grading: '第1级/公开', weight: 1.0 },
      { id: 'rule_002', patterns: ['name', '名称'], category: '属性类/名称标题', grading: '第1级/公开', weight: 0.9 },
      { id: 'rule_003', patterns: ['price', '金额'], category: '度量类/计量数值', grading: '第2级/内部', weight: 0.95 }
    ]
  }

  // 初始化行业规则（如果不存在）
  if (Object.keys(allIndustryRules.value).length === 0) {
    allIndustryRules.value = {
      '金融': [
        { id: 'fin_001', field: 'account_no', category: '身份类/金融账户', grading: '第4级/机密', description: '银行账号' },
        { id: 'fin_002', field: 'balance', category: '度量类/计量数值', grading: '第3级/敏感', description: '账户余额' }
      ],
      '医疗': [
        { id: 'med_001', field: 'patient_id', category: 'ID类/主键ID', grading: '第3级/敏感', description: '患者ID' }
      ],
      '教育': [],
      '工业': [],
      '商业': [],
      '政务': [],
      '其他': []
    }
  }

  // 检测冲突
  await detectConflicts()
}

const showAddDialog = () => {
  isEdit.value = false
  ruleForm.value = {
    field: '',
    category: '',
    grading: '',
    description: '',
    industry: selectedIndustry.value
  }
  dialogVisible.value = true
}

const editRule = (rule, isIndustry = false) => {
  isEdit.value = true
  ruleForm.value = {
    ...rule,
    field: rule.field || rule.patterns?.[0] || '',
    industry: isIndustry ? selectedIndustry.value : ''
  }
  dialogVisible.value = true
}

const saveRule = async () => {
  if (!ruleForm.value.field || !ruleForm.value.category || !ruleForm.value.grading) {
    ElMessage.warning('请填写完整信息')
    return
  }

  if (isEdit.value) {
    // 编辑模式：更新现有规则
    if (activeTab.value === 'industry') {
      const rules = allIndustryRules.value[selectedIndustry.value] || []
      const index = rules.findIndex(r => r.id === ruleForm.value.id)
      if (index !== -1) {
        rules[index] = { ...ruleForm.value }
        allIndustryRules.value[selectedIndustry.value] = [...rules]
      }
    } else {
      const index = generalRules.value.findIndex(r => r.id === ruleForm.value.id)
      if (index !== -1) {
        generalRules.value[index] = { ...ruleForm.value, patterns: [ruleForm.value.field] }
      }
    }
  } else {
    // 添加模式
    const newRule = {
      id: `${activeTab.value === 'industry' ? selectedIndustry.value.substring(0, 3) : 'rule'}_${Date.now()}`,
      field: ruleForm.value.field,
      patterns: activeTab.value === 'general' ? [ruleForm.value.field] : undefined,
      category: ruleForm.value.category,
      grading: ruleForm.value.grading,
      description: ruleForm.value.description,
      weight: ruleForm.value.weight || 0.9
    }

    if (activeTab.value === 'industry') {
      if (!allIndustryRules.value[selectedIndustry.value]) {
        allIndustryRules.value[selectedIndustry.value] = []
      }
      allIndustryRules.value[selectedIndustry.value].push(newRule)
    } else {
      generalRules.value.push(newRule)
    }
  }

  // 保存到 localStorage
  saveToStorage()
  dialogVisible.value = false
  ElMessage.success(isEdit.value ? '规则更新成功' : '规则添加成功')
}

// localStorage 保存
const saveToStorage = async () => {
  localStorage.setItem('kb_general_rules', JSON.stringify(generalRules.value))
  localStorage.setItem('kb_industry_rules', JSON.stringify(allIndustryRules.value))

  // 同时保存到后端
  try {
    await fetch(`${API_BASE_URL}/knowledge/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        general_rules: generalRules.value,
        industry_rules: allIndustryRules.value
      })
    })
  } catch (e) {
    console.error('同步到后端失败:', e)
  }
}

// 检测冲突
const detectConflicts = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/knowledge/conflicts`)
    if (res.ok) {
      const data = await res.json()
      conflicts.value = data.conflicts || []
    }
  } catch (e) {
    console.error('检测冲突失败:', e)
  }
}

// localStorage 加载
const loadFromStorage = async () => {
  // 先尝试从后端加载
  try {
    const res = await fetch(`${API_BASE_URL}/knowledge/rules`)
    if (res.ok) {
      const data = await res.json()
      if (data.general_rules && data.general_rules.length > 0) {
        generalRules.value = data.general_rules
        localStorage.setItem('kb_general_rules', JSON.stringify(data.general_rules))
      }
      if (data.industry_rules && Object.keys(data.industry_rules).length > 0) {
        allIndustryRules.value = data.industry_rules
        localStorage.setItem('kb_industry_rules', JSON.stringify(data.industry_rules))
      }
      return
    }
  } catch (e) {
    console.error('从后端加载失败:', e)
  }

  // 后端加载失败，从 localStorage 加载
  const savedGeneral = localStorage.getItem('kb_general_rules')
  const savedIndustry = localStorage.getItem('kb_industry_rules')

  if (savedGeneral) {
    try {
      generalRules.value = JSON.parse(savedGeneral)
    } catch (e) {
      console.error('加载通用规则失败:', e)
    }
  }

  if (savedIndustry) {
    try {
      allIndustryRules.value = JSON.parse(savedIndustry)
    } catch (e) {
      console.error('加载行业规则失败:', e)
    }
  }
}

const deleteRule = async (rule, isIndustry = false) => {
  try {
    await ElMessageBox.confirm('确定要删除这条规则吗?', '提示', {
      type: 'warning'
    })

    if (isIndustry) {
      // 删除行业规则
      const rules = allIndustryRules.value[selectedIndustry.value] || []
      allIndustryRules.value[selectedIndustry.value] = rules.filter(r => r.id !== rule.id)
    } else {
      // 删除通用规则
      generalRules.value = generalRules.value.filter(r => r.id !== rule.id)
    }

    // 保存并同步
    await saveToStorage()
    // 重新检测冲突
    await detectConflicts()
    ElMessage.success('规则删除成功')
  } catch {
    // 用户取消
  }
}

onMounted(() => {
  loadData()
})

// 导出规则
const handleExport = (type) => {
  let data, filename, content

  if (type === 'general') {
    data = generalRules.value
    filename = 'general_rules.json'
    content = JSON.stringify(data, null, 2)
  } else if (type === 'industry') {
    // 导出所有行业规则
    filename = 'industry_rules.json'
    content = JSON.stringify(allIndustryRules.value, null, 2)
  } else {
    // 导出全部
    content = JSON.stringify({
      general_rules: generalRules.value,
      industry_rules: allIndustryRules.value
    }, null, 2)
    filename = 'all_rules.json'
  }

  const blob = new Blob([content], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
  ElMessage.success(`已导出: ${filename}`)
}

// 导入规则
const handleFileImport = async (file) => {
  try {
    const text = await file.raw.text()
    const data = JSON.parse(text)

    // 判断是通用规则还是行业规则
    if (Array.isArray(data)) {
      // 通用规则
      const validRules = data.filter(r => r.id && r.patterns && r.category && r.grading)
      if (validRules.length > 0) {
        generalRules.value = [...generalRules.value, ...validRules]
        ElMessage.success(`成功导入 ${validRules.length} 条通用规则`)
      } else {
        ElMessage.warning('未找到有效的通用规则')
      }
    } else if (typeof data === 'object') {
      // 行业规则
      let totalCount = 0
      for (const [industry, rules] of Object.entries(data)) {
        if (Array.isArray(rules)) {
          const validRules = rules.filter(r => r.id && r.field && r.category && r.grading)
          if (validRules.length > 0) {
            // 更新对应行业的规则
            if (!allIndustryRules.value[industry]) {
              allIndustryRules.value[industry] = []
            }
            allIndustryRules.value[industry] = [...allIndustryRules.value[industry], ...validRules]
            totalCount += validRules.length
          }
        }
      }
      if (totalCount > 0) {
        ElMessage.success(`成功导入 ${totalCount} 条行业规则`)
      } else {
        ElMessage.warning('未找到有效的行业规则')
      }
    }

      // 保存并同步
      await saveToStorage()
      // 重新检测冲突
      await detectConflicts()
  } catch (e) {
    ElMessage.error('导入失败: 文件格式错误')
  }
}

// 下载模板
const downloadTemplate = (type) => {
  let content, filename

  if (type === 'general') {
    filename = 'general_rules_template.json'
    content = JSON.stringify([
      {
        id: 'rule_001',
        patterns: ['字段名', '正则表达式'],
        category: '属性类/名称标题',
        grading: '第1级/公开',
        weight: 0.9,
        description: '规则描述'
      }
    ], null, 2)
  } else {
    filename = `industry_rules_${selectedIndustry.value}_template.json`
    const templateData = {}
    templateData[selectedIndustry.value] = [
      {
        id: `${selectedIndustry.value.substring(0, 3)}_001`,
        field: '字段名',
        category: '属性类/名称标题',
        grading: '第1级/公开',
        description: '规则描述'
      }
    ]
    content = JSON.stringify(templateData, null, 2)
  }

  const blob = new Blob([content], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
  ElMessage.success(`已下载模板: ${filename}`)
}
</script>

<style lang="scss" scoped>
.knowledge-container {
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

.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.tab-toolbar {
  margin-bottom: 15px;
}

.stat-card {
  display: flex;
  align-items: center;
  padding: 20px;
  border-radius: 12px;
  background: white;
  border: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  transition: transform 0.3s, box-shadow 0.3s;

  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
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
  &.teal { background: linear-gradient(135deg, #319795 0%, #285e61 100%); }
  &.green { background: linear-gradient(135deg, #68a67d 0%, #4a8a5f 100%); }
  &.red { background: linear-gradient(135deg, #b05a5a 0%, #904040 100%); }
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

.rules-section {
  max-width: 1200px;
  margin: 0 auto;
}

.rules-card {
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  border: none;

  :deep(.el-card__header) {
    border-bottom: 1px solid #e2e8f0;
    color: #2d3748;
    font-weight: 500;
  }
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

:deep(.el-tabs__content) {
  overflow: visible;
}

:deep(.el-tabs__header) {
  margin-bottom: 15px;
}

:deep(.el-tab-pane) {
  width: 100%;
}

:deep(.el-table) {
  width: 100% !important;
  table-layout: fixed;
}

:deep(.el-table__header) {
  width: 100% !important;
}

:deep(.el-table__body) {
  width: 100% !important;
}

:deep(.el-table__body-wrapper) {
  width: 100% !important;
}

:deep(.el-table__inner-wrapper) {
  width: 100% !important;
}

.tab-toolbar {
  margin-bottom: 15px;
  display: flex;
  align-items: center;
}

.conflict-toolbar {
  margin-bottom: 15px;
}

:deep(.el-form-item) {
  margin-bottom: 18px;
}
</style>
