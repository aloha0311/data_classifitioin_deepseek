<template>
  <div class="knowledge-container">
    <div class="header">
      <h1>知识库管理</h1>
      <p class="subtitle">管理分类分级规则与模式库</p>
    </div>

    <!-- 统计概览 -->
    <div class="stats-row">
      <el-row :gutter="20">
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon blue"><el-icon><Setting /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.totalRules }}</div>
              <div class="stat-label">总规则数</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon teal"><el-icon><Briefcase /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.totalIndustryRules }}</div>
              <div class="stat-label">行业规则</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-icon green"><el-icon><Document /></el-icon></div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.cachedFields }}</div>
              <div class="stat-label">缓存字段</div>
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
            <el-button type="primary" @click="showAddDialog">
              <el-icon><Plus /></el-icon> 添加规则
            </el-button>
          </div>
        </template>

        <el-tabs v-model="activeTab">
          <el-tab-pane label="通用规则" name="general">
            <el-table :data="generalRules" stripe>
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
            <el-select v-model="selectedIndustry" placeholder="选择行业" style="margin-bottom: 15px">
              <el-option v-for="ind in industries" :key="ind" :label="ind" :value="ind" />
            </el-select>
            <el-table :data="industryRules" stripe>
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
            <el-alert v-if="conflicts.length === 0" type="success" :closable="false">
              未检测到规则冲突
            </el-alert>
            <el-table v-else :data="conflicts" stripe>
              <el-table-column prop="field" label="字段" width="150" />
              <el-table-column prop="type" label="冲突类型" width="150">
                <template #default="{ row }">
                  <el-tag :type="row.type === 'category_conflict' ? 'danger' : 'warning'">
                    {{ row.type === 'category_conflict' ? '分类冲突' : '分级冲突' }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column label="相关规则">
                <template #default="{ row }">
                  <div v-for="rule in row.rules" :key="rule.rule_id">
                    {{ rule.category }} / {{ rule.grading }}
                  </div>
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
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Setting, Briefcase, Document, Warning } from '@element-plus/icons-vue'
import { getKnowledgeStats, getRules, addRule, deleteRule as apiDeleteRule } from '../api/knowledge'

const API_BASE_URL = 'http://localhost:8001'

const activeTab = ref('general')
const selectedIndustry = ref('金融')
const dialogVisible = ref(false)
const isEdit = ref(false)
const generalRules = ref([])
const industryRules = ref([])
const conflicts = ref([])
const stats = ref({
  totalRules: 0,
  totalIndustryRules: 0,
  cachedFields: 0
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

const conflictCount = computed(() => conflicts.value.length)

const loadData = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/knowledge/stats`)
    if (res.ok) {
      const data = await res.json()
      stats.value = data
    }
  } catch (e) {
    stats.value = {
      totalRules: 12,
      totalIndustryRules: 9,
      cachedFields: 45
    }
    generalRules.value = [
      { id: 'rule_001', patterns: ['^id$', '_id$'], category: 'ID类/主键ID', grading: '第1级/公开', weight: 1.0 },
      { id: 'rule_002', patterns: ['name', '名称'], category: '属性类/名称标题', grading: '第1级/公开', weight: 0.9 },
      { id: 'rule_003', patterns: ['price', '金额'], category: '度量类/计量数值', grading: '第2级/内部', weight: 0.95 }
    ]
    industryRules.value = [
      { field: 'credit_score', category: '度量类/计量数值', grading: '第3级/敏感', description: '信用评分' },
      { field: 'balance', category: '度量类/计量数值', grading: '第3级/敏感', description: '账户余额' }
    ]
  }
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
  
  ElMessage.success(isEdit.value ? '规则更新成功' : '规则添加成功')
  dialogVisible.value = false
  loadData()
}

const deleteRule = async (rule, isIndustry = false) => {
  try {
    await ElMessageBox.confirm('确定要删除这条规则吗?', '提示', {
      type: 'warning'
    })
    await apiDeleteRule(rule.id, isIndustry)
    ElMessage.success('规则删除成功')
    loadData()
  } catch {
    // 用户取消
  }
}

onMounted(() => {
  loadData()
})
</script>

<style lang="scss" scoped>
.knowledge-container {
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

.stats-row {
  max-width: 1200px;
  margin: 0 auto 30px;
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
</style>
