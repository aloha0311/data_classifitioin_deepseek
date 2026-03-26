import axios from 'axios'

const API_BASE_URL = 'http://localhost:8001'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

export const classifyAPI = {
  // 单字段分类
  classifyField: async (fieldName, industry, samples) => {
    const response = await apiClient.post('/classify', {
      field_name: fieldName,
      industry: industry,
      samples: samples
    })
    return response.data
  },

  // 批量分类
  classifyBatch: async (fields) => {
    const response = await apiClient.post('/classify/batch', {
      fields: fields
    })
    return response.data
  },

  // 文件分类
  classifyFile: async (file, industry) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('industry', industry)
    
    const response = await axios.post(`${API_BASE_URL}/classify/file`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 120000
    })
    return response.data
  }
}

export const knowledgeAPI = {
  // 获取统计信息
  getStats: async () => {
    const response = await apiClient.get('/knowledge/stats')
    return response.data
  },

  // 获取标签体系
  getLabels: async () => {
    const response = await apiClient.get('/labels')
    return response.data
  },

  // 获取行业列表
  getIndustries: async () => {
    const response = await apiClient.get('/industries')
    return response.data
  },

  // 添加规则
  addRule: async (rule) => {
    const response = await apiClient.post('/knowledge/rules', rule)
    return response.data
  },

  // 删除规则
  deleteRule: async (ruleId) => {
    const response = await apiClient.delete(`/knowledge/rules/${ruleId}`)
    return response.data
  },

  // 冲突检测
  detectConflicts: async () => {
    const response = await apiClient.get('/knowledge/conflicts')
    return response.data
  }
}

export const systemAPI = {
  // 健康检查
  healthCheck: async () => {
    const response = await apiClient.get('/health')
    return response.data
  },

  // 服务信息
  getServiceInfo: async () => {
    const response = await apiClient.get('/')
    return response.data
  }
}

export default {
  classify: classifyAPI,
  knowledge: knowledgeAPI,
  system: systemAPI
}
