import { knowledgeAPI } from './index'

export const getKnowledgeStats = () => knowledgeAPI.getStats()
export const getRules = () => knowledgeAPI.getLabels()
export const addRule = (rule) => knowledgeAPI.addRule(rule)
export const deleteRule = (ruleId) => knowledgeAPI.deleteRule(ruleId)
export const getLabels = () => knowledgeAPI.getLabels()
export const getIndustries = () => knowledgeAPI.getIndustries()
export const detectConflicts = () => knowledgeAPI.detectConflicts()
