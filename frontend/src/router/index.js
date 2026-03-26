import { createRouter, createWebHistory } from 'vue-router'
import ClassificationView from '../views/ClassificationView.vue'
import KnowledgeBaseView from '../views/KnowledgeBaseView.vue'
import DashboardView from '../views/DashboardView.vue'

const routes = [
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: DashboardView
  },
  {
    path: '/classify',
    name: 'Classification',
    component: ClassificationView
  },
  {
    path: '/knowledge',
    name: 'KnowledgeBase',
    component: KnowledgeBaseView
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/dashboard'
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
