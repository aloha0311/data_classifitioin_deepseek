import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
import path from 'path'

export default defineConfig({
  plugins: [
    vue(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
      imports: ['vue', 'vue-router', 'pinia']
    }),
    Components({
      resolvers: [ElementPlusResolver()]
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    historyApiFallback: true,
    proxy: {
      '/health': {
        target: 'http://localhost:8001',
        changeOrigin: true
      },
      '/classify/file/stream': {
        target: 'http://localhost:8001',
        changeOrigin: true
      },
      '/classify/batch': {
        target: 'http://localhost:8001',
        changeOrigin: true
      },
      '/labels': {
        target: 'http://localhost:8001',
        changeOrigin: true
      },
      '/industries': {
        target: 'http://localhost:8001',
        changeOrigin: true
      }
    }
  }
})
