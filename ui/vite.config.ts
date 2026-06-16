import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://scn26-10g:8001',
      '/ws': {
        target: 'http://scn26-10g:8001',
        ws: true,
      },
    },
  },
})
