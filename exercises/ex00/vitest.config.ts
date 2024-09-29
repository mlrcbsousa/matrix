import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    includeSource: ['src/**/*.ts'],
    exclude: ['src/index.ts'],
  },
  define: {
    'import.meta.vitest': 'undefined',
  },
})