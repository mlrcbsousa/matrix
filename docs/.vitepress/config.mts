import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "matrix",
  description: "Documentation for the 42 School Project matrix",
  base: '/matrix/',
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      // { text: 'Home', link: '/' },
    ],

    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'Start here', link: '/introduction' },
          { text: 'Testing', link: '/testing' },
          { text: 'Core Structures', link: '/core-structures' },
        ]
      },
      {
        text: 'Exercises',
        items: [
          { text: 'Exercise 00', link: '/ex00' },
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/mlrcbsousa/matrix' }
    ]
  }
})
