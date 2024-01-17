import mdx from '@astrojs/mdx'
import sitemap from '@astrojs/sitemap'
import tailwind from '@astrojs/tailwind'
import { defineConfig } from 'astro/config'

import rehypeKatex from 'rehype-katex'
import rehypeShikiji from 'rehype-shikiji'
import remarkMath from 'remark-math'

import { transformerNotationDiff } from 'shikiji-transformers'

import { SITE_METADATA } from './src/consts.ts'

// https://astro.build/config
export default defineConfig({
  prefetch: true,
  site: SITE_METADATA.siteUrl,
  integrations: [mdx(), sitemap(), tailwind()],
  markdown: {
    syntaxHighlight: false,
    rehypePlugins: [
      rehypeKatex,
      [
        rehypeShikiji,
        {
          themes: {
            light: 'catppuccin-latte',
            dark: 'catppuccin-mocha',
          },
          transformers: [transformerNotationDiff()],
          wrap: true,
        },
      ],
    ],
    remarkPlugins: [remarkMath],
  },
})
