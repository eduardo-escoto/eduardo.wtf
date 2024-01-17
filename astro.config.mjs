import mdx from '@astrojs/mdx'
import sitemap from '@astrojs/sitemap'
import tailwind from '@astrojs/tailwind'
import { defineConfig } from 'astro/config'

import rehypeAutolinkHeadings from 'rehype-autolink-headings'
import rehypeKatex from 'rehype-katex'
import rehypeShikiji from 'rehype-shikiji'
import rehypeSlug from 'rehype-slug'

import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkReadingTime from 'remark-reading-time'
import readingMdxTime from 'remark-reading-time/mdx'
import remarkSmartypants from 'remark-smartypants'
import remarkToc from 'remark-toc'

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
      rehypeSlug,
      [rehypeAutolinkHeadings, { behavior: 'wrap' }],
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
    remarkPlugins: [
      remarkMath,
      remarkSmartypants,
      remarkGfm,
      remarkToc,
      remarkReadingTime,
      readingMdxTime,
    ],
  },
})
