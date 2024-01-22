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

import vitesseDark from 'shikiji/themes/vitesse-dark'
import vitesseLight from 'shikiji/themes/vitesse-light'

import {
  transformerMetaHighlight,
  transformerMetaWordHighlight,
  transformerNotationDiff,
  transformerNotationErrorLevel,
  transformerNotationFocus,
  transformerNotationHighlight,
  transformerNotationWordHighlight,
} from 'shikiji-transformers'

import { SITE_METADATA } from './src/consts.ts'

const vitesseLightOverride = {
  ...vitesseLight,
  bg: 'var(--color-code-light-bg)',
  fg: 'var(--color-code-light-fg)',
}

const vitesseDarkOverride = {
  ...vitesseDark,
  bg: 'var(--color-code-dark-bg)',
  fg: 'var(--color-code-dark-fg)',
}

const shikiji_transformers = [
  transformerMetaHighlight,
  transformerMetaWordHighlight,
  transformerNotationDiff,
  transformerNotationErrorLevel,
  transformerNotationFocus,
  transformerNotationHighlight,
  transformerNotationWordHighlight,
]

// https://astro.build/config
export default defineConfig({
  prefetch: true,
  site: SITE_METADATA.siteUrl,
  integrations: [mdx(), sitemap(), tailwind()],
  markdown: {
    syntaxHighlight: false,
    rehypePlugins: [
      rehypeSlug,
      [
        rehypeAutolinkHeadings,
        {
          behavior: 'wrap',
          properties: { class: 'header-link' },
        },
      ],
      rehypeKatex,
      [
        rehypeShikiji,
        {
          themes: {
            light: vitesseLightOverride,
            dark: vitesseDarkOverride,
          },
          transformers: shikiji_transformers,
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
