import remarkFrontmatter from 'remark-frontmatter'
import remarkParse from 'remark-parse'
import remarkStringify from 'remark-stringify'
import { read, toVFile } from 'to-vfile'
import { unified } from 'unified'
import { VFile } from 'vfile'

import { matter } from 'vfile-matter'
import fs from 'fs'
import path from 'path'

const base_path = './src/content/blog/'
const tag_base_path = './src/content/tags/'

function myUnifiedPluginHandlingYamlMatter() {
  return function (tree, file) {
    matter(file)
  }
}
const files = await fs.readdirSync(base_path, { recursive: true })

const tags = new Set(
  (
    await Promise.all(
      files
        .filter((file) => {
          return path.extname(file) === '.md'
        })
        .map(async (file) => {
          const parsed = await unified()
            .use(remarkParse)
            .use(remarkStringify)
            .use(remarkFrontmatter, ['yaml'])
            .use(myUnifiedPluginHandlingYamlMatter)
            .process(await read(path.resolve(base_path, file)))

          return await parsed.data.matter.tags
        })
    )
  ).flat()
).forEach((tag) => {
  const formattedTag = tag.charAt(0).toUpperCase() + tag.slice(1)
  const outputString = `---
name: ${formattedTag}
description: Posts about ${formattedTag}
---

This tag contains all posts that are related to ${formattedTag}.
`
  fs.writeFileSync(path.resolve(tag_base_path, `${tag}.mdx`), outputString)
})
