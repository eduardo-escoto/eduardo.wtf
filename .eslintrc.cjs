module.exports = {
  root: true,
  parser: '@typescript-eslint/parser',
  env: {
    browser: true,
    amd: true,
    node: true,
    es6: true,
  },
  plugins: ['@typescript-eslint'],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/eslint-recommended',
    'plugin:@typescript-eslint/recommended',
    // 'plugin:jsx-a11y/recommended',
    'plugin:prettier/recommended',
    'plugin:astro/recommended',
    'plugin:import/recommended',
    'plugin:import/typescript',
  ],
  overrides: [
    {
      // Define the configuration for `.astro` file.
      files: ['*.astro'],
      // Allows Astro components to be parsed.
      parser: 'astro-eslint-parser',
      // Parse the script in `.astro` as TypeScript by adding the following configuration.
      // It's the setting you need when using TypeScript.
      parserOptions: {
        parser: '@typescript-eslint/parser',
        extraFileExtensions: ['.astro'],
      },
    },
  ],
  parserOptions: {
    project: true,
    tsconfigRootDir: __dirname,
  },
  rules: {
    'prettier/prettier': 'error',
    'jsx-a11y/anchor-is-valid': [
      'off',
      {
        components: ['Link'],
        specialLink: ['hrefLeft', 'hrefRight'],
        aspects: ['invalidHref', 'preferButton'],
      },
    ],
    'sort-imports': [
      'error',
      {
        ignoreCase: true,
        ignoreDeclarationSort: true,
      },
    ],
    '@typescript-eslint/no-unused-vars': 0,
    '@typescript-eslint/explicit-module-boundary-types': 'off',
    '@typescript-eslint/no-var-requires': 'off',
    '@typescript-eslint/ban-ts-comment': 'off',
    'import/no-unresolved': 'off',
    'import/order': [
      1,
      {
        groups: ['external', 'builtin', 'internal', 'sibling', 'parent', 'index'],
        pathGroups: [
          {
            pattern: 'components',
            group: 'internal',
          },
          {
            pattern: 'layouts',
            group: 'internal',
          },
          {
            pattern: 'styles',
            group: 'internal',
          },
          {
            pattern: 'content',
            group: 'internal',
          },
          {
            pattern: 'public',
            group: 'internal',
            position: 'after',
          },
        ],
        pathGroupsExcludedImportTypes: ['internal'],
        alphabetize: {
          order: 'asc',
          caseInsensitive: true,
        },
      },
    ],
  },
}
