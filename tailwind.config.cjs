const colors = require('tailwindcss/colors')
const { fontFamily } = require('tailwindcss/defaultTheme')

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      lineHeight: {
        11: '2.75rem',
        12: '3rem',
        13: '3.25rem',
        14: '3.5rem',
      },
      fontFamily: {
        sans: ['Space Grotesk', ...fontFamily.sans],
      },
      colors: {
        primary: colors.emerald,
        gray: colors.stone,
        'code-light-bg': colors.stone[100], // vitesse-light bg not dark enough
        'code-light-fg': '#393A34', //vitesse light
        'code-dark-bg': '#121212', // vitesse dark
        'code-dark-fg': '#DBD7CAEE', //vitesse dark
      },
      typography: ({ theme }) => ({
        DEFAULT: {
          css: {
            a: {
              color: theme('colors.primary.500'),
              '&:hover': {
                color: `${theme('colors.primary.600')}`,
              },
              code: { color: theme('colors.primary.400') },
            },
            '.header-link': {
              textDecoration: 'none',
              color: theme('colors.gray.900'),
              transition: '0.2s',
              '&:hover': {
                color: theme('colors.primary.600'),
              },
            },
            'h1,h2': {
              fontWeight: '700',
              letterSpacing: theme('letterSpacing.tight'),
            },
            h3: {
              fontWeight: '600',
            },
            pre: {
              color: theme('colors.code-light-fg'),
              backgroundColor: theme('colors.code-light-bg'),
            },
            'p > code': {
              color: theme('colors.code-light-fg'),
              backgroundColor: theme('colors.code-light-bg'),
              paddingLeft: '8px',
              paddingRight: '8px',
              paddingTop: '3px',
              paddingBottom: '3px',
              borderRadius: '6px',
            },
            'code::before': {
              content: '',
            },
            'code::after': {
              content: '',
            },
          },
        },
        invert: {
          css: {
            a: {
              color: theme('colors.primary.500'),
              '&:hover': {
                color: `${theme('colors.primary.400')}`,
              },
              code: { color: theme('colors.primary.400') },
            },
            '.header-link': {
              textDecoration: 'none',
              color: theme('colors.gray.100'),
              transition: '0.2s',
              '&:hover': {
                color: theme('colors.primary.400'),
              },
            },
            'h1,h2,h3,h4,h5,h6': {
              color: theme('colors.gray.100'),
            },
            img: {
              backgroundColor: theme('colors.white'),
            },
            pre: {
              color: theme('colors.code-dark-fg'),
              backgroundColor: theme('colors.code-dark-bg'),
            },
            'p > code': {
              color: theme('colors.code-dark-fg'),
              backgroundColor: theme('colors.code-dark-bg'),
              paddingLeft: '8px',
              paddingRight: '8px',
              paddingTop: '3px',
              paddingBottom: '3px',
              borderRadius: '6px',
            },
            'p > code::before': {
              content: '',
            },
            'p > code::after': {
              content: '',
            },
          },
        },
      }),
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
    function ({ addBase, theme }) {
      function extractColorVars(colorObj, colorGroup = '') {
        return Object.keys(colorObj).reduce((vars, colorKey) => {
          let newVars = {}
          const value = colorObj[colorKey]
          const save_vars = ['code-light-bg', 'code-light-fg', 'code-dark-bg', 'code-dark-fg']
          if (save_vars.includes(colorKey)) {
            newVars =
              typeof value === 'string'
                ? { [`--color${colorGroup}-${colorKey}`]: value }
                : extractColorVars(value, `-${colorKey}`)
          }

          return { ...vars, ...newVars }
        }, {})
      }

      addBase({
        ':root': extractColorVars(theme('colors')),
      })
    },
  ],
}
