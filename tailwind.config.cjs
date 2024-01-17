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
        primary: colors.pink,
        gray: colors.gray,
        white: '#ffffff',
        pre_grey_text: '#4C4F69',
        pre_grey_dark_text: '#CDD6F4',
        pre_bg_grey: '#eff1f5',
        pre_bg_dark_grey: '#1e1e2e',
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
                color: theme('colors.primary.500'),
              },
            },
            'h1,h2': {
              fontWeight: '700',
              letterSpacing: theme('letterSpacing.tight'),
            },
            h3: {
              fontWeight: '600',
            },
            code: {
              color: theme('colors.indigo.500'),
            },
            pre: {
              color: theme('colors.pre_grey_text'),
              backgroundColor: theme('colors.pre_bg_grey'),
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
            'h1,h2,h3,h4,h5,h6': {
              color: theme('colors.gray.100'),
            },
            img: {
              backgroundColor: theme('colors.white'),
            },
            pre: {
              color: theme('colors.pre_grey_dark_text'),
              backgroundColor: theme('colors.pre_bg_dark_grey'),
            },
          },
        },
      }),
    },
  },
  plugins: [require('@tailwindcss/forms'), require('@tailwindcss/typography')],
}
