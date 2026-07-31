/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "surface-dim": "#0d131f",
        "inverse-primary": "#006a63",
        "surface-bright": "#333946",
        "outline": "#869491",
        "tertiary-container": "#ffab67",
        "on-secondary": "#223144",
        "on-tertiary-fixed-variant": "#6f3800",
        "on-primary": "#003733",
        "background": "#0d131f",
        "inverse-surface": "#dde2f3",
        "primary": "#6feee1",
        "surface-container-lowest": "#080e1a",
        "on-primary-fixed": "#00201d",
        "inverse-on-surface": "#2a303d",
        "surface-container-high": "#242a36",
        "on-secondary-container": "#a7b6ce",
        "primary-container": "#4fd1c5",
        "outline-variant": "#3c4947",
        "surface": "#0d131f",
        "on-surface-variant": "#bbc9c7",
        "secondary": "#b8c8e0",
        "on-tertiary-fixed": "#2f1500",
        "secondary-fixed": "#d4e4fc",
        "tertiary": "#ffd1af",
        "surface-tint": "#5adace",
        "surface-container-low": "#161c27",
        "on-tertiary": "#4e2600",
        "on-secondary-fixed-variant": "#39485c",
        "secondary-container": "#39485c",
        "surface-container-highest": "#2f3542",
        "error-container": "#93000a",
        "on-error-container": "#ffdad6",
        "on-tertiary-container": "#773d00",
        "error": "#ffb4ab",
        "tertiary-fixed": "#ffdcc4",
        "on-surface": "#dde2f3",
        "tertiary-fixed-dim": "#ffb77f",
        "on-background": "#dde2f3",
        "primary-fixed": "#79f7ea",
        "primary-fixed-dim": "#5adace",
        "surface-variant": "#2f3542",
        "on-primary-fixed-variant": "#00504a",
        "on-secondary-fixed": "#0d1c2e",
        "secondary-fixed-dim": "#b8c8e0",
        "on-error": "#690005",
        "surface-container": "#1a202c",
        "on-primary-container": "#005750"
      },
      borderRadius: {
        "DEFAULT": "0.25rem",
        "lg": "0.5rem",
        "xl": "0.75rem",
        "full": "9999px"
      },
      spacing: {
        "xs": "4px",
        "gutter": "20px",
        "md": "24px",
        "base": "8px",
        "margin": "24px",
        "sm": "12px",
        "lg": "48px",
        "xl": "80px"
      },
      fontFamily: {
        "label-caps": ["Inter"],
        "headline-lg-mobile": ["Inter"],
        "headline-lg": ["Inter"],
        "headline-xl": ["Inter"],
        "body-md": ["Inter"],
        "mono-label": ["JetBrains Mono"],
        "body-sm": ["Inter"],
        "label-sm": ["JetBrains Mono"]
      },
      fontSize: {
        "label-caps": ["12px", { lineHeight: "16px", letterSpacing: "0.05em", fontWeight: "600" }],
        "headline-lg-mobile": ["24px", { lineHeight: "32px", fontWeight: "600" }],
        "headline-lg": ["28px", { lineHeight: "36px", letterSpacing: "-0.01em", fontWeight: "600" }],
        "headline-xl": ["36px", { lineHeight: "44px", letterSpacing: "-0.02em", fontWeight: "700" }],
        "body-md": ["16px", { lineHeight: "24px", fontWeight: "400" }],
        "mono-label": ["13px", { lineHeight: "18px", fontWeight: "500" }],
        "body-sm": ["14px", { lineHeight: "20px", fontWeight: "400" }],
        "label-sm": ["12px", { lineHeight: "16px", letterSpacing: "0.05em", fontWeight: "500" }]
      },
      keyframes: {
        scan: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(400%)' },
        }
      }
    }
  },
  plugins: [
    require('@tailwindcss/typography'),
    require('@tailwindcss/forms'),
    require('@tailwindcss/container-queries')
  ],
}
