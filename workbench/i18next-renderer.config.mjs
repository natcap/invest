import { defineConfig } from 'i18next-cli';

export default defineConfig({
  locales: ['en', 'es', 'zh'],
  extract: {
    keySeparator: false,
    nsSeparator: false,
    input: ['src/renderer/**/*.{js,jsx,ts,tsx}'],
    output: 'src/renderer/i18n/{{language}}.json',
  },
});
