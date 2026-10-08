import { defineConfig } from 'i18next-cli';

export default defineConfig({
  locales: ['en', 'es', 'zh'],
  extract: {
    keySeparator: false,
    nsSeparator: false,
    input: ['src/main/**/*.{js,jsx,ts,tsx}'],
    output: 'src/main/i18n/{{language}}.json',
  },
});
