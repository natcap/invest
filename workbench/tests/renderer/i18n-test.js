import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

i18n
  .use(initReactI18next)
  .init({
    lng: 'en',
    interpolation: {
      escapeValue: false,
    },
    resources: {
      ll: {
        translation: {
          Open: 'σρєи',
          Language: 'ℓαиgυαgє'
        }
      }
    },
  });

export default i18n;
