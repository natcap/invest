import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import esMessages from './es.json';
import zhMessages from './zh.json';

i18n
  .use(initReactI18next)
  .init({
    resources: {
      es: {
        translation: esMessages,
      },
      zh: {
        translation: zhMessages,
      },
    },
    interpolation: {
      escapeValue: false, // react already safe from xss
    },
    keySeparator: false,
    nsSeparator: false,
    returnEmptyString: false,
    saveMissing: true,
    lng: 'en',
  });

export default i18n;
