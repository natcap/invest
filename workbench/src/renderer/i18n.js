import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import es_messages from './i18n/es.json';
import zh_messages from './i18n/zh.json';

i18n
  .use(initReactI18next) // passes i18n down to react-i18next
  .init({
    resources: {
      es: {
        translation: es_messages
      },
      zh: {
        translation: zh_messages
      },
    },
    lng: 'en',
    interpolation: {
      escapeValue: false // react already safes from xss
    },
    saveMissing: true,
  });

export default i18n;
