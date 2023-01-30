import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import es_messages from './i18n/es.js';
import zh_messages from './i18n/zh.js';

const resources = {
  es: {
    translation: es_messages
  },
  zh: {
    translation: zh_messages
  },
};

i18n
  .use(initReactI18next) // passes i18n down to react-i18next
  .init({
    resources,
    lng: 'en',
    interpolation: {
      escapeValue: false // react already safes from xss
    },
    saveMissing: true,
  });

export default i18n;
