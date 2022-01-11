import GettextJS from 'gettext.js';
import gettextParser from 'gettext-parser';
import { ipcMain } from 'electron';
import { ipcMainChannels } from './ipcMainChannels';
import fetch from 'node-fetch';
import fs from 'fs';

const i18n = new GettextJS();

/** Read PO file into gettext.js formatted message catalog object. */
function readMessageCatalog(messageCatalogPath) {
  const input = fs.readFileSync(messageCatalogPath);
  // parse PO file into an object in gettext-parser format
  // https://github.com/smhg/gettext-parser#data-structure-of-parsed-mopo-files
  const rawPO = gettextParser.po.parse(input);

  // convert from gettext-parser format to gettext.js format
  // https://github.com/guillaumepotier/gettext.js#required-json-format
  const formattedPO = {};
  formattedPO[''] = {};
  formattedPO[''].language = rawPO.headers.language;
  formattedPO['']['plural-forms'] = rawPO.headers['plural-forms'];
  delete rawPO.translations[''][''];
  for (const msgid of Object.keys(rawPO.translations[''])) {
    if (rawPO.translations[''][msgid].msgstr.length === 1) {
      formattedPO[msgid] = rawPO.translations[''][msgid].msgstr[0];
    } else {
      formattedPO[msgid] = rawPO.translations[''][msgid].msgstr;
    }
  };
  return formattedPO;
}

/** Load message catalogs for each language so they're available to i18n. */
async function loadMessageCatalogs() {
  // load each language's message catalog PO file into an object
  // for easy access when we switch languages
  fs.readdir(
    `${__dirname}/../static/internationalization/locales`,
    async (err, languages) => {
      if (languages) {
        for (const language of languages) {
          const messageCatalogPath = `${__dirname}/../static/internationalization/locales/${language}/LC_MESSAGES/messages.po`;
          i18n.loadJSON(readMessageCatalog(messageCatalogPath), 'messages');
        }
      }
    }
  );
  console.log('loaded message catalogs')
}
loadMessageCatalogs();

export default function setupSetLanguage() {
  ipcMain.handle(
    ipcMainChannels.SET_LANGUAGE,
    (e, languageCode) => {
      console.log('set language', languageCode);
      i18n.setLocale(languageCode); }
  );

  ipcMain.on(
    ipcMainChannels.GETTEXT,
    (event, message) => {
      event.returnValue = i18n.gettext(message);
    }
  );
}
