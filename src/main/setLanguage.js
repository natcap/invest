import gettext_js from 'gettext.js';
import { ipcMain } from 'electron';
import { ipcMainChannels } from './ipcMainChannels';
import fetch from 'node-fetch';
import fs from 'fs';


const i18n = new gettext_js();

var gettextParser = require("gettext-parser");

/** Read PO file into gettext.js formatted message catalog object. */
function readMessageCatalog(messageCatalogPath) {
  const input = fs.readFileSync(messageCatalogPath);
  // parse PO file into an object in gettext-parser format
  // https://github.com/smhg/gettext-parser#data-structure-of-parsed-mopo-files
  const raw_po = gettextParser.po.parse(input)

  // convert from gettext-parser format to gettext.js format
  // https://github.com/guillaumepotier/gettext.js#required-json-format
  const formatted_po = {};
  formatted_po[''] = {};
  formatted_po['']['language'] = raw_po['headers']['language']
  formatted_po['']['plural-forms'] = raw_po['headers']['plural-forms']
  delete raw_po['translations']['']['']
  for (const msgid in raw_po['translations']['']) {
    if (raw_po['translations'][''][msgid]['msgstr'].length === 1) {
      formatted_po[msgid] = raw_po['translations'][''][msgid]['msgstr'][0]
    } else {
      formatted_po[msgid] = raw_po['translations'][''][msgid]['msgstr']
    }
  };
}

/** Load message catalogs for each language so they're available to i18n. */
async function loadMessageCatalogs() {
  // load each message catalog JSON file into an object
  // for easy access when we switch languages
  fs.readdir(
    `${__dirname}/../static/internationalization/locales`,
    async function (err, languages) {
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

export function setupSetLanguage() {
  ipcMain.handle(
    ipcMainChannels.SET_LANGUAGE,
    (event, languageCode) => {
      i18n.setLocale(languageCode);
    }
  );

  ipcMain.on(
    ipcMainChannels.GETTEXT,
    (event, message) => {
      event.returnValue = i18n.gettext(message);
    }
  );
}
