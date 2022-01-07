import gettext_js from 'gettext.js';
import { ipcMain } from 'electron';
import { ipcMainChannels } from './ipcMainChannels';
import fetch from 'node-fetch';
import fs from 'fs';


const i18n = new gettext_js();

var gettextParser = require("gettext-parser");

async function loadMessageCatalogs() {
  // load each message catalog JSON file into an object
  // for easy access when we switch languages


  fs.readdir(
    `${__dirname}/../static/internationalization/locales`,
    async function (err, languages) {
      if (languages) {
        for (const language of languages) {
          const messageCatalogPath = `${__dirname}/../static/internationalization/locales/${language}/LC_MESSAGES/messages.po`;

          const input = fs.readFileSync(messageCatalogPath);
          const raw_po = gettextParser.po.parse(input)
          delete raw_po['translations']['']['']
          console.log(raw_po)

          // format in the way that i18n expects
          const formatted_po = {};
          formatted_po[''] = {};
          formatted_po['']['language'] = raw_po['headers']['language']
          formatted_po['']['plural-forms'] = raw_po['headers']['plural-forms']

          for (const msgid in raw_po['translations']['']) {
            if (raw_po['translations'][''][msgid]['msgstr'].length === 1) {
              formatted_po[msgid] = raw_po['translations'][''][msgid]['msgstr'][0]
            } else {
              formatted_po[msgid] = raw_po['translations'][''][msgid]['msgstr']
            }
          };

          console.log(formatted_po);
          i18n.loadJSON(formatted_po, 'messages');
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
