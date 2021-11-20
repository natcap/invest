import gettext_js from 'gettext.js';
import { ipcMain } from 'electron';
import { ipcMainChannels } from './ipcMainChannels';
import fetch from 'node-fetch';
import fs from 'fs';


const i18n = new gettext_js();

async function loadMessageCatalogs() {
  // load each message catalog JSON file into an object
  // for easy access when we switch languages
  fs.readdir(
    `${__dirname}/../static/internationalization/locales`,
    async function (err, languages) {
      if (languages) {
        for (const language of languages) {
          const messageCatalogPath = `${__dirname}/../static/internationalization/locales/${language}/LC_MESSAGES/messages.json`;
          const data = await import(messageCatalogPath).then(data => data['default']);
          i18n.loadJSON(data, 'messages');
        }
      }
    }
  )
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
