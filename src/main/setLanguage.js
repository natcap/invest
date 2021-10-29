import gettext_js from 'gettext.js';
import { ipcMain } from 'electron';
import { ipcMainChannels } from './ipcMainChannels';
import fetch from 'node-fetch';
import fs from 'fs';


const i18n = new gettext_js();

async function loadMessageCatalogs() {
  // load each message catalog JSON file into an object
  // for easy access when we switch languages
  const languages = ['es'];
  for (const languageCode of languages) {
    const messageCatalogPath = `../../internationalization/locales/${languageCode}/LC_MESSAGES/messages.json`;
    const data = await import(messageCatalogPath).then(data => data['default']);
    i18n.loadJSON(data, 'messages');
  }
}
loadMessageCatalogs();



export function setupSetLanguage() {
  ipcMain.handle(
    ipcMainChannels.SET_LANGUAGE,
    (event, languageCode) => {
      console.log('handle setting language', languageCode);
      i18n.setLocale(languageCode);
    }
  );

  ipcMain.on(
    ipcMainChannels.GETTEXT,
    (event, message) => {
      console.log('handling', message);
      console.log(i18n.gettext(message));
      event.returnValue = i18n.gettext(message);
    }
  );
}
