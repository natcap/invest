import GettextJS from 'gettext.js';
import gettextParser from 'gettext-parser';
import { ipcMain } from 'electron';
import fs from 'fs';
import { getLogger } from './logger';
import { ipcMainChannels } from './ipcMainChannels';
import ELECTRON_DEV_MODE from './isDevMode';

const logger = getLogger(__filename.split('/').slice(-1)[0]);
const i18n = new GettextJS();

/** Read PO file into gettext.js formatted message catalog object. */
function readMessageCatalog(messageCatalogPath) {
  const input = fs.readFileSync(messageCatalogPath);
  // parse PO file into an object in gettext-parser format
  // https://github.com/smhg/gettext-parser#data-structure-of-parsed-mopo-files
  const rawPO = gettextParser.po.parse(input);

  // convert from gettext-parser format to gettext.js format
  // https://github.com/smhg/gettext-parser#data-structure-of-parsed-mopo-files
  // https://github.com/guillaumepotier/gettext.js#required-json-format
  const formattedPO = {
    '': {
      language: rawPO.headers.Language,
      'plural-forms': rawPO.headers['Plural-Forms'],
    },
  };
  // leave out the empty message which contains the header string
  delete rawPO.translations[''][''];
  Object.keys(rawPO.translations['']).forEach((msgid) => {
    if (rawPO.translations[''][msgid].msgstr.length === 1) {
      [formattedPO[msgid]] = rawPO.translations[''][msgid].msgstr;
    } else {
      formattedPO[msgid] = rawPO.translations[''][msgid].msgstr;
    }
  });
  return formattedPO;
}

/** Load message catalogs for each language so they're available to i18n. */
function loadMessageCatalogs() {
  // load each language's message catalog PO file into an object
  // for easy access when we switch languages
  const languages = fs.readdirSync(`${__dirname}/../static/internationalization/locales`);
  if (languages) {
    languages.forEach((language) => {
      const messageCatalogPath = `${__dirname}/../static/internationalization/locales/${language}/LC_MESSAGES/messages.po`;
      i18n.loadJSON(readMessageCatalog(messageCatalogPath), 'messages');
    });
  }
  logger.debug('loaded message catalogs');
}

export default function setupSetLanguage() {
  loadMessageCatalogs();

  ipcMain.handle(
    ipcMainChannels.SET_LANGUAGE,
    (e, languageCode) => {
      logger.debug('setting language to', languageCode);
      i18n.setLocale(languageCode);
    }
  );

  ipcMain.on(
    ipcMainChannels.GETTEXT,
    (event, message) => {
      event.returnValue = i18n.gettext(message);
    }
  );

  ipcMain.handle(
    ipcMainChannels.IS_DEV_MODE,
    () => ELECTRON_DEV_MODE
  );
}
