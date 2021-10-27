import i18n from 'gettext.js';


// load each message catalog JSON file into an object
// for easy access when we switch languages
const messageCatalogs = {};
for (const languageCode of languages) {
  const messageCatalogPath = `../static/internationalization/locales/${languageCode}/LC_MESSAGES/messages.json`;
  const fileContents = await fetch(messageCatalogPath).then(response => response.text());
  messageCatalogs[languageCode] = JSON.parse(fileContents);
}


export function setLanguage(languageCode) {
    i18n.loadJSON(messageCatalogs[languageCode], 'messages');
    window._ = i18n.gettext();
}
