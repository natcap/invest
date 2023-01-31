# Internationalization

See the internationalization readme in the invest repo.

We are using the javascript internationalization package `i18next` and its react extension `react-i18next`. `i18next` takes in translation resources as a javascript object. It's convenient to store translations in JSON format. Vite automatically serves JSON files as Javascript modules, so we can directly import translations from JSON.

The translations for each language live in `workbench/src/renderer/i18n/xx.json`. The JSON object in each file maps English messages to translations.

Nothing needs to be done during routine development. As we make changes to the workbench text, it will inevitably get out of sync with the translations, and that's okay. Strings that have no translation will fall back to English. When it's time to update our translations, this is the process:

## Getting a new batch of translations

1. Extract messages from the source code:
   `i18next "src/**/*.{js,jsx}" --output out.json`
   This command is provided by the `i18next-parser` package and configured by `workbench/i18next-parser.config.mjs`. `out.json` should contain a JSON object mapping each translated message from the source code to an empty string.

2. Merge into the existing translation file:
   `jq -s add src/renderer/i18n/<LANG>.json out.json > src/renderer/i18n/<LANG>.json`
   This will add new keys from `out.json` into `src/renderer/i18n/<LANG>.json` and leave those that already have translations:
   ```
   {
      "text that's already been translated": "translation",
      "new text that doesn't have a translation yet": ""
   }
   ```
4. Commit the changes:
   ```
   git add src/renderer/i18n/<LANG>.json
   git commit -m "add new messages into <LANG> translation file"
   ```
3. (if the translator uses PO format) Convert JSON to PO

4. Send `src/renderer/i18n/<LANG>.[json,po]` to the translator and wait to receive a copy with translations added.

5. (if the translator uses PO format) Convert PO to JSON
   If the translator works with PO files, we can convert them to JSON using this tool: https://github.com/i18next/i18next-gettext-converter

6. Replace `src/renderer/i18n/<LANG>.json` with the new copy received from the translator

7. Commit the changes:
   ```
   git add src/renderer/i18n/<LANG>.json
   git commit -m "add translations for <LANG>"
   ```
