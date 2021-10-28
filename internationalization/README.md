# Internationalization

* PO files should never need to be edited manually.

### Adding a new string that should be translated

1. Manually add an entry for the string in `messages.pot`. An entry has the format
   ```
   msgid "<string>"
   msgstr ""
   ```
   where `<string>` is replaced with your string.

### Modifying or removing a translated string that already exists
1. Manually edit or remove the entry for the string in `messages.pot`.

Do *not* remove the entry from the PO files. When we update them with `msgmerge`, entries that are no longer needed will be commented out but remain in the files. This will save translator time if we need them again in the future.

## Getting a new batch of translations
This does *not* need to happen immediately after any change. Ideally we would get translation updates before each release, but that may not be possible, and that's okay. This process can happen at any time, whenever a translator is available to us.

1. Update the appropriate `messages.po` file from the `messages.pot` file using `msgmerge`. From the `invest-workbench` root directory, replacing `<LANG>` with the code for the language being translated to, run
   ```
   msgmerge internationalization/locales/<LANG>/LC_MESSAGES/messages.po internationalization/messages.pot
   ```
2. Look over the changes to the PO and confirm that it looks right.

3. Commit the changes to the PO file.
   ```
   git add internationalization/locales/<LANG>/LC_MESSAGES/messages.po
   git commit -m "update <LANG> message catalog from template"
   ```

4. Send a copy of `internationalization/locales/<LANG>/LC_MESSAGES/messages.po` to the translator for that language. They will fill in the `msgstr` lines with the translation for each `msgid`.

5. Replace the old PO file with the newly translated one.

6. Commit the changes to the PO file.
   ```
   git add internationalization/locales/<LANG>/LC_MESSAGES/messages.po
   git commit -m "update <LANG> message catalog with new translations"
   ```




## Extracting messages
Because of the small number of messages, I created messages.pot manually. This could be done instead with `xgettext`, which supports javascript, though I'm not sure if it works with JSX syntax.

