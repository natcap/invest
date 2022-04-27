# Internationalization

## Summary of files
None of the translations files (.pot, .po, .mo) should be manually edited by us.

### `messages.pot`
Message catalog template file. This contains all the strings ("messages") that are translated, without any translations. All the PO files are derived from this.

### `locales/`
Locale directory. The contents of this directory are organized in a specific structure that `gettext` expects. `locales/` contains one subdirectory for each language for which there are any translations (not including the default English). The subdirectories are named after the corresponding ISO 639-1 language code. Each language subdirectory contains a directory `LC_MESSAGES`, which then contains the message catalog files for that language.

### `locales/<lang>/LC_MESSAGES/messages.po`
Human-readable message catalog file. Messages are added to this file from the PO template, and translations for the messages are added by the translator.

### `locales/<lang>/LC_MESSAGES/messages.mo`
Machine-readable message catalog file. This is compiled from the corresponding PO file. `gettext` accesses this to look up string translations at runtime. These are not checked in, but created as part of the install process (currently in the `setup.py` script). They are not checked in because they duplicate the information that's in the PO files, and creating them is not computationally expensive.

## Process to update translations

No changes are immediately needed when we add, remove, or edit strings that are translated. We only need to update the translations files when we are going to send them to the translator. Ideally this would happen for each language before each release, but that may not be possible, and that's okay. This process can happen at any time, whenever a translator is available to us. Any string for which a translation is unavailable will automatically fall back to the English version.

When we are ready to get a new batch of translations, here is the process.

1. Run the following from the root invest directory, replacing `<LANG>` with the language code:
```
pybabel extract \
   --no-wrap \
   --project InVEST \
   --version 3.10 \
   --msgid-bugs-address esoth@stanford.edu \
   --copyright-holder "Natural Capital Project" \
   --output src/natcap/invest/internationalization/messages.pot \
   src/

pybabel update \  # update message catalog from template
   --locale <LANG> \
   --input-file src/natcap/invest/internationalization/messages.pot \
   --output-file src/natcap/invest/internationalization/locales/<LANG>/LC_MESSAGES/messages.po
```

2. Check that the changes look correct, then commit:
```
git diff
git add src/natcap/invest/internationalization/messages.pot src/natcap/invest/internationalization/locales/<LANG>/LC_MESSAGES/messages.po
git commit -m "extract message catalog template and update <LANG> catalog from it"
```
This looks through the source code for strings wrapped in the `gettext(...)` function and writes them to the message catalog template. Then it updates the message catalog for the specificed language. New strings that don't yet have a translation will have an empty `msgstr` value. Previously translated messages that are no longer needed will be commented out but remain in the file. This will save translator time if they're needed again in the future.

3. Send `src/natcap/invest/internationalization/locales/<LANG>/LC_MESSAGES/messages.po` to the translator and wait to get it back. The translator will fill in the `msgstr` values for any new or edited messages.

4. Replace `src/natcap/invest/internationalization/locales/<LANG>/LC_MESSAGES/messages.po` with the updated version received from the translator and commit.
```
git add internationalization/locales/<LANG>/LC_MESSAGES/messages.po
git commit -m "update <LANG> message catalog with new translations"
```

## Process to add support for a new language

```
mkdir -p src/natcap/invest/internationalization/locales/<LANG>/LC_MESSAGES/  # create the expected directory structure
pybabel init --input-file src/natcap/invest/internationalization/messages.pot --output-file src/natcap/invest/internationalization/locales/<LANG>/LC_MESSAGES/messages.po --locale <LANG> # initialize the message catalog from the template
```
Then follow the "Process to update translations" instructions above, starting from step 2.

## Which messages are translated?

* Model titles
* `ARGS_SPEC` `name` and `about` text
* Validation messages
* Strings that appear in the UI, such as button labels and tooltip text

We are not translating:

* "InVEST"
* Log messages - most are not helpful to the user anyway, there are a lot of them, and receiving log files in other languages would make it difficult for us to debug issues.
