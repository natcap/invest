# Internationalization

This directory contains

## Summary of files
None of the translations files (.pot, .po, .mo) should be manually edited by us.

### `messages.pot`
Message catalog template file. This contains all the strings ("messages") that are translated, without any translations. All the PO files are derived from this.

### `locales/`
Locale directory. The contents of this directory are organized in a specific structure that `gettext` expects. `locales/` contains one subdirectory for each language for which there are any translations (not including the default English). The subdirectories are named after the corresponding language code. Each language subdirectory contains a directory `LC_MESSAGES`, which then contains the message catalog files for that language.

### `locales/<lang>/LC_MESSAGES/messages.po`
Human-readable message catalog file. Messages are added to this file from the PO template, and translations for the messages are added by the translator.

### `locales/<lang>/LC_MESSAGES/messages.mo`
Machine-readable message catalog file. This is compiled from the corresponding PO file. `gettext` accesses this to look up string translations at runtime. These are *not* checked in, but they are created as part of the `setup.py install` process. They are not checked in because they duplicate the information that's in the PO files, and creating them is not computationally expensive.


## Process to update translations

No changes are immediately needed when we add, remove, or edit strings that are translated. We only need to update the translations files when we are going to send them to the translator. Ideally this would happen for each language before each release, but that may not be possible, and that's okay. Any string for which a translation is unavailable will automatically fall back to the English version.

When we are ready to get a new batch of translations, here is the process. :

1. Update the PO template file

2. Update the PO files based on the PO template file

3. Send the PO files to the translator and wait to get them back.

   The translator will fill in the PO files with translations for any new or edited messages.

4. Check in the new files

   The new `messages.pot`, and `messages.po` for each language, should be checked in.

## Which messages are translated?


* `ARGS_SPEC` `model_name`, and `name` and `about` text for each arg
* Validation messages
* Strings that appear in the UI, excluding log messages

We are not translating log messages at this time because most are not helpful to the user, there are a lot of them, and receiving log files in other languages would make it difficult for us to debug issues.
