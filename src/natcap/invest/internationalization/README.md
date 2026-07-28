# Internationalization

## Summary of files
None of the translation files (`.pot`, `.po`, `.mo`) should be manually edited by us.

### `messages.pot`
Message catalog template file. This contains all the strings ("messages") that are translated, without any translations. All the `.po` files are derived from this.

### `babel_config.ini`
Mappings file that tells pybabel where to look when extracting messages into the message catalog. By default, pybabel will extract messages from Python source files; we need this mappings file to ensure it also extracts messages from the Jinja templates that are used for HTML reports.

### `locales/`
Locale directory. The contents of this directory are organized in a specific structure that `gettext` expects. `locales/` contains one subdirectory for each language for which there are any translations (not including the default English). The subdirectories are named after the corresponding ISO 639-1 language code. Each language subdirectory contains a directory `LC_MESSAGES`, which then contains the message catalog files for that language.

### `locales/<lang>/LC_MESSAGES/messages.po`
Human-readable message catalog file. Messages are added to this file from the PO template (`.pot` file), and translations for the messages are added by the translator.

Messages in `.po` files may be annotated with flags (comma-separated strings immediately preceding a `msgid`). Some flags are defined by the translation tooling and built into its automated processes; others may be custom, with special meaning and processes defined by a particular project or dev team.

Common flags in the `invest` codebase include the following:
- `fuzzy` indicates a translation needs verification. This label can be applied in two scenarios: (1) the automated tools have attempted to translate the message based on similar messages that have already been translated, or (2) a translator has some doubt about the translation and has intentionally flagged it for review (e.g., using Poedit's "Needs work" button).
- `python-brace-format` indicates there are variables inside curly braces, for example: `{model_name}`.
- `python-format` indicates there are variables in this format: `%(model_name)s`. (Note the `s` is part of the formatting, NOT the English letter `s`.)
- `no-python-format` is a custom flag we use to identify messages with `%` characters that have been incorrectly flagged with `python-format`.

### `locales/<lang>/LC_MESSAGES/messages.mo`
Machine-readable message catalog file. This is compiled from the corresponding `.po` file. `gettext` accesses this to look up string translations at runtime. These are created as part of the install process (currently in the `setup.py` script). They are not tracked in the git repo because they duplicate the information that's in the `.po` files, and creating them is not computationally expensive.

## Process to update translations
No changes are immediately needed when we add, remove, or edit strings that are translated. We only need to update the translations files when we are going to send them to the translator. Ideally this would happen for each language before each release, but that may not be possible, and that's okay. This process can happen at any time, whenever a translator is available to us. Any string for which a translation is unavailable will automatically fall back to the English version.

When we are ready to get a new batch of translations, here is the process. These instructions assume you have defined the two-letter locale code in an environment variable `$LL`.

### Evaluating the need for updated translations
There are times when we might want to know how many messages in the current catalog have existing translations. This information can help us decide when it's time to request new/updated translations, and it can help us define the scope of work when preparing a translation contract.

1. First, update the message catalog template (`.pot`) file. Run the following from the root `invest` directory:
    ```
    pybabel extract \
      --no-wrap \
      --project InVEST \
      --version $(python -m setuptools_scm) \
      --msgid-bugs-address natcap-software@lists.stanford.edu \
      --copyright-holder "Natural Capital Alliance" \
      --mapping src/natcap/invest/internationalization/babel_config.ini \
      --output src/natcap/invest/internationalization/messages.pot \
      src/
    ```

2. Next, update the message catalog (`.po`) file. Run the following from the root `invest` directory, replacing `$LL` with the language code:
    ```
    # update message catalog from template
    pybabel update \
      --locale $LL \
      --input-file src/natcap/invest/internationalization/messages.pot \
      --output-file src/natcap/invest/internationalization/locales/$LL/LC_MESSAGES/messages.po
      --no-fuzzy-matching
    ```

    The `--no-fuzzy-matching` flag tells pybabel to skip the step where it attempts to translate new/updated messages based on translations that already exist in the catalog. When we prepare `.po` files for translators, we _don't_ skip fuzzy matching. While the quality of "fuzzy" translations varies (from flawless to nonsensical), we generally consider them to be helpful. For example, when a message contains a word or phrase that could be translated in multiple ways but should be kept consistent within InVEST, a fuzzy translation can save the translator some effort by matching that word or phrase to other instances in already-translated messages in the message catalog.

    When gauging translation need or estimating scope of work, however, it's important to remember that all fuzzy translations will require translator review. This is why we use the `--no-fuzzy-matching` flag here.

3. Finally, run the following from the root `invest` directory, replacing `$LL` with the language code:
    ```
    pybabel compile \
      --input-file src/natcap/invest/internationalization/locales/$LL/LC_MESSAGES/messages.po \
      --output-file src/natcap/invest/internationalization/locales/$LL/LC_MESSAGES/messages.mo \
      --statistics
    ```

    This will generate the machine-readable `.mo` file, and—thanks to the `--statistics` flag—it will also output the number (and percent) of messages that have already been translated, for example:
    ```
    727 of 1743 messages (41%) translated
    ```

4. Make a note of the statistics reported in step 3 so you can share them and/or use them to inform decisions.

5. Do not commit any changes. Revert all changes to `.mo` and `.po` files.

    _If you are ready to prepare files for translators_, jump to [step 2 in "Before requesting translation"](#before-requesting-translation).

    Otherwise, revert all changes to the `.pot` file.

### Before requesting translation
1. First, update the message catalog template file. Run the following from the root `invest` directory:
    ```
    pybabel extract \
      --no-wrap \
      --project InVEST \
      --version $(python -m setuptools_scm) \
      --msgid-bugs-address natcap-software@lists.stanford.edu \
      --copyright-holder "Natural Capital Alliance" \
      --mapping src/natcap/invest/internationalization/babel_config.ini \
      --output src/natcap/invest/internationalization/messages.pot \
      src/
    ```

    This looks through the source code for strings wrapped in the `gettext(...)` function and writes them to the message catalog template.

2. Open the updated `.pot` file and search for `python-format`. For each match, verify whether `python-format` has been correctly applied.

    If it is correct, leave it as-is.

    For example, here is a correctly applied `python-format` flag:
    ```
    #, python-format
    msgid "InVEST Results: %(model_name)s"
    ```

    If it is incorrect, change `python-format` to `no-python-format`.

    For example, here is an _incorrectly_ applied `python-format` flag:
    ```
    #, python-format
    msgid "Average of the highest 10% of wind speeds that blow in the direction of each sector."
    ```

3. Next, run the following from the root `invest` directory, replacing `$LL` with the language code:
    ```
    # update message catalog from template
    pybabel update \
      --locale $LL \
      --input-file src/natcap/invest/internationalization/messages.pot \
      --output-file src/natcap/invest/internationalization/locales/$LL/LC_MESSAGES/messages.po
    ```
    This updates the message catalog for the specified language. New strings that don't yet have a translation will have an empty `msgstr` value. Previously translated messages that are no longer needed will be commented out but remain in the file. This will save translator time if they're needed again in the future.

    If you need to update message catalogs for multiple languages, repeat this step as needed.

4. Open each `.po` file and search for `no-python-format`. Each message flagged with `no-python-format` will also have `python-format` (automatically—and annoyingly—applied by the `pybabel update` command), but we know this is incorrect since we have already identified this message with `no-python-format`. Remove the `python-format` flag. This will prevent these messages from being flagged with errors in GUI editors, such as Poedit.

    When you're done, there should be zero messages with both the `no-python-format` flag and the `python-format` flag.

    For example, a message containing a literal `%` character should have `no-python-format` but not `python-format`:
    ```
    #, no-python-format
    msgid ""
    "Proportion of the highest 10% of wave power values on record that are in "
    "each sector."
    ```

5. Double-check the changes to the `.pot` and `.po` files, then commit them. Use a descriptive commit message, for example: "Extract messages and use resulting updated (.pot) template to update es, zh (.po) catalogs."

### Request translation
1. Send `src/natcap/invest/internationalization/locales/$LL/LC_MESSAGES/messages.po` to the translator. The translator will fill in the `msgstr` values for any new or edited messages.

    **Note:** the process for requesting translation may vary by language and/or over time and may require other steps before we can send files to the translator. When in doubt, check with the NatCap Software Team Lead for the current requirements.

### After receiving translations
1. Check to see if there are any unresolved questions or comments from the translator. It's convenient to use a GUI (such as [Poedit](https://poedit.com/)) for this, since you can easily sort messages so that all the ones marked "Needs work" (flagged with "fuzzy" in the source file) appear first.

    If there are just a few translations in need of review—and you happen to know the target language—it may be most expedient to review them on your own. Sometimes the translator will have described a particular doubt by leaving a comment on a message. Other messages may not have comments. In any case, if you are able to identify the issue and feel confident resolving it, either make revisions (if needed) or accept the translation as-is by manually removing the "Needs work" flag. If there is a comment (and it is no longer needed), remove it.

    In general, it's best to get in touch with the translator so you can help them finalize the translations by answering any lingering questions. Once the translations are complete, make sure all comments have been removed and all "Needs work" flags have been removed.

2. Replace `src/natcap/invest/internationalization/locales/$LL/LC_MESSAGES/messages.po` with the updated version received from the translator (including additional updates from step 1, if any) and commit. Use a descriptive commit message, for example: "Update es message catalog with new translations."

3. Check for formatting problems, and correct any you find. These may be flagged as errors in Poedit, or you might find them by spot-checking messages containing HTML tags or flagged with `python-format` or `python-brace-format` in the raw `.po` text file, or they may be more subtle and uncovered only during manual testing. Do the best you can at this stage; we can always make more updates later if needed.

4. If you made any changes in step 3, commit them. Use a descriptive commit message, for example: "Update es message catalog with corrected formatting."

## Process to add support for a new language
```
mkdir -p src/natcap/invest/internationalization/locales/$LL/LC_MESSAGES/  # create the expected directory structure
pybabel init --input-file src/natcap/invest/internationalization/messages.pot --output-file src/natcap/invest/internationalization/locales/$LL/LC_MESSAGES/messages.po --locale $LL # initialize the message catalog from the template
```
Then jump to step 4 in the ["Process to update translations: Before requesting translation"](#before-requesting-translation) instructions.

## Which messages are translated?

* Model titles
* `MODEL_SPEC` `name` and `about` text
* Validation messages
* Strings that appear in HTML reports, such as section headings, figure captions, and table column headers

Strings that appear exclusively in the Workbench UI, such as button labels and tooltip text, are also translated, but they are handled separately. See the [Workbench README](../../../../workbench/readme.md#internationalization) for details.

We are not translating:

* "InVEST"
* Log messages - most are not helpful to the user anyway, there are a lot of them, and receiving log files in other languages would make it difficult for us to debug issues.
