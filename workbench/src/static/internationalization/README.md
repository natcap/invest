# Internationalization

See the internationalization readme in the invest repo.

The only difference is that here in the workbench, the POT file is edited manually instead of generated automatically.

This is because:
* There are a small number of messages so it was easy to create the POT file manually.
* The standard message extraction tool, `xgettext`, supports JavaScript but did not work perfectly here. It probably got thrown off by the JSX syntax.
* Changes to message strings will likely be small and infrequent.

If it becomes burdensome to keep up with manually editing the POT file, we can look for a message extraction tool that supports JSX syntax or write our own script for it. For now, we'll need to manually update the POT file whenever we change a message string. It makes sense to do this at the same time, rather than all in one batch before translating, so that we don't forget any.

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

See the InVEST internationalization README. The only difference is to use `msgmerge` instead of `pybabel` to update the PO file:
```
$ msgmerge internationalization/locales/<LANG>/LC_MESSAGES/messages.po internationalization/messages.pot
```

## Adding support for a new language
```
$ mkdir -p internationalization/locales/<LANG>/LC_MESSAGES

$ msginit --no-translator --input internationalization/messages.pot --output-file internationalization/locales/<LANG>/LC_MESSAGES/messages.po
```

