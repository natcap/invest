#!/bin/sh
for locale_dir in src/static/internationalization/locales/*; do
    yarn po2json --pretty $locale_dir/LC_MESSAGES/messages.po $locale_dir/LC_MESSAGES/messages.json
done
