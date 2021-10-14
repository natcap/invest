#!/bin/sh

for locale_dir in src/natcap/invest/translations/locales/*; do
    po2json $locale_dir/LC_MESSAGES/messages.po $locale_dir/LC_MESSAGES/messages.json
done
