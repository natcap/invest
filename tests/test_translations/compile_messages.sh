#!/bin/sh

for locale_dir in src/natcap/invest/translations/locales/*; do
    pybabel compile \
        --input-file $locale_dir/LC_MESSAGES/messages.po \
        --output-file $locale_dir/LC_MESSAGES/messages.mo
done
