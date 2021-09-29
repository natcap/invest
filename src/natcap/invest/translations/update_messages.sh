#!/bin/sh

pybabel extract --output src/natcap/invest/translations/messages.pot src/
for locale_dir in src/natcap/invest/translations/locales/*; do
    echo $locale
    echo $(basename $locale_dir)
    pybabel update \
        --locale $(basename $locale_dir) \
        --input-file src/natcap/invest/translations/messages.pot \
        --output-file $locale_dir/LC_MESSAGES/messages.po
done
