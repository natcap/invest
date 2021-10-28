#!/bin/sh

tmpfile=tmp_messages

# put all translated strings into a file
grep -E -h -o -r '_\("[^"]*"\)' src/ > $tmpfile  # find translated strings in double quotes
grep -E -h -o -r "_\('[^']*'\)" src/ >> $tmpfile # find translated strings in single quotes
grep -E -h -o -r '_\(`[^`]*`\)' src/ >> $tmpfile # find translated strings in backticks

# sort them alphabetically
string:1:${#string}-2

# format them as a POT template file
while read p; do
    echo 'msgid "$p"' >> messages.pot
    echo 'msgstr ""' >> messages.pot
    echo '' >> messages.pot
done < $tmpfile
