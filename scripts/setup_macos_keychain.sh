#!/bin/bash -ve

# The environment variables referenced should be set by the calling github actions step

security list-keychains
security create-keychain -p $KEYCHAIN_PASS $KEYCHAIN_NAME
security list-keychains -s $KEYCHAIN_NAME
security list-keychains


echo 'listed keychains'
# unlock the keychain so we can import to it (stays unlocked 5 minutes by default)
security unlock-keychain -p $KEYCHAIN_PASS $KEYCHAIN_NAME
echo 'unlocked keychain'

# add the certificate to the keychain
# -T option says that the codesign executable can access the keychain
# for some reason this alone is not enough, also need the following step
security import $P12_FILE_PATH -k $KEYCHAIN_NAME -P $CERT_KEY_PASS -T /usr/bin/codesign

echo 'imported cert'
# this is essential to avoid the UI password prompt
security set-key-partition-list -S apple-tool:,apple: -s -k $KEYCHAIN_PASS $KEYCHAIN_NAME


