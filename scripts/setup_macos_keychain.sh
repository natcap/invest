#!/bin/bash -ve

#
# Arguments:
#  $1 = the name to give the new keychain
#  $2 = the keychain password to set
#  $3 = path to .p12 certificate file to add to keychain
#  $4 = .p12 certificate file password


security list-keychains
security create-keychain -p $2 $1
security list-keychains -s $1
security list-keychains

echo $1
echo $3

echo 'listed keychains'
# unlock the keychain so we can import to it (stays unlocked 5 minutes by default)
security unlock-keychain -p $2 $1
echo 'unlocked keychain'

# add the certificate to the keychain
# -T option says that the codesign executable can access the keychain
# for some reason this alone is not enough, also need the following step
security import $3 -k $1 -P $4 -T /usr/bin/codesign

echo 'imported cert'
# this is essential to avoid the UI password prompt
security set-key-partition-list -S apple-tool:,apple: -s -k $2 $1


