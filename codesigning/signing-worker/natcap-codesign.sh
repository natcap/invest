#!/bin/bash

# TODO: write a cron job to verify that the service is still running (heartbeat)
# TODO: incorporate slack updates to let us know when something was signed, or
#       if the service crashed

while true
do
	$DATA=$(get from service)  # This is a json object with the https url, gs url, etc.
	$BASENAME=$(jq ".basename" $DATA)
	wget -O $BASENAME $(jq ".https-url" $DATA)
    python3 natcap_codesign.py /opt/natcap-codesign/codesign-cert-chain.pem "$BASENAME"
	gcloud storage upload $BASENAME $(jq ".gs-uri"  $DATA)
	sleep 30
done
