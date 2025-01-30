#!/bin/bash

# TODO: write a cron job to verify that the service is still running (heartbeat)
# TODO: incorporate slack updates to let us know when something was signed, or
#       if the service crashed

while true
do
    DATA=$(curl -i -H "Accept: application/json" "https://us-west1-natcap-servers.cloudfunctions.net/codesigning-queue{\"token\": \"$ACCESS_TOKEN\"}")
    # The response body will be empty when there is nothing in the queue.
    if [ -z "$DATA" ]; then
        sleep 30
        continue
    fi
	BASENAME=$(jq ".basename" <<< $DATA)
	wget -O $BASENAME $(jq ".https-url" <<< $DATA)
    python3 opt/natcap-codesign/natcap_codesign.py /opt/natcap-codesign/codesign-cert-chain.pem "$BASENAME"
	gcloud storage upload $BASENAME $(jq ".gs-uri" <<< $DATA)
	sleep 30
done
