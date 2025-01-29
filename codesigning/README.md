# InVEST Codesigning Service


## Future Work

### Subscribe to GCS events

GCP Cloud Functions have the ability to subscribe to bucket events, which
should allow us to subscribe very specifically to just those `finalize` events
that apply to the Windows workbench binaries.  Doing so will require reworking this cloud function into 2 cloud functions:

1. An endpoint for ncp-inkwell to poll for the next binary to sign
2. A cloud function that subscribes to GCS bucket events and enqueues the binary to sign.

Relevant docs include:
* https://cloud.google.com/functions/docs/writing/write-event-driven-functions#cloudevent-example-python

