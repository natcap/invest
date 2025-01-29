.PHONY: deploy-cloudfunction deploy-worker

deploy-cloudfunction:
	gcloud functions deploy \
		codesigning-queue \
		--trigger-http \
		--region us-west1 \
		--allow-unauthenticated \
		--entry-point main \
		--runtime python312 \
		--run-service-account codesigning-service@natcap-servers.iam.gserviceaccount.com \
		--source gcp-cloudfunc/


# NOTE: This must be executed from a computer that has SSH access to ncp-inkwell.
deploy-worker:
	cd signing-worker
	ansible-playbook \
		--ask-become-pass \
		--inventory-file inventory.ini \
		playbook.yml
