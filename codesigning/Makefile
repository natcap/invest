.PHONY: deploy-cloudfunction deploy-worker

deploy-cloudfunction:
	gcloud functions deploy \
		--project natcap-servers \
		codesigning-queue \
		--memory=256Mi \
		--trigger-http \
		--gen2 \
		--region us-west1 \
		--allow-unauthenticated \
		--entry-point main \
		--runtime python312 \
		--source gcp-cloudfunc/

# NOTE: This must be executed from a computer that has SSH access to ncp-inkwell.
deploy-worker:
	cd signing-worker && ansible-playbook \
		--ask-become-pass \
		--inventory-file inventory.ini \
		playbook.yml
