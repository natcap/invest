ENV := env
CWD := $(shell pwd)
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
deploy-worker: env
	cd signing-worker && $(CWD)/$(ENV)/bin/ansible-playbook \
		--ask-become-pass \
		--inventory inventory.ini \
		playbook.yml

env:
	python -m venv $(ENV)
	$(CWD)/$(ENV)/bin/python -m pip install -r requirements.txt
