USERNAME := $(shell whoami 2> /dev/null)
# Use a default value of `nobody` if variable is empty
USERNAME := $(or $(USERNAME),$(USERNAME),nobody) 
JOBYAML_TRAIN=train.yaml
JOBYAML_NOTEBOOK=notebook.yaml
JOBYAML_INFLUENCES=influences.yaml
# By default this deploys to the users default account
# By commenting the line below and uncommenting the line
# after it, you can deploy to the eai.lizard_muffin account instead
TOOLKIT_ACCOUNT_NAME := $(shell eai account get --fields id --no-header)
#TOOLKIT_ACCOUNT_NAME := eai.lizard_muffin

DEMO_IMAGE_VERSION=latest
DEMO_IMAGE_TAG=registry.console.elementai.com/$(TOOLKIT_ACCOUNT_NAME)/fast-if:$(DEMO_IMAGE_VERSION)

## Print training image detail
.PHONY: print-dev-image
print-dev-image:
	@echo "### USER: ${USERNAME}"
	@echo "### ACCOUNT: ${ACCOUNT_ID}"
	@echo "### DEMO_IMAGE_TAG: ${DEMO_IMAGE_TAG}"

.PHONY: clean
clean:
	docker rmi $(DOCKER_IMAGE_TAG) || true

.PHONY: tag
tag: build
	docker tag $(DEMO_IMAGE_TAG) $(DEMO_IMAGE_TAG)

.PHONY: build
build:
	DOCKER_BUILDKIT=1 docker build -t $(DEMO_IMAGE_TAG) .

.PHONY: push
push: tag
	docker push $(DEMO_IMAGE_TAG)

.PHONY: toolkit-train
toolkit-train: push
	eai job submit -i $(DEMO_IMAGE_TAG) -f $(JOBYAML_TRAIN) --account $(TOOLKIT_ACCOUNT_NAME)
	eai job logs -f

.PHONY: toolkit-influences
toolkit-influences: push
	eai job submit -i $(DEMO_IMAGE_TAG) -f $(JOBYAML_INFLUENCES) --account $(TOOLKIT_ACCOUNT_NAME)
	eai job logs -f

.PHONY: toolkit-notebook
toolkit-notebook: push
	eai job submit -i $(DEMO_IMAGE_TAG) -f $(JOBYAML_NOTEBOOK) --account $(TOOLKIT_ACCOUNT_NAME)
	eai job logs -f

.PHONY: toolkit-shuriken
toolkit-shuriken: push
	saga run -v configs/shuriken/experiments.star tracking_data=eai.lizard_muffin.mlflow_data

