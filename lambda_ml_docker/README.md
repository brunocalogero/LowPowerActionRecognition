## WRITTEN BY: BRUNO CALOGERO
## WHOLE CODE BY: BRUNO CALOGERO

## First time usage

- chmod u+x ./deploy_model.sh
- chmod u+x ./start_docker.sh

- make setup
- make deploy-model (only do this once - sends model over to s3)
- make remove
- make deploy


## After first usage (if dependancies and deployment code changed):

- make clean
- make setup
- make remove
- make deploy


## local Docker build:

- make clean
- make setup
- docker build -t lambda_ml .


## Debug and test usage:

- `sls logs -f app -t` (for debugging)
- `serverless invoke local -f {function_name} --data {}` (CLI usage)
- `curl -X POST https://aqu0f7gi64.execute-api.eu-west-2.amazonaws.com/dev/spamorham -w "\n" -d "Am I spam or am I ham?"` (testing REST CALL with vanilla example)

## Actual Testing
- CF `/utils/test_post.py` for more info
- essentially, once all is setup correctly, copy the link after the make deploy in `test_post.py` and run
