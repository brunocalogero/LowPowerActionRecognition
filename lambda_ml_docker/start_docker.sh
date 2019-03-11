#!/usr/bin/env bash

# Copyright: (c) 2019, Bruno Calogero <brunocalogero@hotmail.com>
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

set -euo pipefail

if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    echo "AWS_ACCESS_KEY_ID not set in environment."
    echo "Getting AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from profile \"${AWS_PROFILE:=default}\"..."
    export AWS_ACCESS_KEY_ID=$(aws --profile ${AWS_PROFILE} configure get aws_access_key_id)
    export AWS_SECRET_ACCESS_KEY=$(aws --profile ${AWS_PROFILE} configure get aws_secret_access_key)
fi

docker run --rm -ti \
    -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} lambda_ml /bin/bash
