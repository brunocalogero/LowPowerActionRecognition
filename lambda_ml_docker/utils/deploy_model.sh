#!/usr/bin/env bash

# Copyright: (c) 2019, Bruno Calogero <brunocalogero@hotmail.com>
# GNU General Public License v3.0+ (see COPYING or https://www.gnu.org/licenses/gpl-3.0.txt)

BASEDIR=$( dirname "${BASH_SOURCE[0]}" )

set -euo pipefail

read -e -p "Enter bucket name: " -i "modelhoster" BUCKET
read -e -p "Enter object prefix: " -i "model" PREFIX

set -x

aws s3 cp --recursive ${BASEDIR}/model s3://${BUCKET}/${PREFIX}/
