#!/usr/bin/env bash

node /autotune/oref0/bin/oref0-autotune-prep.js /autotune/data/pumphistory.json /autotune/data/profile.json /autotune/data/glucose.json > /autotune/data/prepped_glucose.json

node /autotune/oref0/bin/oref0-autotune-core.js /autotune/data/prepped_glucose.json /autotune/data/profile.json /autotune/data/glucose.json > /autotune/data/result.json

cp /autotune/data/prepped_glucose.json /autotune/data/result/prepped_glucose.json

cp /autotune/data/result.json /autotune/data/result/result.json
