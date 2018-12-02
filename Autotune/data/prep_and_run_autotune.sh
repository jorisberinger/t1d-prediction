#!/usr/bin/env bash

node /autotune/oref0/bin/oref0-autotune-prep.js /autotune/data/input/ns-treatments.json /autotune/data/input/profile.json /autotune/data/input/glucose1.json /autotune/data/input/profile.pump.json > /autotune/data/input/prepped_glucose.json

node /autotune/oref0/bin/oref0-autotune-core.js /autotune/data/input/prepped_glucose.json /autotune/data/input/profile.json /autotune/data/input/profile.pump.json > /autotune/data/autotune-result.json

