FROM node:11
WORKDIR /autotune

# get openAPS Project from github
RUN git clone https://github.com/openaps/oref0.git

WORKDIR /autotune/oref0

# install openAPS modules
RUN npm install
RUN npm install -git
RUN npm link
RUN npm link oref0

# Run Autotune from openAPS, needs input files in /autotune/data/input
CMD bash /autotune/data/prep_and_run_autotune.sh