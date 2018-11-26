FROM node:11
WORKDIR /autotune


RUN git clone https://github.com/openaps/oref0.git

WORKDIR /autotune/oref0

RUN npm install
RUN npm install -git
RUN npm link
RUN npm link oref0

COPY data/carbhistory.json /autotune/data/carbhistory.json
COPY data/glucose.json /autotune/data/glucose.json
COPY data/profile.json /autotune/data/profile.json
COPY data/pumphistory.json /autotune/data/pumphistory.json
COPY data/prep_and_run_autotune.sh /autotune/data/prep_and_run_autotune.sh



CMD bash /autotune/data/prep_and_run_autotune.sh