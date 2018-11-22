FROM node:11
WORKDIR /autotune


RUN git clone https://github.com/openaps/oref0.git

WORKDIR /autotune/oref0

RUN npm install
RUN npm install -git
RUN npm link
RUN npm link oref0

VOLUME ./t1d-prediction/Autotune/data ./data