FROM nikolaik/python-nodejs:latest

WORKDIR /t1d

WORKDIR /autotune

# get openAPS Project from github
RUN git clone https://github.com/openaps/oref0.git

WORKDIR /autotune/oref0

# install openAPS modules
RUN npm install
RUN npm install -git
RUN npm link
RUN npm link oref0
RUN npm install -g nodemon

RUN pip install pandas matplotlib coloredlogs imageio scipy sklearn statsmodels

ENV T1DPATH=/t1d/

WORKDIR /t1d

#CMD nodemon --exec python main.py -L
CMD python python/main.py