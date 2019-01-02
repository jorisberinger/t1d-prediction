FROM nikolaik/python-nodejs:latest

WORKDIR /t1d

RUN pip install pandas matplotlib

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

WORKDIR /t1d

CMD nodemon --exec python main.py -L
#CMD python main.pyd