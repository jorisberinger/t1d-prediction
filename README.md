# How to run the app
## Use a Docker container to run the python and nodejs code inside.

## Build Docker image
`docker build -t t1d-pred:latest .`

## Run Docker image
`docker run --rm -v C:\Users\joris\apps\master\t1d-prediction:/t1d -v C:\Users\joris\apps\master\data:/t1d/data t1d-pred:latest`

## Run Docker image with autowatch to run code on changes
`docker run --rm -v C:\Users\joris\apps\master\t1d-prediction:/t1d -v C:\Users\joris\apps\master\data:/t1d/data t1d-pred:latest nodemon --exec python python/main.py -L`

### To stop running containers
`docker stop $(docker ps -a -q --filter ancestor=t1d-pred:latest --format="{{.ID}}")`

