# How to run the app
## Use a Docker container to run the python and nodejs code inside.

## Build Docker image
`docker build -t t1d-pred:latest .`

## Run Docker image
`docker run --rm -v ./Python:/t1d -v ../data:/t1d/data -v .\Autotune/data/Input:/t1d/input -v ./Autotune/data:/autotune/data t1d-pred:latest`

## Run Docker image with autowatch to run code on changes
`docker run --rm -v ./Python:/t1d -v ../data:/t1d/data -v ./Autotune/data/Input:/t1d/input -v ./Autotune/data:/autotune/data t1d-pred:latest nodemon --exec python main.py -L`

### To stop running containers
`docker stop $(docker ps -a -q --filter ancestor=t1d-pred:latest --format="{{.ID}}")`

