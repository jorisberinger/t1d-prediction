
docker build --rm -f "t1d-prediction\Autotune\Dockerfile" -t autotune:latest t1d-prediction\Autotune && docker run --rm -v C:\Users\joris\apps\master\t1d-prediction\Autotune\data\result:/autotune/data/result autotune:latest 

