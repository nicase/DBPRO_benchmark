#!/bin/bash

container_name="weaviate-1"
port_mapping="6333:6333"

docker-compose up -d
container_id=$(docker ps -aqf "name=$container_name")
container_ip_addr=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $container_id)
container_port=$(docker port "$container_id" | grep -m 1 "tcp" | awk '{print $NF}' | cut -d ':' -f 2)
cp .env .env.bk
#echo "WEAVIATE_URL=\"$container_ip_addr\"" >> .env
echo "WEAVIATE_PORT=\"$container_port\"" >> .env

python3 attribute_filtering/weaviate_attribute_filtering.py &

python_pid=$!

while ps -p $python_pid > /dev/null; do
    
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"    
    sleep 5
done
docker-compose down
mv .env.bk .env
docker rm -f "$container_id"
