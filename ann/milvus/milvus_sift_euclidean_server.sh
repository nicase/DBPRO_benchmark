
docker-compose -f docker-compose-mlv.yml up -d

container_id=$(docker ps -qf "name=milvus-standalone")

container_ip=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $container_id)

cp .env .env.bk
echo "MILVUS_URL=$container_ip" >> .env

python3 ann/milvus/milvus_ann_sift_euclidean.py &

python_pid=$!

while ps -p $python_pid > /dev/null; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"
    sleep 5
done

mv .env.bk .env

docker-compose -f docker-compose-mlv.yml down
