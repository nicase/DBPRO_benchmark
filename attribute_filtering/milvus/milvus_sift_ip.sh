docker-compose -f docker-compose-mlv.yml up -d

container_id="milvus-standalone"

python3 milvus_sift_attribute_filtering_ip.py &

python_pid=$!

while ps -p $python_pid > /dev/null; do

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"
    sleep 5

done
docker-compose -f docker-compose-mlv.yml down
