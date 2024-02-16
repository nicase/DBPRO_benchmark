container_name="qdrant_container"
memory_limit="512m"
cpu_limit="0.5"

docker run -d --name "$container_name" --memory="$memory_limit" --cpus="$cpu_limit" qdrant/qdrant

container_id=$(docker ps -aqf "name=$container_name")

python3 ann/qdrant_ann.py &

python_pid=$!

while ps -p $python_pid > /dev/null; do
    
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"    
    sleep 5

done

docker rm -f "$container_id"
