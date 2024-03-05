container_name="qdrant_container"
memory_limit="512m"
cpu_limit="0.5"
port_mapping="6333:6333"
LOGS_file="SIFT_cosine.log"

echo "Run this script from the root dir!!"

# SIFT_COSINE
docker run -d --name "$container_name" -p $port_mapping --memory="$memory_limit" --cpus="$cpu_limit" qdrant/qdrant

container_id=$(docker ps -aqf "name=$container_name")

container_port=$(docker port "$container_id" | grep "tcp" | awk '{print $NF}' | cut -d ':' -f 2)
cp .env .env.bk
echo "" >> .env
echo "QDRANT_PORT=\"$container_port\"" >> .env

python3 ann/qdrant/qdrant_SIFT_cosine.py &

python_pid=$!

while ps -p $python_pid > /dev/null; do

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp" > "$LOGS_file" 
    sleep 5

done

docker rm -f "$container_id"
mv .env.bk .env

