container_name="weviate"
memory_limit="512m"
cpu_limit="0.5"
port_mapping="8080:8080"

docker run -d --name "$container_name" -p $port_mapping --memory="$memory_limit" --cpus="$cpu_limit" weviate/weviate

container_id=$(docker ps -aqf "name=$container_name")

container_port=$(docker port "$container_id" | grep "tcp" | awk '{print $NF}' | cut -d ':' -f 2)
echo "WEVIATE_PORT=\"$container_port\"" >> .env

python3 attribute_filtering/weviate_attribute_filtering.py &

python_pid=$!

while ps -p $python_pid > /dev/null; do
    
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"    
    sleep 5

done

docker rm -f "$container_id"
