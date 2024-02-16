container_id="milvus-standalone"

python3 attribute_filtering/milvus_attribute_filtering.py &

python_pid=$!

while ps -p $python_pid > /dev/null; do

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"    
    sleep 5

done
