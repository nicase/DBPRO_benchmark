container_id="milvus-standalone"

python3 ann/milvus_ann.py &

python_pid=$!

while ps -p $python_pid > /dev/null; do

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"    
    sleep 5

done
