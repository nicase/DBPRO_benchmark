container_id="349e7ccd7e68b8791385e59818f0e601b9b1be99140a13619bcd3a19c5ddbba1"

docker start $container_id

python3 ann/qdrant_ann.py &

python_pid=$!

while ps -p $python_pid > /dev/null; do
    
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"    
    sleep 5

done

docker stop $container_id