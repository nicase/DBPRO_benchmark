container_id="349e7ccd7e68b8791385e59818f0e601b9b1be99140a13619bcd3a19c5ddbba1"

while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"
    
    sleep 5
done

