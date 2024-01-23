container_id="349e7ccd7e68b8791385e59818f0e601b9b1be99140a13619bcd3a19c5ddbba1"

while true; do
    # Get the current timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Run docker stats and format the output with the timestamp
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"
    
    # Adjust the sleep duration based on your monitoring needs
    sleep 1
done

