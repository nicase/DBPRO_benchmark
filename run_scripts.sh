#!/bin/bash

# Execute ann/qdrant_ann.py
echo "Executing qdrant_ann.py"
python3 ann/qdrant_ann.py

# Check the exit status of the previous command
if [ $? -eq 0 ]; then
    # If the previous command was successful, proceed to the next script
    echo "qdrant_ann.py executed successfully"
    
    # Execute ann/milvus_ann.py
    echo "Executing milvus_ann.py"
    python3 ann/milvus_ann.py

    # Check the exit status of the previous command
    if [ $? -eq 0 ]; then
        # If the previous command was successful, proceed to the next script
        echo "milvus_ann.py executed successfully"

        # Execute ann/weaviate_ann.py
        echo "Executing weaviate_ann.py"
        python3 ann/weaviate_ann.py

        # Check the exit status of the previous command
        if [ $? -eq 0 ]; then
            # If the previous command was successful, print success message
            echo "weaviate_ann.py executed successfully"
        else
            # If the third script failed, print an error message
            echo "Error: weaviate_ann.py failed to execute"
        fi
    else
        # If the second script failed, print an error message
        echo "Error: milvus_ann.py failed to execute"
    fi
else
    # If the first script failed, print an error message
    echo "Error: qdrant_ann.py failed to execute"
fi
