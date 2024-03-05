if [ "$1" = "prod" ]; then
    environment="prod"
else
    environment="test"
fi

python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/attribute_filtering/calculate_groundtruth/gloVe/add_attributes.py "$environment" &
PID_GLOVE=$!


python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/attribute_filtering/calculate_groundtruth/SIFT/add_attributes.py "$environment" &
PID_SIFT=$!

wait $PID_GLOVE
wait $PID_SIFT

python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/attribute_filtering/calculate_groundtruth/gloVe/cosine.py &
python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/attribute_filtering/calculate_groundtruth/gloVe/dot.py &
python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/attribute_filtering/calculate_groundtruth/gloVe/euclidean.py &

python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/attribute_filtering/calculate_groundtruth/SIFT/cosine.py &
python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/attribute_filtering/calculate_groundtruth/SIFT/dot.py &
python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/attribute_filtering/calculate_groundtruth/SIFT/euclidean.py &

wait

echo "All scripts finished successfully"