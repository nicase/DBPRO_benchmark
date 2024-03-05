if [ "$1" = "prod" ]; then
    environment="prod"
else
    environment="test"
fi

python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/ann/calculate_groundtruth/gloVe/cosine.py "$environment" &
python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/ann/calculate_groundtruth/gloVe/dot.py "$environment" &
python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/ann/calculate_groundtruth/gloVe/euclidean.py "$environment" &

python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/ann/calculate_groundtruth/SIFT/cosine.py "$environment" &
python3 /Users/nicolascamerlynck/Documents/WS2324/DBPRO/DBPRO_benchmark/ann/calculate_groundtruth/SIFT/dot.py "$environment" &

wait

echo "All scripts executed succesfully"