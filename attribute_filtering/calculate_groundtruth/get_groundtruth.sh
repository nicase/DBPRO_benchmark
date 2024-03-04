
python3 gloVe/add_attributes.py &
PID_GLOVE=$!


python3 SIFT/add_attributes.py &
PID_SIFT=$!

wait $PID_GLOVE
wait $PID_SIFT

python3 gloVe/cosine.py &
python3 gloVe/dot.py &
python3 gloVe/euclidean.py &

python3 SIFT/cosine.py &
python3 SIFT/dot.py &
python3 SIFT/euclidean.py &