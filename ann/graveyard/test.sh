if [ "$1" = "prod" ]; then
    echo "Running prod env..."
    cp .env.prod .env

else
    echo "Running local env..."
fi