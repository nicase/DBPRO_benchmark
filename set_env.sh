if [ "$1" = "prod" ]; then
    echo "PROD env set"
    cp .env.prod .env

else
    echo "LOCAL env set"
    cp .env.local .env
fi