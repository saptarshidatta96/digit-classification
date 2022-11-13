# Docker Commands

1. `docker build -t dc:v1 -f Dockerfile .`

2. `docker run -p 12345:12345 -it dc:v1`

# Curl Command

`curl http://127.0.0.1:12345/predict -X POST  -H 'Content-Type: application/json' -d '{
        "image1": [ "0", "0", "1", "11", "14", "15", "3", "0", "0", "1", "13", "16", "12", "16", "8", "0", "0", "8", "16", "4", "6", "16", "5", "0", "0", "5", "15", "11", "13", "14", "0", "0", "0", "0", "2", "12", "16", "13", "0", "0", "0", "0", "0", "13", "16", "16", "6",  "0",  "0",  "0",  "0", "16", "16", "16", "7",  "0",  "0",  "0",  "0", "11", "13", "12",  "1",  "0"],
        "image2": [ "0", "0", "1", "11", "14", "15", "3", "0", "0", "1", "13", "16", "12", "16", "8", "0", "0", "8", "16", "4", "6", "16", "5", "0", "0", "5", "15", "11", "13", "14", "0", "0", "0", "0", "2", "12", "16", "13", "0", "0", "0", "0", "0", "13", "16", "16", "6",  "0",  "0",  "0",  "0", "16", "16", "16", "7",  "0",  "0",  "0",  "0", "11", "13", "12",  "1",  "0"]
    }'`
