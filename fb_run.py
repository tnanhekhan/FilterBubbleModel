import sys
from fb_server import server

if len(sys.argv) == 1:
    server.port = 8521
else:
    server.port = int(sys.argv[1])

server.launch()