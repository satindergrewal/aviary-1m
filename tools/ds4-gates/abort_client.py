#!/usr/bin/env python3
"""Dead-client simulator for abort_paths_gate.sh (P0-3).

Sends a /completion POST over a raw socket, then blocks reading the (streaming)
response. On SIGTERM it dies the way the MODE says a client dies:
  fin   - graceful close: shutdown(WR) then close -> server peer sees EOF (FIN)
  rst   - abort: SO_LINGER(1,0) then close -> server peer sees ECONNRESET (RST)
For SIGKILL death, the gate just kill -9's this process (kernel closes with FIN).

Why not curl: backgrounded jobs in non-job-control shells IGNORE SIGINT, and curl
gives no control over FIN-vs-RST - two silent gate invalidations came from that.
"""
import json
import signal
import socket
import struct
import sys
import urllib.parse

url, prompt_file, mode = sys.argv[1], sys.argv[2], sys.argv[3]
u = urllib.parse.urlparse(url)

with open(prompt_file, "rb") as f:
    body = f.read()

sock = socket.create_connection((u.hostname, u.port), timeout=600)

def die(_sig, _frm):
    try:
        if mode == "rst":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            sock.close()  # -> RST
        else:  # fin
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()  # -> FIN
    finally:
        sys.exit(0)

signal.signal(signal.SIGTERM, die)

req = (
    f"POST /completion HTTP/1.1\r\n"
    f"Host: {u.hostname}:{u.port}\r\n"
    f"Content-Type: application/json\r\n"
    f"Content-Length: {len(body)}\r\n"
    f"Connection: keep-alive\r\n\r\n"
).encode() + body
sock.sendall(req)

# Block reading the response until killed; discard everything.
try:
    while True:
        data = sock.recv(65536)
        if not data:
            break
except Exception:
    pass
