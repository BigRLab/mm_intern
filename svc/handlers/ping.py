# coding=utf8
"""java-moa-proxy 周期性 ping, 用于探活"""


def ping(environ, start_response):
    start_response('200 OK', [('Content-type', 'text/plain')])
    yield "pong"
