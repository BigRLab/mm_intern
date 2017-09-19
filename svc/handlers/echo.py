# coding=utf8
"""test moa --> proxy --> fcgi"""
import json


def echo(environ, start_response):
    env_copy = environ.copy()
    # 这两个key-value不能被序列化,转str
    env_copy["wsgi.errors"] = "%s" % (env_copy["wsgi.errors"], )
    env_copy["wsgi.input"] = "%s" % (env_copy["wsgi.input"], )

    # 返回响应
    response = {"ec": 0, "em": "ok", "result": env_copy}
    response = json.dumps(response, indent=4)
    start_response('200 OK', [('Content-type', 'text/plain')])
    yield response
