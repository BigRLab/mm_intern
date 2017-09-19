# coding=utf8
from gevent import monkey
monkey.patch_all()

import time
import os
import json
from cgi import FieldStorage
from gevent_fastcgi.server import FastCGIServer
from gevent_fastcgi.wsgi import WSGIRequestHandler
from urls import URLS, DEFAULT_HANDLER, PING_URI
from core import log

def dispatch(environ, start_response):
    start_time = time.time()
    if environ.get("SCRIPT_NAME", "") == PING_URI or environ.get("SCRIPT_FILENAME", "") == PING_URI:
        request_uri = PING_URI
    else:
        env_copy = environ.copy()
        post_data = __parse_post_body(env_copy)
        environ['start_time'] = start_time
        environ["post_data"] = post_data

        # 优先使用moa-json-params-m指定的值作为请求方法
        try:
            request_uri = "/%s" % (post_data["params"]["m"], )
        except KeyError:
            request_uri = environ.get("PATH_INFO", "")
            if not request_uri:
                request_uri = environ.get("REQUEST_URI", "").split("?")[0]
        request_uri = request_uri.rstrip("/")
        log.write_info('Request ' + json.dumps(post_data, ensure_ascii=False))

    handler_func = URLS.get(request_uri, DEFAULT_HANDLER)
    return handler_func(environ, start_response)


def __parse_post_body(environ):
    post_data = {}
    storage = FieldStorage(fp=environ['wsgi.input'], environ=environ, keep_blank_values=True)

    if environ["REQUEST_METHOD"] == "POST":
        # moa-proxy post json, but use bad header: application/x-www-form-urlencoded
        # we must parse total body as a json
        if environ["CONTENT_TYPE"] == "application/x-www-form-urlencoded":
            try:
                key = storage.keys()[0]
                post_data = json.loads(key)
                return post_data
            except:
                pass
        # 使FieldStorage支持application/json
        elif environ["CONTENT_TYPE"] == "application/json":
            post_data = json.loads(storage.value)
            return post_data

    # 兼容moa-proxy post完毕, 正常解析get/post
    # warning: get data also parse into post data
    for k in storage.keys():
        post_data[k] = storage.getvalue(k)

    return post_data


request_handler = WSGIRequestHandler(dispatch)
server = FastCGIServer(('127.0.0.1', 9000), request_handler)
server.serve_forever()
