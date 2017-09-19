# coding=utf8
"""URL映射表"""

from handlers.echo import echo
from handlers.ping import ping
from handlers.live_spam_predict import predict as liveSpamPredict

PING_URI = "/php-fpm-ping"

URLS = {
    "/echo": echo,
    "/ping": ping,
    "/liveSpamPredict": liveSpamPredict,
    PING_URI: ping,
}

DEFAULT_HANDLER = echo
