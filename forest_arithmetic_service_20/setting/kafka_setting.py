# -*- coding: utf-8 -*-
"""Setting of Kafka connect.

KAFKA连接设置文件。

1. KAFKA_LISTEN_TOPIC: 需要消费的Topic。

2. KAFKA_PUSH_TOPIC: 需要发送的Topic。

3. KAFKA_ADDRESS: Kafka地址。

"""

from typing import List

KAFKA_LISTEN_TOPIC: List[str] = [
    "AddDefence",
    "SysConfig",
    "WeatherInfo",
    "PresetArea",
    "imageData"
]


KAFKA_PUSH_TOPIC: List[str] = [
    "RequestConfig",
    "AlarmInfo",
    "AreaResponse",
    "yes",
    "defenceDevice"
]

# kafka地址
KAFKA_HOST: str = '192.168.2.13'
KAFKA_PORT: int = 9092

KAFKA_ADDRESS: List[str] = [KAFKA_HOST + ':' + str(KAFKA_PORT)]
