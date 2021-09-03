import json
import time
from logging import getLogger
from queue import Queue
from threading import Thread
from typing import List, Dict, NoReturn, Optional

from kafka import KafkaConsumer

from forest_arithmetic_service_20.common_tools import DaemonWatcher
from forest_arithmetic_service_20.setting.kafka_setting import KAFKA_LISTEN_TOPIC, KAFKA_HOST, KAFKA_PORT

# global variable
logger = getLogger('kafka_pull')
"""Logger: KAFKA消费者日志"""

kafka_consumer: Optional[KafkaConsumer] = None
"""KafkaConsumer: KAFKA消费者对象"""

KAFKA_QUEUE_MAPPER: Dict[str, Queue] = {x: Queue() for x in KAFKA_LISTEN_TOPIC}
"""Dict[str, Queue]: 缓存队列映射字典"""

# setting:
current_kafka_listen_topic: List[str] = KAFKA_LISTEN_TOPIC

current_kafka_host: str = str(KAFKA_HOST)
"""kafka host"""
current_kafka_port: str = str(KAFKA_PORT)
"""kafka port"""

current_kafka_address = [current_kafka_host + ':' + current_kafka_port]


def initial_kafka_consumer() -> NoReturn:
    """初始化Kafka"""
    global kafka_consumer
    kafka_consumer = KafkaConsumer(
        *current_kafka_listen_topic,
        bootstrap_servers=current_kafka_address,
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )
    logger.warning(f'Kafka Consumer initial success. {current_kafka_address}')


def connect_kafka_consumer() -> bool:
    """连接检测"""
    global kafka_consumer
    return kafka_consumer and kafka_consumer.bootstrap_connected()


class KafkaConsumerDaemonWatcher(DaemonWatcher):

    def rebuild(self):
        initial_kafka_consumer()

    def rebuild_condition(self) -> bool:
        return connect_kafka_consumer()


class KafkaConsumerCache(Thread):
    """The global cache consumer.

    监听所有的KAFKA_TOPIC并且获取数据，然后根据topic放入KAFKA_QUEUE_MAPPER中。

    需要获取kafka消息时，只需要根据topic来获取对应中的队列，再获取队列中的对象即可。
    """

    def __init__(self):
        super(KafkaConsumerCache, self).__init__()
        self.setDaemon(True)
        Thread.start(self)

    def run(self) -> None:
        global kafka_consumer

        # logger.warning('kafka consumer sub thread start.')
        logger.warning(f'开启 KAFKA 自动消费者线程. 目前响应的全局TOPIC: {list(KAFKA_QUEUE_MAPPER.keys())}')

        while True:

            try:
                for data in kafka_consumer:
                    d = data._asdict()

                    logger.warning(f'kafka consumer get msg: {str(d)}')

                    current_topic = d.get('topic')

                    if current_topic in KAFKA_LISTEN_TOPIC:
                        current_queue = KAFKA_QUEUE_MAPPER[current_topic]
                        current_data = d.get('value')

                        current_queue.put(current_data)


            except Exception:
                time.sleep(1)
                logger.debug('Failed in kafka consumer cache.')


watcher = KafkaConsumerDaemonWatcher()


def build_kafka_consumer():
    watcher.start()
    consumer_cache = KafkaConsumerCache()
