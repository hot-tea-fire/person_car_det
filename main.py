import time
from logging import getLogger

from forest_arithmetic_service_20.clients_main import build_kafka_consumer
from forest_arithmetic_service_20.common_tools import logger_out_basic, logger_out_profile

logger = getLogger('kafka_pull')
"""Logger: KAFKA消费者日志"""


def fun():
    print('hello world')
    logger.warning('hello world')


if __name__ == '__main__':
    # 打印项目基本信息
    logger_out_basic()
    # 打印项目各组件地址
    logger_out_profile()

    # 建立主进程的KAFKA监听器
    # 接收器
    build_kafka_consumer()

    fun()
    while 1:
        time.sleep(3)
