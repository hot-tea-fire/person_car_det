
import time

from abc import abstractmethod
from threading import Thread

from logging import getLogger
from prettytable import PrettyTable

from forest_arithmetic_service_20.setting.http_setting import http_server_port
from forest_arithmetic_service_20.setting import common_setting, kafka_setting, model_setting

logger = getLogger('simple')


def logger_out_basic():
    """项目基本配置"""
    logger.warning('project initialing...')

    tb = PrettyTable()
    tb.field_names = ['设置名称', '生效的值', '备注']

    tb.add_row(['project_path', common_setting.project_path, '项目主路径'])
    tb.add_row(['log_path', common_setting.log_path, '日志路径'])
    tb.add_row(['cache', common_setting.cache_path, '缓存路径'])
    tb.add_row(['observers', common_setting.observer_folder_path, '算法检测路径'])

    logger.warning('基础配置表：\n' + tb.get_string())


def logger_out_profile():
    """项目各组件的地址"""

    tb = PrettyTable()
    tb.field_names = ['设置名称', '生效的值', '备注']

    tb.add_row(['Kafka地址', kafka_setting.KAFKA_ADDRESS[0], 'Kafka地址'])
    tb.add_row(['Triton Server地址', model_setting.TRITON_SERVER_ADDRESS, '算法推导服务器地址'])
    tb.add_row(['布撤防监听接口端口', http_server_port, '布防监听端口'])

    logger.warning('各数据源地址：\n' + tb.get_string())


class DaemonWatcher(Thread):

    def __init__(self, check_cycle_time: int = common_setting.watcher_check_cycle) -> None:
        self.check_cycle_time = check_cycle_time
        super().__init__(daemon=True)

    @abstractmethod
    def rebuild(self):
        pass

    @abstractmethod
    def rebuild_condition(self) -> bool:
        pass

    def run(self):
        logger.warning(f'Start DaemonWatcher: {self.__class__.name}')
        while True:
            try:
                condition = self.rebuild_condition()
            except Exception as e:
                logger.warning((f'{self.__class__.__name__} condition failed.'))
                condition = False

            try:
                if condition:
                    logger.warning(f'{self.__class__.__name__} checked success!')
                    time.sleep(self.check_cycle_time)
                else:
                    self.rebuild()
                    assert self.rebuild_condition(), 'check after rebuild failed.'
            except Exception as e:
                logger.error(f'{self.__class__.__name__} build failed. {e}')
                time.sleep(5)
            finally:
                time.sleep(1)
