# -*- coding: utf-8 -*-
"""通用设置.
"""
from os import path, mkdir

# 项目主路径: 默认为common.setting的上一级目录

# Dynamic
project_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))

# 日志路径：主路径下的logs folder
log_path = path.join(project_path, 'logs')

# 缓存主路径
cache_path = path.join(project_path, 'cache')

# observer path
# 算法检测路径
observer_folder_path = path.join(log_path, 'observer')

# watcher retry delay:
watcher_check_cycle: int = 5 * 60
