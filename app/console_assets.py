"""Console 静态资源加载（进阶报告 P1-⑥）。

登录页与控制台的前端 HTML/JS/CSS 已拆分为 app/console/ 下的独立文件；
本模块在导入时一次性读入内存（原 main.py 的字符串常量语义不变：
LOGIN_HTML / DASHBOARD_HTML 仍是模块级常量，既有 import 与测试零改动）。
"""

import os

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name: str) -> str:
    path = os.path.join(_DIR, "console", name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


LOGIN_HTML = _load("login.html")
DASHBOARD_HTML = _load("dashboard.html")
