"""pytest 全局配置。

- 把 app/ 加入 sys.path：app 内模块使用顶层导入（`import config`、`from runtime_state import app_state`），
  测试模块必须能在根命名空间里 import 它们。
- 在导入任何 app 模块之前把 STATE_DIR 指向一个独立临时目录：
  runtime_state / logger 在模块导入时就会读 STATE_DIR（读 web_state.json、建日志文件），
  绝不能让它碰真实数据或真实挂载卷。
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"

# 必须在 import 任何 app 模块之前执行（pytest 收集 conftest 先于测试模块）
_STATE_TMP = tempfile.mkdtemp(prefix="vertex2openai_test_state_")
os.environ["STATE_DIR"] = _STATE_TMP

sys.path.insert(0, str(APP_DIR))


@pytest.fixture(autouse=True)
def _clear_account_context():
    """每个测试前后清空请求级账号快照（contextvar 跨测试会残留）。"""
    import runtime_state
    runtime_state._current_cookie_account.set(None)
    runtime_state._current_sa_account.set(None)
    yield
    runtime_state._current_cookie_account.set(None)
    runtime_state._current_sa_account.set(None)
