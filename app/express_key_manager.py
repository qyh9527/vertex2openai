import random
from typing import List, Optional, Tuple
import config as app_config
from runtime_state import app_state


class ExpressKeyManager:
    """管理 Agent Platform (原 Vertex AI) Express Mode API Key，支持随机或轮询选择。

    来源优先级：控制台持久化的 express_keys（web_state.json）> 环境变量
    VERTEX_EXPRESS_API_KEY（初始值/兜底）。控制台保存后调 refresh_keys() 热生效。
    """

    def __init__(self):
        self._controlled_keys: Optional[List[str]] = None
        self.express_keys: List[str] = []
        self.round_robin_index: int = 0
        self.refresh_keys()

    def refresh_keys(self):
        controlled = app_state.get_express_keys()
        self._controlled_keys = controlled
        self.express_keys = controlled if controlled else list(app_config.VERTEX_EXPRESS_API_KEY_VAL)
        self.round_robin_index = 0
        source = "控制台" if controlled else "环境变量"
        print(f"🔄 [密钥刷新] 已从{source}加载 {len(self.express_keys)} 个 Express API Key。")

    def get_total_keys(self) -> int:
        return len(self.express_keys)

    def get_random_express_key(self) -> Optional[Tuple[int, str]]:
        if not self.express_keys:
            print("❌ [密钥配置] 未配置 VERTEX_EXPRESS_API_KEY，无法调用 Gemini Express Mode。")
            return None

        indexed_keys = list(enumerate(self.express_keys))
        random.shuffle(indexed_keys)
        original_idx, key = indexed_keys[0]
        print(f"🔑 [密钥选择] 已随机选择第 {original_idx + 1} 个 Express API Key。")
        return original_idx, key

    def get_roundrobin_express_key(self) -> Optional[Tuple[int, str]]:
        if not self.express_keys:
            print("❌ [密钥配置] 未配置 VERTEX_EXPRESS_API_KEY，无法调用 Gemini Express Mode。")
            return None

        if self.round_robin_index >= len(self.express_keys):
            self.round_robin_index = 0

        key = self.express_keys[self.round_robin_index]
        original_idx = self.round_robin_index
        self.round_robin_index = (self.round_robin_index + 1) % len(self.express_keys)
        print(f"🔑 [密钥选择] 已按轮询策略选择第 {original_idx + 1} 个 Express API Key。")
        return original_idx, key

    def get_express_api_key(self) -> Optional[Tuple[int, str]]:
        if app_state.get_setting("roundrobin", app_config.ROUNDROBIN):
            return self.get_roundrobin_express_key()
        return self.get_random_express_key()

    def get_all_keys_indexed(self) -> List[Tuple[int, str]]:
        return list(enumerate(self.express_keys))
