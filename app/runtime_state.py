import contextvars
import copy
import json
import os
import random
import tempfile
import threading

import config as app_config

# S-3：允许把状态落到挂载卷，避免 docker compose 重建后设置与 Cookie 全部丢失。
STATE_DIR = os.environ.get("STATE_DIR", ".")
STATE_FILE = os.path.join(STATE_DIR, "web_state.json")

# 多账号模式下，一次请求内所有 Cookie 凭证读取必须来自**同一份账号**：
# _get_cookie_string() 与 _get_project_id() 在重试/流式路径里会被多次调用，
# 若每次都重新轮询就会串号（cookie 是 A 的、project 是 B 的）。
# contextvars 天然按 asyncio task（= 一次请求）隔离：首个读取点选号并缓存，
# 同一请求后续读取复用；failover 切通道重发也复用同一账号（一致性优先）。
_current_cookie_account = contextvars.ContextVar(
    "current_cookie_account", default=None)

# 服务账号通道同一套快照机制：一次请求内 project/location/sa_json 恒来自同一账号。
_current_sa_account = contextvars.ContextVar(
    "current_sa_account", default=None)


def _sa_env_fallback() -> list:
    """服务账号环境变量兜底：VERTEX_SA_JSON（内联）或 VERTEX_SA_FILE（文件路径）。

    控制台未保存账号列表时使用，返回单账号列表（sa_json 为原始 JSON 字符串，
    location 默认 global，project_id 从 SA JSON 自身读取）。非法输入返回空列表。
    """
    raw = (app_config.VERTEX_SA_JSON or "").strip()
    if not raw and app_config.VERTEX_SA_FILE:
        try:
            with open(app_config.VERTEX_SA_FILE, "r", encoding="utf-8") as f:
                raw = f.read().strip()
        except OSError as e:
            print(f"⚠️ [服务账号] 读取 VERTEX_SA_FILE 失败，已忽略：{e}")
            return []
    if not raw:
        return []
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️ [服务账号] VERTEX_SA_JSON/VERTEX_SA_FILE 不是合法 JSON，已忽略。")
        return []
    if not isinstance(info, dict) or not info.get("client_email") or not info.get("private_key"):
        print("⚠️ [服务账号] 环境变量里的 SA JSON 缺少 client_email/private_key，已忽略。")
        return []
    return [{
        "project_id": str(info.get("project_id") or ""),
        "location": "global",
        "sa_json": raw,
    }]


class AppState:
    """运行态管理器（内存优先 + 写时落盘）。

    P1-4 的改动要点：
      - 旧实现每个 getter 都调 `_load_state()` 同步读盘。全项目有 20+ 处
        get_settings/get_setting/get_effective_settings 调用，且全在 async 请求路径上
        （每压缩一张图片还要再读一次），高并发下会把事件循环串行化。
        现在只在启动和显式 reload() 时读盘。
      - 落盘改为「临时文件 + os.replace」，避免进程崩溃时把 web_state.json 写坏。
      - getter 返回深拷贝，防止调用方无意间改到 model_overrides 这层嵌套字典。
      - 文件权限 0600：里面存着完整的 Google 会话 Cookie。
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._state = {"channel_strategy": "express"}
        self._cookie_rr_index = 0   # 多 Cookie 账号轮询指针（内存态，重启重头轮）
        self._sa_rr_index = 0       # 多服务账号轮询指针（内存态，重启重头轮）
        self._load_from_disk()

    # ---------- 持久化 ----------

    def _load_from_disk(self) -> None:
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # 旧版布尔开关 use_web_proxy → 三档 channel_strategy 迁移。
                # 判断以「磁盘原始数据」为准：_state 的初始默认键不能作为
                # "已迁移"的标志（否则旧文件永远走不到迁移分支）。
                if "channel_strategy" not in data and "use_web_proxy" in data:
                    data["channel_strategy"] = "cookie" if data.get("use_web_proxy") else "express"
                    data.pop("use_web_proxy", None)
                    print(f"🔄 [状态管理器] 检测到旧版通道开关 use_web_proxy，"
                          f"已迁移为通道策略 channel_strategy={data['channel_strategy']}。")
                self._state.update(data)
        except Exception as e:
            print(f"⚠️ [状态管理器] 无法读取持久化配置文件，已自动降级为内存模式: {e}")

    def _save(self) -> None:
        """原子写：先写同目录临时文件再 os.replace（同一文件系统内是原子操作）。"""
        try:
            target_dir = os.path.dirname(STATE_FILE) or "."
            os.makedirs(target_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".web_state-", suffix=".tmp", dir=target_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._state, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, STATE_FILE)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            try:
                os.chmod(STATE_FILE, 0o600)   # 里面有完整 Google Cookie
            except OSError:
                pass
        except Exception as e:
            print(f"⚠️ [状态管理器] 无法保存状态到磁盘: {e}")

    def reload(self) -> None:
        """显式从磁盘重载（外部改了文件时用）。"""
        with self._lock:
            self._load_from_disk()

    # ---------- 通道开关与凭证 ----------

    CHANNEL_STRATEGIES = ("express", "cookie", "vertex", "hybrid")

    def set_channel_strategy(self, strategy: str) -> bool:
        """四档通道策略：express=只走 API Key / cookie=只走 Cookie 直连 / vertex=只走服务账号 / hybrid=混合自动。"""
        strategy = (strategy or "").strip().lower()
        if strategy not in self.CHANNEL_STRATEGIES:
            return False
        with self._lock:
            if self._state.get("channel_strategy") != strategy:
                self._state["channel_strategy"] = strategy
                self._save()
            print(f"🔄 [状态管理器] 通道策略已更新：{strategy}")
            return True

    def get_channel_strategy(self) -> str:
        with self._lock:
            strategy = self._state.get("channel_strategy")
            if strategy not in self.CHANNEL_STRATEGIES:
                return "express"
            return strategy

    # ---- 旧布尔接口（向后兼容，内部映射到策略；新代码请用策略接口）----

    def enable_web_proxy(self, enabled: bool):
        self.set_channel_strategy("cookie" if enabled else "express")

    def is_web_proxy_enabled(self) -> bool:
        return self.get_channel_strategy() == "cookie"

    def set_google_cookie(self, cookie_str: str):
        with self._lock:
            self._state["google_cookie"] = cookie_str
            self._save()
            print("🔄 [状态管理器] 谷歌独立 Cookie 已保存到运行状态")

    def get_google_cookie(self) -> str:
        with self._lock:
            return self._state.get("google_cookie", "")

    def set_project_id(self, project_id: str):
        with self._lock:
            self._state["google_project_id"] = project_id
            self._save()
            print(f"🔄 [状态管理器] 项目 ID 已保存: {project_id}")

    def get_project_id(self) -> str:
        with self._lock:
            return self._state.get("google_project_id", "")

    # ---------- 多账号凭证管理（Express Key 列表 / Cookie 账号列表）----------

    def get_express_keys(self):
        """控制台管理的 Express Key 列表；None = 从未保存过（此时用环境变量）。"""
        with self._lock:
            keys = self._state.get("express_keys")
            if isinstance(keys, list):
                return [k for k in keys if k]
            return None

    def set_express_keys(self, keys: list) -> list:
        """整表覆盖保存 Express Key 列表（空列表 = 清空控制台列表，回落环境变量）。"""
        clean = []
        seen = set()
        for k in (keys or []):
            if not isinstance(k, str):   # None/数字等一律跳过，防误存 "None"
                continue
            k = k.strip()
            if k and k not in seen:
                clean.append(k)
                seen.add(k)
        with self._lock:
            self._state["express_keys"] = clean
            self._save()
        print(f"🔄 [状态管理器] 已保存 {len(clean)} 个 Express API Key（控制台管理）。")
        return clean

    def get_cookie_accounts(self) -> list:
        """Cookie 账号列表 [(cookie, project_id), ...]。

        新格式存 _state["cookie_accounts"]；旧版单账号字段（google_cookie /
        google_project_id）自动迁移成单元素列表视图，兼容老配置。
        """
        with self._lock:
            accounts = self._state.get("cookie_accounts")
            if isinstance(accounts, list):
                return [a for a in accounts if isinstance(a, dict) and a.get("cookie")]
            cookie = self._state.get("google_cookie", "")
            if cookie:
                return [{"cookie": cookie, "project_id": self._state.get("google_project_id", "")}]
            return []

    def set_cookie_accounts(self, accounts: list) -> list:
        """整表覆盖保存 Cookie 账号列表。

        每项必须含非空 cookie（project_id 可空）。保存后同步旧字段
        google_cookie / google_project_id = 第一个账号，保证旧读取接口仍有效。
        传空列表 = 清空全部账号（Cookie 通道回落环境变量）。
        """
        clean = []
        for a in (accounts or []):
            if not isinstance(a, dict):
                continue
            cookie = str(a.get("cookie") or "").strip()
            if not cookie:
                continue
            clean.append({
                "cookie": cookie,
                "project_id": str(a.get("project_id") or "").strip(),
            })
        with self._lock:
            self._state["cookie_accounts"] = clean
            self._state.pop("google_cookie", None)
            self._state.pop("google_project_id", None)
            if clean:
                # 旧读取接口（get_google_cookie / get_project_id / location 钉定）取第一个账号
                self._state["google_cookie"] = clean[0]["cookie"]
                self._state["google_project_id"] = clean[0]["project_id"]
            self._cookie_rr_index = 0
            _current_cookie_account.set(None)
            self._save()
        print(f"🔄 [状态管理器] 已保存 {len(clean)} 个 Cookie 账号。")
        return clean

    def get_current_cookie_account(self) -> tuple:
        """取本次请求使用的 Cookie 账号（请求级快照，见模块注释）。

        单账号直接返回；多账号按 roundrobin 设置轮询或随机选择，
        同一请求内（含重试、流式、failover 重发）复用同一账号。
        无任何账号时返回 (None, None)。
        """
        account = _current_cookie_account.get()
        if account is None:
            accounts = self.get_cookie_accounts()
            if not accounts:
                return (None, None)
            if len(accounts) == 1:
                account = accounts[0]
            else:
                use_roundrobin = bool(self.get_setting("roundrobin", app_config.ROUNDROBIN))
                with self._lock:
                    if use_roundrobin:
                        idx = self._cookie_rr_index % len(accounts)
                        self._cookie_rr_index = (self._cookie_rr_index + 1) % len(accounts)
                    else:
                        idx = random.randrange(len(accounts))
                    account = accounts[idx]
                print(f"🔑 [账号选择] 多 Cookie 账号轮询/随机选中第 {idx + 1}/{len(accounts)} 份。")
            _current_cookie_account.set(account)
        return (account.get("cookie", ""), account.get("project_id", ""))

    # ---------- 服务账号（第三通道）凭证管理 ----------

    def get_sa_accounts(self) -> list:
        """服务账号列表 [{project_id, location, sa_json}, ...]（控制台列表优先，空则回落环境变量）。"""
        with self._lock:
            accounts = self._state.get("sa_accounts")
            if isinstance(accounts, list):
                accounts = [a for a in accounts if isinstance(a, dict) and a.get("sa_json")]
                if accounts:
                    return accounts
        return _sa_env_fallback()

    def get_sa_accounts_console(self) -> list:
        """仅控制台持久化列表（不含环境变量兜底）；账号增删改用这个。"""
        with self._lock:
            accounts = self._state.get("sa_accounts")
            if isinstance(accounts, list):
                return [a for a in accounts if isinstance(a, dict) and a.get("sa_json")]
        return []

    def set_sa_accounts(self, accounts: list) -> list:
        """整表覆盖保存服务账号列表；空列表 = 清空控制台列表，回落环境变量。

        每项必须含非空 sa_json（project_id/location 可空，为空时用默认值）。
        """
        clean = []
        seen = set()
        for a in (accounts or []):
            if not isinstance(a, dict):
                continue
            sa_json = str(a.get("sa_json") or "").strip()
            if not sa_json or sa_json in seen:
                continue
            seen.add(sa_json)
            clean.append({
                "project_id": str(a.get("project_id") or "").strip(),
                "location": (str(a.get("location") or "global").strip() or "global"),
                "sa_json": sa_json,
            })
        with self._lock:
            self._state["sa_accounts"] = clean
            self._sa_rr_index = 0
            _current_sa_account.set(None)
            self._save()
        print(f"🔑 [状态管理器] 已保存 {len(clean)} 个服务账号。")
        return clean

    # ---------- 自定义模型列表（控制台可编辑，合并进模型列表） ----------

    def get_custom_models(self) -> list:
        """控制台添加的自定义模型名（持久化；合并进 /v1/models 与控制台模型下拉）。"""
        with self._lock:
            models = self._state.get("custom_models")
            if isinstance(models, list):
                return [str(m).strip() for m in models if str(m).strip()]
        return []

    def set_custom_models(self, models: list) -> list:
        """整表覆盖保存自定义模型列表（去空、去重）。"""
        clean = []
        seen = set()
        for m in (models or []):
            name = str(m).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            clean.append(name)
        with self._lock:
            self._state["custom_models"] = clean
            self._save()
        print(f"🔧 [状态管理器] 已保存 {len(clean)} 个自定义模型。")
        return clean

    def get_current_sa_account(self) -> tuple:
        """取本次请求使用的服务账号（请求级快照，同 Cookie 账号机制）。

        返回 (project_id, location, sa_json)。多账号按 roundrobin 设置轮询或随机选择，
        同一请求内（含重试、流式、failover 重发）复用同一账号；无任何账号返回 (None, None, None)。
        """
        account = _current_sa_account.get()
        if account is None:
            accounts = self.get_sa_accounts()
            if not accounts:
                return (None, None, None)
            if len(accounts) == 1:
                account = accounts[0]
            else:
                use_roundrobin = bool(self.get_setting("roundrobin", app_config.ROUNDROBIN))
                with self._lock:
                    if use_roundrobin:
                        idx = self._sa_rr_index % len(accounts)
                        self._sa_rr_index = (self._sa_rr_index + 1) % len(accounts)
                    else:
                        idx = random.randrange(len(accounts))
                    account = accounts[idx]
                print(f"🔑 [账号选择] 多服务账号轮询/随机选中第 {idx + 1}/{len(accounts)} 份。")
            _current_sa_account.set(account)
        return (account.get("project_id", ""), account.get("location", "global"), account.get("sa_json", ""))

    # ---------- 混合自动的可配置行为（hybrid 策略下生效）----------

    def get_hybrid_channels(self) -> list:
        """混合自动的通道顺序（只保留已知通道键，非法/空则回落默认顺序）。"""
        raw = self.get_setting("hybrid_channels", app_config.DEFAULT_SETTINGS["hybrid_channels"])
        known = ("express", "cookie", "vertex")
        if isinstance(raw, list):
            out = [c for c in raw if c in known]
            if out:
                return out
        return list(app_config.DEFAULT_SETTINGS["hybrid_channels"])

    def get_channel_retry(self, channel: str):
        """该通道的独立重试次数覆盖；无覆盖/非法返回 None（= 用全局 retry_max）。"""
        overrides = self.get_setting(
            "channel_retry_overrides", app_config.DEFAULT_SETTINGS["channel_retry_overrides"])
        if isinstance(overrides, dict) and channel in overrides:
            v = overrides.get(channel)
            if v is not None:
                try:
                    return max(0, min(50, int(v)))
                except (TypeError, ValueError):
                    return None
        return None

    # ---------- 控制台可调设置 ----------

    def get_settings(self) -> dict:
        """完整设置（内置默认 + 持久化覆盖），保证所有键都存在；返回深拷贝。"""
        with self._lock:
            merged = copy.deepcopy(app_config.DEFAULT_SETTINGS)
            stored = self._state.get("settings")
            if isinstance(stored, dict):
                for k, v in stored.items():
                    if k in merged:
                        merged[k] = copy.deepcopy(v)
            return merged

    def get_setting(self, key: str, default=None):
        with self._lock:
            stored = self._state.get("settings")
            if isinstance(stored, dict) and key in stored:
                return copy.deepcopy(stored[key])
            if key in app_config.DEFAULT_SETTINGS:
                return copy.deepcopy(app_config.DEFAULT_SETTINGS[key])
            return default

    def update_settings(self, patch: dict) -> dict:
        """合并更新设置，只接受已知键，返回更新后的完整设置。"""
        if not isinstance(patch, dict):
            return self.get_settings()
        with self._lock:
            current = self._state.get("settings")
            current = dict(current) if isinstance(current, dict) else {}
            accepted = 0
            for k, v in patch.items():
                if k in app_config.DEFAULT_SETTINGS and k != "model_overrides":
                    current[k] = v
                    accepted += 1
            self._state["settings"] = current
            self._save()
            print(f"🔧 [状态管理器] 已更新 {accepted} 项运行时设置。")
        return self.get_settings()

    # ---------- 按模型参数覆盖 ----------

    def get_model_overrides(self) -> dict:
        with self._lock:
            stored = self._state.get("settings")
            if isinstance(stored, dict) and isinstance(stored.get("model_overrides"), dict):
                return copy.deepcopy(stored["model_overrides"])
            return {}

    def set_model_override(self, model_name: str, patch: dict) -> dict:
        model_name = (model_name or "").strip()
        if not model_name or not isinstance(patch, dict):
            return {}
        clean = {k: v for k, v in patch.items() if k in app_config.PER_MODEL_KEYS}
        with self._lock:
            settings = self._state.get("settings")
            settings = dict(settings) if isinstance(settings, dict) else {}
            overrides = settings.get("model_overrides")
            overrides = dict(overrides) if isinstance(overrides, dict) else {}
            overrides[model_name] = clean
            settings["model_overrides"] = overrides
            self._state["settings"] = settings
            self._save()
            print(f"🔧 [状态管理器] 已保存模型 {model_name} 的专属参数（{len(clean)} 项）。")
            return clean

    def clear_model_override(self, model_name: str) -> bool:
        model_name = (model_name or "").strip()
        with self._lock:
            settings = self._state.get("settings")
            if not isinstance(settings, dict):
                return False
            overrides = settings.get("model_overrides")
            if not isinstance(overrides, dict) or model_name not in overrides:
                return False
            overrides.pop(model_name, None)
            settings["model_overrides"] = overrides
            self._state["settings"] = settings
            self._save()
            print(f"🔧 [状态管理器] 已清除模型 {model_name} 的专属参数。")
            return True

    def get_effective_settings(self, model_name: str) -> dict:
        """该模型生效的设置：全局默认叠加该模型专属覆盖（仅 PER_MODEL_KEYS）。"""
        base = self.get_settings()
        overrides = base.get("model_overrides") or {}
        ov = overrides.get((model_name or "").strip())
        if isinstance(ov, dict):
            for k in app_config.PER_MODEL_KEYS:
                if k in ov:
                    base[k] = ov[k]
        return base


# 单例模式导出
app_state = AppState()
