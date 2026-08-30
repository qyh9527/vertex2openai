import time
from fastapi import APIRouter, Depends, Request
from typing import List, Dict, Any, Set
from auth import get_api_key
from model_loader import get_express_models, refresh_models_config_cache
from runtime_state import app_state

router = APIRouter()
_last_model_fetch_time = 0


@router.get("/v1/models")
async def list_models(fastapi_request: Request, api_key: str = Depends(get_api_key)):
    global _last_model_fetch_time

    current_time = time.time()
    if current_time - _last_model_fetch_time > 3600:
        await refresh_models_config_cache()
        _last_model_fetch_time = current_time

    express_key_manager_instance = fastapi_request.app.state.express_key_manager
    
    # 动态放行：开启 Cookie 直连 / 服务账号 / 混合策略，或配置有 Express API Key，
    # 均可安全获取模型列表（模型列表与通道无关，任一通道有凭证即可）
    has_web_proxy = app_state.get_channel_strategy() != "express"   # cookie / vertex / hybrid 都放行
    has_sa_account = bool(app_state.get_sa_accounts())
    has_express_key = express_key_manager_instance.get_total_keys() > 0

    raw_models = await get_express_models() if (has_express_key or has_web_proxy or has_sa_account) else []

    final_model_list: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()
    # 假流式开关 = 注册 fake- 前缀模型：开启后列表额外暴露 fake-<模型名> 条目，
    # 客户端选中即对该请求强制假流式（其余模型保持真实流式）。
    has_fake_variants = bool(app_state.get_setting("fake_streaming", False))

    def add_model(base_id: str):
        suffixes = [""]
        if "gemini" in base_id.lower() and "image" not in base_id.lower():
            suffixes.append("-search")

        for suffix in suffixes:
            final_id = f"{base_id}{suffix}"
            if final_id in processed_ids:
                continue
            final_model_list.append({
                "id": final_id,
                "object": "model",
                "created": int(current_time),
                "owned_by": "google",
                "permission": [],
                "root": base_id,
                "parent": None,
            })
            processed_ids.add(final_id)
            if has_fake_variants:
                fake_id = f"fake-{final_id}"
                if fake_id in processed_ids:
                    continue
                final_model_list.append({
                    "id": fake_id,
                    "object": "model",
                    "created": int(current_time),
                    "owned_by": "google",
                    "permission": [],
                    "root": f"fake-{base_id}",
                    "parent": None,
                })
                processed_ids.add(fake_id)

    for model_id in raw_models:
        add_model(model_id)

    return {"object": "list", "data": sorted(final_model_list, key=lambda item: item["id"])}