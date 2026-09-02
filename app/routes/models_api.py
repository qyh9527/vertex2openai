import time
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Set
from auth import get_api_key
from model_loader import get_express_models
from runtime_state import app_state

router = APIRouter()


async def _list_models():
    current_time = time.time()
    # 不再自动从远程刷新模型配置（曾每 3600s 自动拉取）——远程获取改为控制台
    # 「获取远程模型」按钮手动触发；这里用磁盘缓存 + 本地配置 + 内置兜底 + 自定义模型。
    # 模型列表是配置数据（与通道凭证无关），始终返回，供客户端发现可用模型。
    raw_models = await get_express_models()

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


@router.get("/v1/models")
async def list_models(fastapi_request: Request, api_key: str = Depends(get_api_key)):
    """保持原有无渠道前缀的模型列表入口。"""
    return await _list_models()


@router.get("/{channel}/v1/models")
async def list_models_for_channel(channel: str, fastapi_request: Request,
                                  api_key: str = Depends(get_api_key)):
    """显式渠道入口的模型列表，便于 OpenAI 客户端把 base_url 设为 /<channel>/v1。"""
    if channel not in {"express", "cookie", "vertex"}:
        return JSONResponse(status_code=404, content={"error": {
            "message": f"未知的独立渠道：{channel}。可用渠道为 express / cookie / vertex。",
            "type": "invalid_request_error",
            "code": 404,
        }})
    return await _list_models()
