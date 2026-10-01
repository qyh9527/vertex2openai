import json


_REQUEST_FIELDS = (
    "labels", "media_resolution", "media_processing", "audio_transcription_config",
    "translation_config", "enable_affective_dialog", "speech_config",
    "mediaResolution", "mediaProcessing", "audioTranscriptionConfig",
    "translationConfig", "enableAffectiveDialog", "speechConfig",
)
_PART_FIELDS = ("media_processing", "audio_transcription", "speech_metadata", "part_metadata",
                "media_resolution")


class ConversionReport:
    """请求内的字段级诊断；只存固定路径和原因，不保存字段值。"""

    def __init__(self, channel):
        self.channel = channel or "express"
        self._entries = {}

    def record(self, path, action, reason, channel=None):
        entry = {"path": path, "action": action, "reason": reason,
                 "channel": channel or self.channel}
        self._entries[tuple(entry.values())] = entry

    def to_list(self):
        return [dict(entry) for entry in self._entries.values()]

    def checkpoint(self):
        return dict(self._entries)

    def reset(self, checkpoint):
        self._entries = dict(checkpoint)

    def inspect_request(self, request):
        for field in _REQUEST_FIELDS:
            if getattr(request, field, None) is not None:
                self.record(field, "unsupported", "not_mapped_to_upstream")
        for message in request.messages:
            content = message.content
            if not isinstance(content, list):
                continue
            for part in content:
                if hasattr(part, "model_dump"):
                    part = part.model_dump()
                if not isinstance(part, dict):
                    continue
                for field in _REQUEST_FIELDS:
                    if part.get(field) is not None:
                        self.record("messages.content." + field, "unsupported", "not_mapped_to_upstream")

    def inspect_response(self, response):
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                for field in _PART_FIELDS:
                    if getattr(part, field, None) is not None:
                        self.record("response.parts." + field, "unsupported", "not_representable_in_openai")
                for field in ("inline_data", "file_data"):
                    media = getattr(part, field, None)
                    if getattr(media, "display_name", None) is not None:
                        self.record("response.parts." + field + ".display_name", "omitted", "not_representable_in_openai")

    def attach(self, target):
        target.setdefault("extra_content", {}).setdefault("vertex2openai", {})["conversion_report"] = self.to_list()

    def log(self):
        if self._entries:
            print("[转换诊断] " + json.dumps(self.to_list(), ensure_ascii=False))
