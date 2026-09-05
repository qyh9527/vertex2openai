"""控制台数据概览回归测试：图表不应因稀疏按天数据渲染成整块黑色。

控制台是内嵌在 main.DASHBOARD_HTML 的纯 HTML/JS，当前测试锁住影响用户可见
图表结构的关键契约；浏览器端详图由真实控制台人工/截图验证补充。
"""
from main import DASHBOARD_HTML


def test_dashboard_usage_visualization_uses_dynamic_bars_and_full_date_window():
    """趋势图必须补齐空日期、按真实值更新进度条，并用双系列而非黑色整块。"""
    assert "function buildTrendWindow(days)" in DASHBOARD_HTML
    assert "const slots=buildTrendWindow(TREND_RANGE);" in DASHBOARD_HTML
    assert 'id="t-prompt-bar"' in DASHBOARD_HTML
    assert 'id="t-completion-bar"' in DASHBOARD_HTML
    assert 'class="trend-legend"' in DASHBOARD_HTML
    assert "bg-neutral-800 rounded-t" not in DASHBOARD_HTML


def test_dashboard_health_chart_uses_status_colors_not_black_success_segment():
    """成功分段应使用健康状态绿，不能把成功画成黑色。"""
    assert "const CHART_COLORS={success:'#0ca30c'" in DASHBOARD_HTML
    assert "colors=['#171717'" not in DASHBOARD_HTML


def test_dashboard_request_trend_granularity_switch():
    """请求量趋势：按小时/7 天/30 天三档粒度切换（对齐 sub2api 的 granularity 做法）。"""
    assert "function renderReqTrend(gran)" in DASHBOARD_HTML
    assert "req-gran-hour" in DASHBOARD_HTML
    assert "req-gran-day7" in DASHBOARD_HTML
    assert "req-gran-day30" in DASHBOARD_HTML
    assert "HOURLY=d.hourly||[]" in DASHBOARD_HTML


def test_dashboard_exposes_copyable_independent_channel_routes():
    """通道页必须明确展示三种独立入口，并支持复制当前访问地址。"""
    assert "三种独立渠道接口" in DASHBOARD_HTML
    assert "channel-route-list" in DASHBOARD_HTML
    assert "CHANNEL_ROUTE_META" in DASHBOARD_HTML
    assert "express/v1" in DASHBOARD_HTML or "item.key+'/v1'" in DASHBOARD_HTML
    assert "cookie" in DASHBOARD_HTML and "vertex" in DASHBOARD_HTML
    assert "复制地址" in DASHBOARD_HTML
    assert "navigator.clipboard.writeText" in DASHBOARD_HTML


def test_dashboard_error_categories_panel():
    """错误/拦截分类统计面板：后端 error_categories 驱动，排行展示重复次数。"""
    assert "ERROR_CATS=d.error_categories||{}" in DASHBOARD_HTML
    assert "function renderErrorCats()" in DASHBOARD_HTML
    assert "error-cat-list" in DASHBOARD_HTML


def test_dashboard_top_injection_is_independent_from_input_relay():
    """顶部注入应只原样置顶方案，不能暗含输入模板或自动追加语义。"""
    assert "顶部方案正文（按原文注入，不解析宏）" in DASHBOARD_HTML
    assert "方案正文会原样注入到 messages 第 1 条。" in DASHBOARD_HTML
    assert "未使用 {{input}} 时" not in DASHBOARD_HTML


def test_top_injection_has_binary_switch_and_ternary_random():
    import re
    def values(element_id):
        select = re.search(r'<select id="' + element_id + r'"[^>]*>(.*?)</select>', DASHBOARD_HTML, re.S)
        assert select
        return re.findall(r'<option value="([^"]+)"', select.group(1))
    assert values('top_input_injection_mode') == ['off', 'always']
    assert values('top_input_injection_random') == ['off', 'always', 'non_vertex_only']
    assert "top_input_injection_random:$('top_input_injection_random').value" in DASHBOARD_HTML
    assert "top_input_injection_random:$('top_input_injection_random').checked" not in DASHBOARD_HTML
