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
