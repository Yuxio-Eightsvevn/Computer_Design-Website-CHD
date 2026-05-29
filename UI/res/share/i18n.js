/**
 * i18n 文本配置化加载器
 * 用法：
 *   HTML 元素加 data-i18n="page.key" 属性（保留原始文本作为 fallback）
 *   input 加 data-i18n-placeholder="page.key" 属性
 *   <html> 标签加 data-i18n-title="page.page_title" 属性
 *   页面初始化时调用 await i18n.load('zh')
 *   JS 动态文本用 i18n.t('page.key') 获取
 */
window.i18n = (function () {
    var _data = {};
    var _lang = 'zh';

    function _get(key) {
        var parts = key.split('.');
        var val = _data;
        for (var i = 0; i < parts.length; i++) {
            if (val == null) return key;
            val = val[parts[i]];
        }
        return (val != null && val !== '') ? val : key;
    }

    function _apply() {
        // 替换 textContent
        document.querySelectorAll('[data-i18n]').forEach(function (el) {
            var key = el.getAttribute('data-i18n');
            var attr = el.getAttribute('data-i18n-attr');
            if (attr) {
                el.setAttribute(attr, _get(key));
            } else {
                el.textContent = _get(key);
            }
        });
        // 替换 placeholder
        document.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
            el.placeholder = _get(el.getAttribute('data-i18n-placeholder'));
        });
        // 替换 <title>
        var titleKey = document.documentElement.getAttribute('data-i18n-title');
        if (titleKey) document.title = _get(titleKey);
    }

    return {
        /** 加载指定语言并应用到页面 */
        load: async function (lang) {
            _lang = lang || 'zh';
            try {
                var r = await fetch('/res/share/i18n/' + _lang + '.json', { cache: 'no-cache' });
                if (!r.ok) throw new Error('HTTP ' + r.status);
                _data = await r.json();
                _apply();
            } catch (e) {
                console.warn('[i18n] 加载失败，使用 HTML 原始文本:', e);
            }
        },
        /** 获取文本，找不到时返回 key 本身 */
        t: function (key) {
            return _get(key);
        },
        /** 手动重新应用（动态插入 DOM 后调用） */
        apply: _apply,
        /** 当前语言 */
        get lang() { return _lang; }
    };
})();
