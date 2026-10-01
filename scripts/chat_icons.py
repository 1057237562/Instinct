"""Small local SVG icons for chat controls, independent of emoji fonts."""

import base64


_PATHS = {
    'load': '<path d="M12 3v12m-5-5 5 5 5-5M5 17v4h14v-4"/>',
    'unload': '<path d="M12 3v9M6 5a9 9 0 1 0 12 0"/>',
    'folder': '<path d="M3 7V5h6l2 2h10v13H3z"/><path d="M3 10h18"/>',
    'warning': '<path d="m12 3 10 18H2zM12 9v5"/><circle cx="12" cy="17" r=".5"/>',
    'success': '<circle cx="12" cy="12" r="9"/><path d="m7 12 3 3 7-7"/>',
    'error': '<circle cx="12" cy="12" r="9"/><path d="m8 8 8 8m0-8-8 8"/>',
    'tool': '<path d="M14 6a5 5 0 0 0-6 6l-5 5a3 3 0 0 0 4 4l5-5a5 5 0 0 0 6-6l-3 3-4-4z"/>',
    'thinking': '<path d="M20 11a8 8 0 1 0-15 4l-2 6 6-2a8 8 0 0 0 11-8z"/>'
                '<path d="M8 11h.01M12 11h.01M16 11h.01"/>',
    'refresh': '<path d="M20 7v5h-5M4 17v-5h5M5 8a8 8 0 0 1 13-3l2 3M4 16l2 3a8 8 0 0 0 13-3"/>',
    'close': '<path d="m6 6 12 12M6 18 18 6"/>',
    'delete': '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
    'new_chat': '<path d="M21 11a8 8 0 0 1-8 8H7l-4 3V7a4 4 0 0 1 4-4h6M18 2v8m-4-4h8"/>',
}


def svg_icon(name):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" '
            'viewBox="0 0 24 24" fill="none" stroke="#808080" stroke-width="1.8" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            + _PATHS[name] + '</svg>')


def svg_label(name, text):
    payload = base64.b64encode(svg_icon(name).encode('utf-8')).decode('ascii')
    return f'![](data:image/svg+xml;base64,{payload}) {text}'
