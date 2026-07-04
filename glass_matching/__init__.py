"""
玻璃匹配模块包
包含材料匹配和玻璃牌号匹配的功能
"""

from .match_material_names import (
    match_names,
    ensure_material_library_exists,
    load_schott_table,
)

from .verify_glass_name_matching import (
    GlassLibrary,
    process_prediction_csv,
    process_test_output_csv,
)

__all__ = [
    # Material name matching
    'match_names',
    'ensure_material_library_exists',
    'load_schott_table',

    # Glass library and matching
    'GlassLibrary',
    'process_prediction_csv',
    'process_test_output_csv',
]
