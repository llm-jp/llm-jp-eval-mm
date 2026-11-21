from eval_mm.tasks.jawildtext_handwriting_ocr import _build_reference_from_polygons


def _box(
    x0: float, y0: float, x1: float, y1: float, text: str, direction: str = "horizontal"
) -> dict:
    return {
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "text": text,
        "direction": direction,
    }


class TestBuildReference:
    def test_empty(self):
        assert _build_reference_from_polygons([]) == ""
        assert _build_reference_from_polygons(None) == ""

    def test_horizontal_top_to_bottom(self):
        polygons = [
            _box(10, 200, 500, 240, "三行目"),
            _box(10, 10, 500, 50, "一行目"),
            _box(10, 100, 500, 140, "二行目"),
        ]
        assert _build_reference_from_polygons(polygons) == "一行目\n二行目\n三行目"

    def test_horizontal_same_row_left_to_right(self):
        polygons = [_box(300, 10, 500, 50, "右"), _box(10, 10, 200, 50, "左")]
        assert _build_reference_from_polygons(polygons) == "左\n右"

    def test_vertical_columns_right_to_left(self):
        # A shorter column whose centroid is higher on the page must not come first.
        polygons = [
            _box(100, 10, 140, 900, "三列目", "vertical"),
            _box(300, 10, 340, 300, "二列目", "vertical"),
            _box(500, 10, 540, 900, "一列目", "vertical"),
        ]
        assert _build_reference_from_polygons(polygons) == "一列目\n二列目\n三列目"

    def test_vertical_left_column_starting_lower(self):
        polygons = [
            _box(100, 400, 140, 600, "二列目", "vertical"),
            _box(300, 10, 340, 900, "一列目", "vertical"),
        ]
        assert _build_reference_from_polygons(polygons) == "一列目\n二列目"

    def test_direction_from_geometry_not_label(self):
        # Tall boxes mislabelled as horizontal are still read as vertical columns.
        polygons = [
            _box(100, 10, 140, 900, "二列目", "horizontal"),
            _box(300, 10, 340, 900, "一列目", "horizontal"),
        ]
        assert _build_reference_from_polygons(polygons) == "一列目\n二列目"

    def test_skips_entries_without_text_or_polygon(self):
        polygons = [
            _box(10, 100, 500, 140, "二行目"),
            {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "text": ""},
            {"polygon": [], "text": "位置なし"},
            _box(10, 10, 500, 50, "一行目"),
        ]
        assert _build_reference_from_polygons(polygons) == "一行目\n二行目"
