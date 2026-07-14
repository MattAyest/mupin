import pytest
from hypothesis import given, settings, strategies as st
from src.main import parse_csv


def test_parse_csv_basic():
    text = "a,b,c\n1,2.5,hello\n3,4.0,world"
    result = parse_csv(text)
    assert result == [
        {"a": 1, "b": 2.5, "c": "hello"},
        {"a": 3, "b": 4.0, "c": "world"},
    ]


def test_parse_csv_empty_input():
    assert parse_csv("") == []


def test_parse_csv_only_header():
    assert parse_csv("a,b,c") == []


def test_parse_csv_type_inference():
    text = "int_field,float_field,str_field\n123,12.34,hello\n-5,-3.14,world"
    result = parse_csv(text)
    assert result[0]["int_field"] == 123
    assert isinstance(result[0]["int_field"], int)
    assert result[0]["float_field"] == 12.34
    assert isinstance(result[0]["float_field"], float)
    assert result[0]["str_field"] == "hello"
    assert result[1]["int_field"] == -5
    assert isinstance(result[1]["int_field"], int)
    assert result[1]["float_field"] == -3.14
    assert isinstance(result[1]["float_field"], float)
    assert result[1]["str_field"] == "world"


def test_parse_csv_first_nonempty_line_is_header():
    text = "\n\na,b\n1,2\n3,4"
    result = parse_csv(text)
    assert result == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_parse_csv_skips_empty_lines():
    text = "\n\na,b\n\n1,2\n\n3,4\n\n"
    result = parse_csv(text)
    assert result == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_parse_csv_field_count_mismatch_fewer_fields():
    with pytest.raises(ValueError):
        parse_csv("a,b,c\n1,2")


def test_parse_csv_field_count_mismatch_extra_fields():
    with pytest.raises(ValueError):
        parse_csv("a,b\n1,2,3")


def test_parse_csv_field_count_mismatch_later_row():
    with pytest.raises(ValueError):
        parse_csv("a,b,c\n1,2,3\n4,5")


@st.composite
def csv_data(draw):
    headers = draw(
        st.lists(
            st.text(
                min_size=1,
                max_size=10,
                alphabet=st.characters(categories=("L", "N")),
            ),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )

    cell = st.one_of(
        st.integers(min_value=-1000, max_value=1000).map(lambda x: ("int", str(x))),
        st.floats(
            allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000
        ).map(lambda x: ("float", f"{x:.4f}")),
        st.text(
            min_size=1,
            max_size=10,
            alphabet=st.characters(categories=("L",)),
        ).map(lambda x: ("str", x)),
    )

    rows = draw(
        st.lists(
            st.lists(cell, min_size=len(headers), max_size=len(headers)),
            min_size=0,
            max_size=10,
        )
    )

    header_line = ",".join(headers)
    row_lines = [",".join(value for _, value in row) for row in rows]
    text = "\n".join([header_line] + row_lines)
    return text, headers, rows


@given(data=csv_data())
@settings(max_examples=50)
def test_parse_csv_row_count_and_keys(data):
    text, headers, rows = data
    result = parse_csv(text)
    assert isinstance(result, list)
    assert len(result) == len(rows)
    for row in result:
        assert isinstance(row, dict)
        assert set(row.keys()) == set(headers)


@given(data=csv_data())
@settings(max_examples=50)
def test_parse_csv_type_inference_properties(data):
    text, headers, rows = data
    result = parse_csv(text)
    for i, row in enumerate(rows):
        for j, (typ, original) in enumerate(row):
            key = headers[j]
            if typ == "int":
                assert result[i][key] == int(original)
                assert isinstance(result[i][key], int)
            elif typ == "float":
                assert abs(result[i][key] - float(original)) < 1e-9
                assert isinstance(result[i][key], float)
            else:
                assert result[i][key] == original
                assert isinstance(result[i][key], str)
