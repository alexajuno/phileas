from phileas.stats.render import spark


def test_spark_empty():
    assert spark([]) == ""


def test_spark_single_value():
    assert spark([5.0]) == "█"


def test_spark_all_zero():
    assert spark([0, 0, 0, 0]) == "▁▁▁▁"


def test_spark_monotone():
    out = spark([1, 2, 3, 4, 5, 6, 7, 8])
    assert len(out) == 8
    blocks = "▁▂▃▄▅▆▇█"
    assert blocks.index(out[0]) <= blocks.index(out[-1])


def test_spark_handles_nan_as_zero():
    out = spark([float("nan"), 1.0, 2.0])
    assert len(out) == 3
