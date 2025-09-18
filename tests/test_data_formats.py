import pytest
import torch

from quant_mp.datatypes.float import FloatDataFormat
from quant_mp.datatypes.int import UniformDataFormat
from quant_mp.datatypes.template import get_data_format


class TestUniformDataFormat:
    @pytest.mark.parametrize("format_name", ["int4", "int8"])
    def test_uniform_properties(self, format_name):
        data_format = get_data_format(format_name)
        assert isinstance(data_format, UniformDataFormat)
        assert data_format.signed is True

        # Test basic properties
        if format_name == "int4":
            assert data_format.max_value == 7.0
            assert data_format.min_value == -8.0
            assert data_format.n_values == 16
        elif format_name == "int8":
            assert data_format.max_value == 127.0
            assert data_format.min_value == -128.0
            assert data_format.n_values == 256

        # Test representable values
        values = data_format.get_representable_values()
        assert len(values) == data_format.n_values
        assert torch.min(values).item() == data_format.min_value
        assert torch.max(values).item() == data_format.max_value

    @pytest.mark.parametrize("format_name", ["int4", "int8"])
    def test_uniform_cast(self, format_name):
        data_format = get_data_format(format_name)

        # Test rounding
        x = torch.tensor([1.2, 2.7, -3.4, -4.6, 5.5, -2.5])
        casted = data_format.cast(x)
        expected = torch.tensor([1.0, 3.0, -3.0, -5.0, 6.0, -2])
        assert torch.all(torch.eq(casted, expected))

        # Test clamping
        big_values = torch.tensor([1000, -1000])
        casted = data_format.cast(big_values)
        assert torch.all(casted <= data_format.max_value)
        assert torch.all(casted >= data_format.min_value)

    @pytest.mark.parametrize("format_name", ["int4", "int8"])
    def test_uniform_output_mask(self, format_name):
        data_format = get_data_format(format_name)

        # In-range and out-of-range values
        x = torch.linspace(data_format.min_value, data_format.max_value, 10)
        x = torch.cat(
            [
                x,
                torch.tensor(
                    [
                        data_format.max_value + 0.5,
                        data_format.min_value - 0.5,
                        float("inf"),
                    ]
                ),
            ]
        )
        mask = data_format.get_output_mask(x)
        expected = torch.tensor([True] * 10 + [False] * 3)
        assert torch.all(torch.eq(mask, expected))


class TestFloatDataFormat:
    @pytest.mark.parametrize(
        "format_name", ["fp4_e2m1", "fp4_e3m0", "fp8_e4m3", "fp8_e5m2"]
    )
    def test_float_properties(self, format_name):
        data_format = get_data_format(format_name)
        assert isinstance(data_format, FloatDataFormat)

        # Test basic properties
        if "fp4" in format_name:
            assert data_format.bit_width == 4
        elif "fp8" in format_name:
            assert data_format.bit_width == 8

        assert data_format.signed is True
        assert hasattr(data_format, "exponent")
        assert hasattr(data_format, "mantissa")

        # Check bias computation
        if format_name == "fp4_e2m1":
            assert data_format.exponent == 2
            assert data_format.mantissa == 1
            assert data_format.bias == 1
        elif format_name == "fp4_e3m0":
            assert data_format.exponent == 3
            assert data_format.mantissa == 0
            assert data_format.bias == 3
        elif format_name == "fp8_e4m3":
            assert data_format.exponent == 4
            assert data_format.mantissa == 3
            assert data_format.bias == 7
        elif format_name == "fp8_e5m2":
            assert data_format.exponent == 5
            assert data_format.mantissa == 2
            assert data_format.bias == 15

    @pytest.mark.parametrize(
        "format_name",
        [
            "fp4_e2m1",
            "fp4_e3m0",
            "fp8_e4m3",
            "fp8_e4m3fnuz",
            "fp8_e5m2",
        ],
    )
    def test_float_cast(self, format_name):
        data_format = get_data_format(format_name)
        assert isinstance(data_format, FloatDataFormat)

        # Test casting with values that should have torch equivalent
        x = torch.linspace(-1e10, 1e10, 10000)
        casted = data_format.cast(x)
        assert torch.all(casted <= data_format.max_value)
        assert torch.all(casted >= data_format.min_value)
        x = torch.linspace(data_format.min_value, data_format.max_value, 10000)
        casted = data_format.cast(x)

        # Compare with torch's native casting
        if data_format.torch_equivalent is not None:
            assert torch.allclose(
                torch.tensor(data_format.max_value),
                torch.tensor(torch.finfo(data_format.torch_equivalent).max),
                rtol=1e-3,
                atol=1e-3,
            )
            assert torch.allclose(
                torch.tensor(data_format.min_value),
                torch.tensor(torch.finfo(data_format.torch_equivalent).min),
                rtol=1e-3,
                atol=1e-3,
            )
            torch_casted = x.to(data_format.torch_equivalent).to(torch.float32)
            # Should be close but not necessarily identical due to rounding differences
            assert torch.allclose(casted, torch_casted, rtol=1e-3, atol=1e-3)
        else:
            representable_values_set = set(
                data_format.get_representable_values().tolist()
            )
            if data_format.nan:
                representable_values_set.add(float("nan"))
            if data_format.inf:
                representable_values_set.add(float("inf"))
                representable_values_set.add(float("-inf"))
            assert all(item in representable_values_set for item in casted.tolist())

    @pytest.mark.parametrize(
        "format_name",
        [
            "fp4_e2m1",
            "fp4_e3m0",
            "fp8_e4m3",
            "fp8_e4m3fnuz",
            "fp8_e5m2",
        ],
    )
    def test_float_output_mask(self, format_name):
        data_format = get_data_format(format_name)

        # In-range and out-of-range values
        x = torch.linspace(data_format.min_value, data_format.max_value, 10)
        x = torch.cat(
            [
                x,
                torch.tensor(
                    [
                        data_format.max_value + 0.5,
                        data_format.min_value - 0.5,
                        float("inf"),
                    ]
                ),
            ]
        )
        mask = data_format.get_output_mask(x)
        expected = torch.tensor([True] * 10 + [False] * 3)
        assert torch.all(torch.eq(mask, expected))
