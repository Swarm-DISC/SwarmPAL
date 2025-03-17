from __future__ import annotations

from swarmpal.schema import validate


def test_validate():
    hapi_data = {
        "data_params": [
            {
                "provider": "hapi",
                "dataset": "SW_OPER_MAGA_LR_1B",
                "parameters": "F,B_NEC",
                "start": "2016-01-01T00:00:00",
                "stop": "2016-01-01T00:00:10",
                "server": "https://vires.services/hapi",
                "pad_times": ["0:00:03", "0:00:05"],
            }
        ]
    }

    vires_data = {
        "data_params": [
            {
                "provider": "vires",
                "collection": "example_collection",
                "measurements": ["measurement1", "measurement2"],
                "models": ["model1", "model2"],
                "start_time": "2025-04-10T00:00:00",
                "end_time": "2025-04-11T00:00:00",
                "server_url": "https://example.com",
                "pad_times": ["0:00:03", "0:00:05"],
                "options": {
                    "asynchronous": True,
                    "show_progress": False,
                },
            }
        ]
    }

    assert validate(hapi_data) is None
    assert validate(vires_data) is None
