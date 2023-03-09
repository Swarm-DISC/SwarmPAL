import pytest
from datatree import DataTree
from xarray import Dataset

from swarmpal.io._paldata import PalDataItem, PalProcess, create_paldata


@pytest.mark.remote
@pytest.fixture
def paldata_MAGA():
    data_params = dict(
        collection="SW_OPER_MAGA_LR_1B",
        measurements=["B_NEC"],
        models=["IGRF"],
        start_time="2016-01-01T00:00:00",
        end_time="2016-01-01T00:01:00",
        server_url="https://vires.services/ows",
        options=dict(asynchronous=False, show_progress=False),
    )
    data = create_paldata(PalDataItem.from_vires(**data_params))
    return data


@pytest.mark.remote
def test_palprocess(paldata_MAGA):
    """Test the creation and use of a basic PalProcess"""
    data = paldata_MAGA

    class MyProcess(PalProcess):
        """Compute the first differences on a given variable"""

        @property
        def process_name(self):
            return "MyProcess"

        def _call(self, datatree):
            # Identify inputs for algorithm
            subtree = datatree[f"{self.config.get('dataset')}"]
            dataset = subtree.ds
            parameter = self.config.get("parameter")
            # Apply the algorithm
            output_data = dataset[parameter].diff(dim="Timestamp")
            # Create an output dataset
            data_out = Dataset(
                data_vars={
                    "output_parameter": output_data,
                }
            )
            # Write the output into a new path in the datatree and return it
            subtree["output"] = DataTree(data=data_out)
            return datatree

    p = MyProcess(config={"dataset": "SW_OPER_MAGA_LR_1B", "parameter": "B_NEC"})
    data = data.swarmpal.apply(p)
    assert "output_parameter" in data["SW_OPER_MAGA_LR_1B/output"]
    assert "MyProcess" in data.swarmpal.pal_meta["."].keys()
