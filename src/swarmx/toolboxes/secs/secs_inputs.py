from swarmx.io import ExternalData


class SecsInputSingleSat(ExternalData):
    """Accessing external data: magnetic data from one satellite"""

    COLLECTIONS = [
        *[f"SW_OPER_MAG{x}_LR_1B" for x in "ABC"],
    ]

    DEFAULTS = {
        "measurements": ["B_NEC", "Flags_B"],
        "model": "'CHAOS-Core' + 'CHAOS-Static'",
        "auxiliaries": ["QDLat", "QDLon", "MLT"],
        "sampling_step": None,
    }


class SecsInputs:
    """Accessing external data: inputs to SECS algorithm

    Examples
    --------
    >>> d = SecsInputs(
    >>>     start_time="2022-01-01", end_time="2022-01-02",
    >>>     model="'CHAOS-Core' + 'CHAOS-Static'",
    >>>     viresclient_kwargs=dict(asynchronous=True, show_progress=True)
    >>> )
    >>> d.s1.xarray  # Returns xarray of data from satellite 1 (Alpha)
    >>> d.s2.xarray  # Returns xarray of data from satellite 2 (Charlie)
    >>> d.s1.get_array("B_NEC")  # Returns numpy array
    """

    def __init__(
        self,
        spacecraft_pair="Alpha-Charlie",
        start_time=None,
        end_time=None,
        model="'CHAOS-Core' + 'CHAOS-Static'",
        viresclient_kwargs=None,
    ):
        if spacecraft_pair != "Alpha-Charlie":
            raise NotImplementedError("Only the Alpha-Charlie pair are configured")
        inputs_1 = SecsInputSingleSat(
            collection="SW_OPER_MAGA_LR_1B",
            model=model,
            start_time=start_time,
            end_time=end_time,
            viresclient_kwargs=viresclient_kwargs,
        )
        inputs_2 = SecsInputSingleSat(
            collection="SW_OPER_MAGC_LR_1B",
            model=model,
            start_time=start_time,
            end_time=end_time,
            viresclient_kwargs=viresclient_kwargs,
        )
        self.s1 = inputs_1
        self.s2 = inputs_2
