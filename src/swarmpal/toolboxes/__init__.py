from __future__ import annotations

by_name = {}


def make_toolbox(toolbox_name, config={}):
    """Instantiates a toolbox with a given name and a configuration."""
    if toolbox_name not in by_name:
        if toolbox_name == "FAC_single_sat":
            from swarmpal.toolboxes.fac.processes import FAC_single_sat  # noqua(I001)

            by_name["FAC_single_sat"] = FAC_single_sat

        else:
            raise ValueError(f"Unknown toolbox {toolbox_name}")

    return by_name[toolbox_name](config=config)
