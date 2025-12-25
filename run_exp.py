from EasyTSAD.Controller import TSADController
from GMB_KAN import GMB_KAN
if __name__ == "__main__":
    # Create a global controller
    gctrl = TSADController()

    # datasets = ["TODS", "AIOPS", "NAB", "Yahoo", "WSD"]
    datasets = ["UCR"]
    dataset_types = "UTS"

    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="datasets",
        datasets=datasets,
    )

    training_schema = "naive"
    method = "GMB_KAN"  # string of your algo class

    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        cfg_path="GMB_KAN/config.toml",
    )

    """============= [EVALUATION SETTINGS] ============="""

    from EasyTSAD.Evaluations.Protocols import (
        EventF1PA,
        PointF1PA,
        PointKthF1PA,
        PointAuprcPA,
    )

    # Specifying evaluation protocols
    gctrl.set_evals(
        [PointF1PA(), EventF1PA(mode="squeeze"), PointKthF1PA(k=5), PointAuprcPA()]
    )

    gctrl.do_evals(method=method, training_schema=training_schema)
